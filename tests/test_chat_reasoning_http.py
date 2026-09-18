"""Exercise the deployed HTTP/middleware/thread/graph path without remote calls."""
import importlib
import json
import re
import socket
from threading import Barrier, Event, Thread, get_ident
from time import monotonic, sleep
from types import SimpleNamespace

import httpx
from langchain_core.messages import AIMessageChunk
import pytest
import uvicorn

from app.core.config import settings
from app.graphs import multi_agent_graph as graph
from app.llm import client as llm
from app.tools import native_registry, tool_registry
from app.main import app

chat = importlib.import_module("app.api.chat")


@pytest.mark.parametrize("has_reasoning,interleaved_tool", [(True, False), (False, False), (True, True)])
def test_http_weather_graph_progress_and_optional_reasoning_reach_java_contract(monkeypatch, has_reasoning, interleaved_tool):
    # The display holdback includes configured opaque credentials. Clear local
    # settings so developer/production .env values cannot affect test timing.
    for name in type(settings).model_fields:
        if re.search(r"api_key|secret|password|token", name, re.I) and isinstance(getattr(settings, name), str):
            monkeypatch.setattr(settings, name, "")
    for field in ("postgres_enabled", "redis_enabled", "qdrant_enabled", "milvus_enabled",
                  "reviewer_enabled", "ai_usage_reporting_enabled"):
        monkeypatch.setattr(settings, field, False)
    monkeypatch.setattr(settings, "internal_api_token", "offline-http-reasoning-test-token")
    monkeypatch.setattr(settings, "python_rate_limit_per_minute", 0)
    monkeypatch.setattr("app.streaming_response.HEARTBEAT_SECONDS", 0.02)
    monkeypatch.setattr(chat, "_project_context_for_response", lambda *args: {})
    monkeypatch.setattr(chat, "extract_project_candidates", lambda *args: {
        "candidateGoals": [], "candidateDecisions": [], "candidateOpenQuestions": []})
    monkeypatch.setattr(chat, "build_plan_candidate", lambda *args: None)
    monkeypatch.setattr(chat, "_prepare_candidate_memories_for_response", lambda *args: [])
    monkeypatch.setattr(chat, "build_execution_suggestions", lambda *args: [])
    monkeypatch.setattr(chat, "run_reviewer_agent", lambda *args, **kwargs: {"approved": True})
    saved = []
    memory_written = Event()

    def write_memory(*args):
        saved.append(args[3])
        memory_written.set()

    monkeypatch.setattr(chat, "write_memory_in_background", write_memory)
    monkeypatch.setattr(graph, "run_supervisor_agent", lambda *args: {
        "intent": "tool", "use_memory": True, "use_rag": False,
        "use_tools": True, "next_agents": ["memory_agent", "tool_agent"]})
    branch_barrier = Barrier(2)
    worker_ids = {}
    release_weather = Event()

    def weather_result(*args):
        assert release_weather.wait(5), "Tool running event was not visible during the real HTTP request"
        return {"ok": True, "content": "上海今天晴，26 摄氏度。"}

    monkeypatch.setattr("app.tools.weather_tool.get_weather", weather_result)
    monkeypatch.setattr(tool_registry, "list_available_tools", lambda: [{"name": "weather", "enabled": True}])
    monkeypatch.setattr(tool_registry, "audit_tool_call", lambda **kwargs: None)

    def memory(*args):
        worker_ids["memory"] = get_ident()
        branch_barrier.wait(timeout=5)
        return {"memory": {}, "memory_context": ""}

    def weather(*args, **kwargs):
        worker_ids["tool"] = get_ident()
        branch_barrier.wait(timeout=5)
        result = native_registry.execute_native("weather", {"city": "上海"}, "offline-user", "offline-conversation", "天气")
        return {"selected_tools": ["weather"], "tool_results": [result], "tool_context": "上海今天晴，26 摄氏度。"}

    monkeypatch.setattr(graph, "run_memory_agent", memory)
    monkeypatch.setattr(graph, "run_tool_agent", weather)
    monkeypatch.setattr(llm, "is_chat_model_configured", lambda: True)
    monkeypatch.setattr(llm, "get_system_prompt", lambda: ("offline test", "test"))
    release_model = Event()
    answer = "上海今天晴，气温约 26 摄氏度。" * 12
    provider_reasoning = "我会根据天气工具返回的信息整理回答。" * 12
    followup_reasoning = "补充查询已经返回，可以整理最后的结论。" * 12 if interleaved_tool else ""

    def model_stream(messages):
        worker_ids["writer"] = get_ident()
        if has_reasoning:
            yield AIMessageChunk(content="", additional_kwargs={"reasoning_content": provider_reasoning})
        assert release_model.wait(5), "HTTP client did not receive progress before the model completed"
        if interleaved_tool:
            # A mock provider seam exercises interleaving in the real HTTP
            # transport. Normal production graph tools precede writer streaming.
            native_registry.execute_native("weather", {"city": "上海"}, "offline-user", "offline-conversation", "天气")
            yield AIMessageChunk(content="", additional_kwargs={"reasoning_content": followup_reasoning})
        yield AIMessageChunk(content=answer)

    monkeypatch.setattr(llm, "build_chat_task_model", lambda **kwargs: SimpleNamespace(stream=model_stream))
    server_socket = socket.socket()
    server_socket.bind(("127.0.0.1", 0))
    server_socket.listen(16)
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off", ws="none"))
    server_thread = Thread(target=lambda: server.run(sockets=[server_socket]), daemon=True)
    server_thread.start()
    frames = []
    try:
        deadline = monotonic() + 5
        while not server.started and monotonic() < deadline:
            sleep(0.01)
        assert server.started
        with httpx.Client(timeout=10, trust_env=False) as http:
            with http.stream("POST", f"http://127.0.0.1:{server_socket.getsockname()[1]}/ai/chat/stream",
                    headers={"X-Internal-Token": "offline-http-reasoning-test-token"},
                    json={"userId": "offline-user", "conversationId": "offline-conversation",
                          "turnId": "offline-http-turn", "message": "上海今天天气怎么样？", "stream": True}) as response:
                assert response.status_code == 200
                assert response.headers["content-type"].startswith("text/event-stream")
                assert response.headers["x-accel-buffering"] == "no"
                name = ""
                data = ""
                for line in response.iter_lines():
                    if line.startswith("event: "):
                        name = line[7:]
                    elif line.startswith("data: "):
                        data += line[6:]
                    elif not line and data:
                        payload = json.loads(data)
                        frames.append((name, payload))
                        if name == "tool.call" and payload["status"] == "running":
                            release_weather.set()
                        if (has_reasoning and name == "reasoning.delta") or (
                            not has_reasoning and name == "agent.step" and payload["step"] == "writer_agent"
                            and payload["status"] == "running"
                        ):
                            release_model.set()
                        name, data = "", ""
        assert memory_written.wait(3)
    finally:
        release_weather.set()
        release_model.set()
        server.should_exit = True
        server_thread.join(timeout=5)
        server_socket.close()
    assert not server_thread.is_alive()
    assert worker_ids["memory"] != worker_ids["tool"]
    assert worker_ids["writer"] not in {worker_ids["memory"], worker_ids["tool"]}
    progress = [data for name, data in frames if name == "agent.step"]
    assert [data["sequence"] for data in progress] == list(range(1, len(progress) + 1))
    assert all(data["turnId"] == "offline-http-turn" for data in progress)
    assert [(data["step"], data["status"]) for data in progress if data["step"] == "tool_agent"] == [
        ("tool_agent", "running"), ("tool_agent", "completed")]
    reasoning = [data for name, data in frames if name == "reasoning.delta"]
    if has_reasoning:
        assert "".join(data["delta"] for data in reasoning) == provider_reasoning + followup_reasoning
        assert [data["sequence"] for data in reasoning] == list(range(1, len(reasoning) + 1))
        assert all(set(data) == {"turnId", "sequence", "step", "delta", "kind"}
                   and data["turnId"] == "offline-http-turn" and data["step"] == "writer_agent"
                   and data["kind"] == "reasoning" for data in reasoning)
    else:
        assert not reasoning
    assert "".join(data["text"] for name, data in frames if name == "delta") == answer
    assert [name for name, data in frames][-2:] == ["meta", "done"]
    assert saved == [answer]
    calls = [data for name, data in frames if name == "tool.call"]
    count = 2 if interleaved_tool else 1
    assert [data["status"] for data in calls] == ["running", "completed"] * count
    assert [data["sequence"] for data in calls] == list(range(1, count * 2 + 1))
    assert calls[0]["callId"] == calls[1]["callId"]
    assert all(data["turnId"] == "offline-http-turn" and data["tool"] == "weather" for data in calls)
    assert "上海" not in json.dumps(calls, ensure_ascii=False)
    if interleaved_tool:
        reasoning_positions = [index for index, (name, _) in enumerate(frames) if name == "reasoning.delta"]
        tool_positions = [index for index, (name, _) in enumerate(frames) if name == "tool.call"]
        assert reasoning_positions[0] < tool_positions[2] < tool_positions[3] < reasoning_positions[1]
        assert calls[2]["callId"] == calls[3]["callId"] != calls[0]["callId"]
