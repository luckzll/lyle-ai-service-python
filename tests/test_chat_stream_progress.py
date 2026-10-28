"""Observe frames before completion; TestClient would hide buffering bugs."""
import asyncio
import importlib
import json
from threading import Event

import pytest
import httpx
from openai import PermissionDeniedError

from app.chat_progress import emit_step
from app.chat_reasoning import emit_reasoning, finish_reasoning
from app.core.config import settings
from app.schemas.chat import ChatRequest
from app.streaming_text import IncrementalAnswer, IncrementalReviewBlocked, streaming_policy

chat = importlib.import_module("app.api.chat")


def event(frame):
    name, data = "", ""
    for line in frame.splitlines():
        if line.startswith("event: "):
            name = line[7:]
        if line.startswith("data: "):
            data += line[6:]
    return name, json.loads(data) if data else {}


async def collect(response, observer=None):
    result = []
    async for raw in response.body_iterator:
        frame = raw.decode() if isinstance(raw, bytes) else raw
        parsed = event(frame)
        result.append(parsed)
        if observer:
            observer(*parsed)
    return result


@pytest.fixture
def stream_env(monkeypatch):
    for field in ("postgres_enabled", "redis_enabled", "qdrant_enabled", "milvus_enabled", "reviewer_enabled"):
        monkeypatch.setattr(settings, field, False)
    monkeypatch.setattr(settings, "reviewer_max_retry", 0)
    monkeypatch.setattr("app.streaming_response.HEARTBEAT_SECONDS", 0.02)
    state = {"contexts": [], "memory": {}, "rag_sufficiency": {}, "agent_trace": []}
    monkeypatch.setattr(chat, "build_turn_attachment_context", lambda request: ("", []))
    monkeypatch.setattr(chat, "_stream_image_tool", lambda message: None)
    monkeypatch.setattr(chat, "prepare_multi_agent_context", lambda *args: state)
    monkeypatch.setattr(chat, "_project_context_for_response", lambda *args: {})
    monkeypatch.setattr(chat, "extract_project_candidates", lambda *args: {
        "candidateGoals": [], "candidateDecisions": [], "candidateOpenQuestions": []})
    monkeypatch.setattr(chat, "build_plan_candidate", lambda *args: None)
    monkeypatch.setattr(chat, "_prepare_candidate_memories_for_response", lambda *args: [])
    monkeypatch.setattr(chat, "build_execution_suggestions", lambda *args: [])
    saved = []
    monkeypatch.setattr(chat, "write_memory_in_background", lambda *args: saved.append(args[3]))
    return state, saved


def request(**kwargs):
    return ChatRequest(userId="progress-user", conversationId="progress-conversation", turnId="progress-turn",
                       message="解释流式响应的工作原理", **kwargs)


def test_prepare_reports_running_and_heartbeat_before_work_finishes(monkeypatch, stream_env):
    release = Event()
    state, _ = stream_env

    def prepare(*args):
        emit_step("rag_agent", "running", summary="正在检索知识库")
        assert release.wait(3), "consumer could not see progress while preparation was running"
        emit_step("rag_agent", "completed", duration_ms=21, timings={"vector_search_ms": 12})
        return state

    monkeypatch.setattr(chat, "prepare_multi_agent_context", prepare)
    monkeypatch.setattr(chat, "stream_answer", lambda *args: iter(["流式响应分段传输正文。 "]))
    running = False

    def observe(name, data):
        nonlocal running
        if name == "agent.step" and data["step"] == "rag_agent" and data["status"] == "running":
            running = True
        # Require an idle heartbeat *after* running, not the initial heartbeat.
        if not name and running:
            release.set()

    try:
        events = asyncio.run(collect(chat.chat_stream(request()), observe))
    finally:
        release.set()
    assert running and any(name == "done" for name, _ in events)
    progress = [data for name, data in events if name == "agent.step"]
    assert [item["sequence"] for item in progress] == list(range(1, len(progress) + 1))
    assert all(item["turnId"] == "progress-turn" for item in progress)
    assert next(item for item in progress if item["step"] == "rag_agent" and item["status"] == "completed")["timings"] == {"vector_search_ms": 12}


@pytest.mark.parametrize(("project_context", "status", "summary"), [
    ({"resolver": {"projectId": None, "requiresClarification": False}, "context": {}},
     "skipped", "未匹配到项目，继续使用本轮对话上下文"),
    ({}, "skipped", "本轮未取得项目识别结果"),
    ({"resolver": {"projectId": 7, "requiresClarification": True}, "context": {"summary": "private project"}},
     "completed", "项目归属仍需确认，暂未采用项目上下文"),
    ({"resolver": {"projectId": 7, "requiresClarification": False}, "context": {"summary": "private project"}},
     "completed", "已识别关联项目并取得项目上下文"),
    ({"resolver": {"projectId": 7}, "context": {"error": "private failure"}},
     "failed", "已识别关联项目，项目上下文暂时不可用"),
    ({"resolver": {"projectId": None, "reason": "project_store_unavailable:private failure"}, "context": {}},
     "failed", "项目资料暂时不可用，继续使用本轮对话上下文"),
    ({"resolver": {"projectId": 7}, "context": {}},
     "completed", "已识别关联项目，未返回项目上下文"),
])
def test_project_feedback_uses_actual_resolution_instead_of_outer_object_truthiness(
    monkeypatch, stream_env, project_context, status, summary,
):
    monkeypatch.setattr(chat, "should_resolve_project", lambda *_: True)
    monkeypatch.setattr(chat, "_project_context_for_response", lambda *_: project_context)
    monkeypatch.setattr(chat, "stream_answer", lambda *_: iter(["当前回答。 "]))
    events = asyncio.run(collect(chat.chat_stream(request())))
    project_events = [data for name, data in events if name == "agent.step" and data["step"] == "project_resolver"]
    assert [item["status"] for item in project_events] == ["running", status]
    assert project_events[-1]["summary"] == summary
    assert "private" not in json.dumps(project_events)
    assert any(name == "done" for name, _ in events)


def test_plain_question_skips_project_resolution_and_marks_rule_review_truthfully(monkeypatch, stream_env):
    monkeypatch.setattr(chat, "should_resolve_project", lambda *_: False)
    def unexpected_resolution(*_):
        raise AssertionError("plain question should not scan projects")
    monkeypatch.setattr(chat, "_project_context_for_response", unexpected_resolution)
    monkeypatch.setattr(chat, "stream_answer", lambda *_: iter(["流式响应逐段传输内容。" * 20]))
    events = asyncio.run(collect(chat.chat_stream(request())))
    project_events = [data for name, data in events if name == "agent.step" and data["step"] == "project_resolver"]
    assert [item["status"] for item in project_events] == ["skipped"]
    reviews = [data for name, data in events if name == "agent.step" and data["step"] == "reviewer_agent"]
    assert reviews[-1]["status"] == "skipped"
    assert reviews[-1]["summary"] == "本地规则检查完成，本轮无需模型复核"
    assert any(name == "done" for name, _ in events)


def test_real_model_review_is_distinguished_from_local_rules(monkeypatch, stream_env):
    monkeypatch.setattr(settings, "reviewer_enabled", True)
    monkeypatch.setattr(chat, "stream_answer", lambda *_: iter(["经过复核的回答。 "]))
    monkeypatch.setattr(chat, "run_reviewer_agent", lambda *_, **__: {
        "approved": True, "skipped": False, "modelReviewUsed": True,
    })
    events = asyncio.run(collect(chat.chat_stream(request())))
    reviews = [data for name, data in events if name == "agent.step" and data["step"] == "reviewer_agent"]
    assert reviews[-1]["status"] == "completed"
    assert reviews[-1]["summary"] == "模型复核完成"


def test_plain_answer_delta_arrives_before_model_finishes_and_matches_saved_answer(monkeypatch, stream_env):
    release = Event()
    _, saved = stream_env
    text = "流式响应会逐段传输生成的文字，让页面在完整回答生成前就能展示内容。" * 5

    def model(*args):
        yield "  " + text
        assert release.wait(3), "first delta was buffered until model completion"
        yield "结束。  "

    monkeypatch.setattr(chat, "stream_answer", model)
    response = chat.chat_stream(request())
    try:
        events = asyncio.run(collect(response, lambda name, data: release.set() if name == "delta" else None))
    finally:
        release.set()
    assert [name for name, _ in events][-2:] == ["meta", "done"]
    deltas = [data["text"] for name, data in events if name == "delta"]
    assert len(deltas) >= 2
    assert "".join(deltas) == text + "结束。"
    asyncio.run(response.background())
    assert saved == [text + "结束。"]
    meta = next(data for name, data in events if name == "meta")
    writer = next(item for item in meta["agentTrace"] if item["agent"] == "writer_agent")
    assert writer["elapsedMs"] >= 0
    assert writer["timings"]["first_delta_ms"] <= writer["timings"]["total_ms"]


def test_reasoning_arrives_while_model_runs_and_is_separate_from_saved_answer(monkeypatch, stream_env):
    release = Event()
    _, saved = stream_env

    def model(*args):
        emit_reasoning("先检查相关资料，再整理回答。" * 20)
        assert release.wait(3), "reasoning was buffered until the answer finished"
        finish_reasoning()
        yield "正文只包含回答。"

    monkeypatch.setattr(chat, "stream_answer", model)
    response = chat.chat_stream(request())
    try:
        events = asyncio.run(collect(response, lambda name, data: release.set() if name == "reasoning.delta" else None))
    finally:
        release.set()
    reasoning = [data for name, data in events if name == "reasoning.delta"]
    assert reasoning and all(item["turnId"] == "progress-turn" for item in reasoning)
    assert all(item["step"] == "writer_agent" and item["kind"] == "reasoning" for item in reasoning)
    assert [item["sequence"] for item in reasoning] == list(range(1, len(reasoning) + 1))
    assert "".join(data["text"] for name, data in events if name == "delta") == "正文只包含回答。"
    assert any(name == "done" for name, _ in events)
    asyncio.run(response.background())
    assert saved == ["正文只包含回答。"]


@pytest.mark.parametrize("gate", ["rag", "model_review"])
def test_whole_answer_gates_keep_text_buffered(monkeypatch, stream_env, gate):
    state, _ = stream_env
    if gate == "rag":
        state["contexts"] = [{"doc_id": "doc", "title": "流式说明", "chunk": "分段传输"}]
    else:
        monkeypatch.setattr(settings, "reviewer_enabled", True)
    seen_review = []
    monkeypatch.setattr(chat, "run_reviewer_agent", lambda *args, **kwargs: seen_review.append(kwargs) or {"approved": True})
    finished = Event()
    text = "分段传输。" * 30

    def model(*args):
        yield text
        yield "完毕。"
        finished.set()

    def observe(name, data):
        if name == "delta":
            assert finished.is_set() and seen_review

    monkeypatch.setattr(chat, "stream_answer", model)
    events = asyncio.run(collect(chat.chat_stream(request()), observe))
    assert [data["text"] for name, data in events if name == "delta"] == [text + "完毕。"]
    assert len(seen_review) == 1
    assert seen_review[0]["model_review_enabled"] is (gate == "model_review")


def test_producer_failure_marks_active_step_and_never_emits_done(monkeypatch, stream_env):
    def prepare(*args):
        emit_step("rag_agent", "running")
        raise RuntimeError("sensitive-provider-secret")

    monkeypatch.setattr(chat, "prepare_multi_agent_context", prepare)
    events = asyncio.run(collect(chat.chat_stream(request())))
    assert any(name == "agent.step" and data["step"] == "rag_agent" and data["status"] == "failed" for name, data in events)
    assert any(name == "error" for name, _ in events)
    assert not any(name in {"done", "meta"} for name, _ in events)
    assert "sensitive-provider-secret" not in json.dumps(events)


def test_model_permission_denial_reports_actionable_error_without_provider_secrets(monkeypatch, stream_env):
    def denied(*args):
        raise PermissionDeniedError(
            "sensitive-provider-secret api_key_source: payg",
            response=httpx.Response(403, request=httpx.Request("POST", "https://provider.test")),
            body={"error": {"type": "access_denied"}},
        )
        yield

    monkeypatch.setattr(chat, "stream_answer", denied)
    events = asyncio.run(collect(chat.chat_stream(request())))
    message = next(data["message"] for name, data in events if name == "error")
    assert "403" in message and "切换其他模型" in message
    assert "sensitive-provider-secret" not in json.dumps(events)
    assert "api_key_source" not in json.dumps(events)
    assert not any(name == "done" for name, _ in events)


def test_unreadable_page_returns_guidance_even_when_selected_model_is_unavailable(monkeypatch, stream_env):
    state, _ = stream_env
    url = "https://www.xiaohongshu.com/user/profile/example"
    state["tool_results"] = [{"tool": "web_search", "ok": False, "pageUrl": url,
                              "error": {"code": "PAGE_CONTENT_UNAVAILABLE"}}]
    monkeypatch.setattr(settings, "reviewer_enabled", True)
    monkeypatch.setattr(chat, "should_resolve_project", lambda *_: True)
    monkeypatch.setattr(chat, "_project_context_for_response", lambda *_: {
        "resolver": {"projectId": None, "requiresClarification": False}, "context": {},
    })
    monkeypatch.setattr(chat, "stream_answer", lambda *args: pytest.fail("no page source to analyze"))
    monkeypatch.setattr("app.agents.reviewer_agent.invoke_reviewer_chain",
                        lambda *args: pytest.fail("fixed guidance must not require a model"))
    req = request()
    req.message = url + "\n你能分析一下我的小红书内容吗？"
    req.aiConfig = {"CHAT_MODEL_PROVIDER": "openai", "CHAT_MODEL_NAME": "openai/gpt-5.5"}
    events = asyncio.run(collect(chat.chat_stream(req)))
    answer = "".join(data["text"] for name, data in events if name == "delta")
    assert "未能读取" in answer and "上传" in answer and "小红书" in answer
    assert any(name == "done" for name, _ in events)
    assert not any(name == "error" for name, _ in events)


def test_incremental_review_never_calls_model_even_if_global_switch_changes(monkeypatch, stream_env):
    calls = []
    monkeypatch.setattr("app.agents.reviewer_agent.invoke_reviewer_chain", lambda *args: calls.append(args))

    def model(*args):
        monkeypatch.setattr(settings, "reviewer_enabled", True)
        yield "这段回答应该始终只走当前请求选择的本地规则检查。" * 10

    monkeypatch.setattr(chat, "stream_answer", model)
    events = asyncio.run(collect(chat.chat_stream(request())))
    assert not calls
    assert any(name == "done" for name, _ in events)


def test_rejected_incremental_chunk_keeps_visible_prefix_but_never_completes(monkeypatch, stream_env):
    _, saved = stream_env

    def model(*args):
        yield "已确认的项目资料应单独管理。" * 12
        # Existing grounding rule disallows treating a user profile as project evidence.
        yield "用户画像就是项目事实。"

    monkeypatch.setattr(chat, "stream_answer", model)
    response = chat.chat_stream(request())
    events = asyncio.run(collect(response))
    visible = "".join(data["text"] for name, data in events if name == "delta")
    assert visible and "用户画像" not in visible
    assert any(name == "agent.step" and data["step"] == "writer_agent" and data["status"] == "failed" for name, data in events)
    assert any(name == "error" for name, _ in events)
    assert not any(name in {"done", "meta"} for name, _ in events)
    asyncio.run(response.background())
    assert not saved


def test_incremental_images_split_across_chunks_cannot_leak_or_duplicate_text():
    answer = IncrementalAnswer(lambda value: chat._strip_unbacked_image_markdown(value, []), lambda value: True)
    chunks = ["  " + "开头文字。" * 30, "![", "长描述" * 60, "](https://", "example.com/fake.png)", "后续文字。" * 30 + "  "]
    visible = "".join(answer.push(chunk) for chunk in chunks) + answer.push(final=True)
    assert visible == chat._strip_unbacked_image_markdown("".join(chunks), [])
    assert "fake.png" not in visible and "长描述" not in visible


def test_incremental_rule_rejection_does_not_publish_rejected_chunk():
    answer = IncrementalAnswer(str.strip, lambda value: "禁止内容" not in value)
    before = answer.push("允许内容。" * 30)
    with pytest.raises(IncrementalReviewBlocked):
        answer.push("禁止内容")
    assert answer.published == before
    assert "禁止内容" not in before


@pytest.mark.parametrize("override", [
    {"memory_statement": True}, {"memory_overview": True}, {"has_attachments": True},
    {"memory": {"governance": {"blockedTerms": [{"value": "x"}]}}},
    {"memory": {"governance": {"lowConfidenceTerms": [{"value": "x"}]}}},
    {"tool_results": [{"ok": False}]}, {"project_context": {"resolver": {"projectId": 9}}},
    {"candidate_goals": [{"title": "goal"}]}, {"candidate_plans": [{"title": "plan"}]},
    {"sufficiency": {"sufficient": False}},
])
def test_dependency_gates_are_conservative(override):
    policy = dict(reviewer_enabled=False, contexts=[], memory={}, sufficiency={}, tool_results=[],
                  project_context={}, candidate_goals=[], candidate_plans=[], memory_statement=False,
                  memory_overview=False, has_attachments=False)
    assert streaming_policy(**policy) == "incremental"
    assert streaming_policy(**{**policy, **override}) != "incremental"
