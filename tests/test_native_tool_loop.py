import json
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError

from app.agents import native_tool_loop as loop
from app.core.config import settings
from app.tools import native_registry as registry, agent_mcp
from app.tools.slots import WeatherArgs, CalendarArgs, slot_records


def call(name, args=None, id="call-1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args or {}, "id": id}])


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "agent_loop_enabled", True)
    monkeypatch.setattr(settings, "agent_loop_max_steps", 5)
    monkeypatch.setattr(settings, "agent_loop_max_tool_calls", 8)
    monkeypatch.setattr(settings, "agent_loop_max_retries", 1)
    monkeypatch.setattr(settings, "agent_loop_max_no_progress", 2)
    monkeypatch.setattr(settings, "agent_loop_max_seconds", 120)
    monkeypatch.setattr(loop.agent_mcp, "catalog", lambda user_id: {})
    model = Mock()
    model.bind_tools.return_value = model
    monkeypatch.setattr(loop, "build_intent_task_model", lambda: model)
    execute = Mock(return_value={"ok": True, "content": "证据"})
    monkeypatch.setattr(loop, "execute_native", execute)
    return model, execute


def test_loop_observes_results_and_selects_next_tool(setup):
    model, execute = setup
    snapshots = []
    responses = iter([call("work_summary"), call("health_check", id="call-2"), AIMessage(content="完成")])
    def invoke(messages):
        snapshots.append(list(messages))
        return next(responses)
    model.invoke.side_effect = invoke
    result = loop.run_native_loop("u1", None, "分析项目并诊断服务")
    assert result["selected_tools"] == ["work_summary", "health_check"]
    assert result["decision"]["reason"] == "completed"
    assert any(isinstance(message, ToolMessage) and message.tool_call_id == "call-1" for message in snapshots[1])
    assert execute.call_args_list[0].args[2] == "u1"


def test_confirmation_stops_remaining_batch(setup):
    model, execute = setup
    model.invoke.return_value = AIMessage(content="", tool_calls=[
        {"name": "todo", "args": {"title": "工作"}, "id": "one"},
        {"name": "health_check", "args": {}, "id": "two"},
    ])
    execute.return_value = {"ok": True, "requiresConfirmation": True, "content": "待确认"}
    result = loop.run_native_loop("u1", None, "创建待办后检查状态")
    assert result["decision"]["reason"] == "awaiting_confirmation"
    assert execute.call_count == 1


def test_clarification_does_not_execute(setup):
    model, execute = setup
    model.invoke.return_value = call("request_clarification", {"question": "哪个城市？", "missing_slots": ["city"]})
    result = loop.run_native_loop("u1", None, "天气如何")
    assert result["decision"]["needs_clarification"]
    assert "哪个城市" in result["tool_context"]
    execute.assert_not_called()


def test_budget_applies_inside_model_batch(setup, monkeypatch):
    model, execute = setup
    monkeypatch.setattr(settings, "agent_loop_max_tool_calls", 1)
    model.invoke.return_value = AIMessage(content="", tool_calls=[
        {"name": "work_summary", "args": {}, "id": "one"},
        {"name": "health_check", "args": {}, "id": "two"},
    ])
    result = loop.run_native_loop("u1", None, "分析")
    assert result["decision"]["reason"] == "tool_limit"
    assert execute.call_count == 1


def test_no_repeated_successful_calls(setup):
    model, execute = setup
    model.invoke.return_value = call("health_check")
    result = loop.run_native_loop("u1", None, "诊断")
    assert result["decision"]["reason"] == "no_progress"
    assert execute.call_count == 1


def test_retry_only_retryable_and_bounded(setup):
    model, execute = setup
    model.invoke.return_value = call("health_check")
    execute.return_value = {"ok": False, "retryable": True, "content": "暂时超时"}
    loop.run_native_loop("u1", None, "诊断")
    assert execute.call_count == 2


def test_unknown_tool_never_dispatches(setup):
    model, execute = setup
    model.invoke.return_value = call("shell", {"command": "anything"})
    loop.run_native_loop("u1", None, "检查")
    execute.assert_not_called()


def test_model_failure_preserves_evidence_without_fallback(setup):
    model, execute = setup
    model.invoke.side_effect = [call("work_summary"), RuntimeError("bad provider")]
    result = loop.run_native_loop("u1", None, "分析")
    assert result["decision"]["reason"] == "model_error"
    assert len(result["tool_results"]) == 1
    assert execute.call_count == 1


def test_knowledge_scope_is_not_model_controlled(setup, monkeypatch):
    model, _ = setup
    retrieve = Mock(return_value={"contexts": []})
    monkeypatch.setattr("app.rag.retriever.retrieve_context_with_diagnostics", retrieve)
    model.invoke.side_effect = [call("knowledge_search", {"query": "证据"}), AIMessage(content="完成")]
    loop.run_native_loop("u1", None, "检索", ["authorized-kb"])
    retrieve.assert_called_once_with("证据", ["authorized-kb"])


def test_slots_reject_identity_and_naive_time():
    with pytest.raises(ValidationError):
        WeatherArgs(city="上海", user_id="another-user")
    with pytest.raises(ValidationError):
        CalendarArgs(title="会议", starts_at="2026-09-10T15:00:00")
    slots = slot_records(WeatherArgs(city="上海"), "上海天气")
    assert slots["city"] == {"value": "上海", "source": "user_explicit", "confidence": 1.0}
    assert slots["date"]["source"] == "default"


def test_file_dispatch_preserves_spaces_and_checks_user(monkeypatch):
    parse = Mock(return_value={"ok": True, "content": "文件"})
    monkeypatch.setattr("app.tools.file_parse_tool.parse_file", parse)
    monkeypatch.setattr(registry.legacy, "audit_tool_call", Mock())
    monkeypatch.setattr(registry.legacy, "list_available_tools", lambda: [{"name": "file_parse", "enabled": True}])
    registry.execute_native("file_parse", {"file_ref": "C:/allowed/my file.txt"}, "u1", None, "读取文件")
    parse.assert_called_once_with("C:/allowed/my file.txt", "u1")


def test_todo_dispatch_preserves_explicit_datetime(monkeypatch):
    preview = Mock(return_value={"ok": True, "requiresConfirmation": True, "content": "待确认"})
    monkeypatch.setattr(registry, "create_pending_operation", preview)
    monkeypatch.setattr(registry.legacy, "audit_tool_call", Mock())
    monkeypatch.setattr(registry.legacy, "list_available_tools", lambda: [{"name": "todo", "enabled": True}])
    registry.execute_native("todo", {"title": "开会", "due_at": "2026-09-10T15:00:00+08:00"}, "u1", "c1", "创建待办")
    assert preview.call_args.kwargs["payload"]["dueAt"] == "2026-09-10T15:00:00+08:00"


@pytest.fixture
def mcp_tool(monkeypatch):
    tool = {"id": 7, "serverId": 3, "toolName": "mail.send", "riskLevel": "write",
            "description": "发邮件", "enabled": True,
            "inputSchema": {"type": "object", "properties": {"to": {"type": "string"}},
                            "required": ["to"], "additionalProperties": False}}
    monkeypatch.setattr(agent_mcp, "_get_tool", lambda user, id: tool)
    monkeypatch.setattr(agent_mcp, "server_fingerprint", lambda user, tool: "server")
    return tool


def test_mcp_write_previews_does_not_send(mcp_tool, monkeypatch):
    preview, remote = Mock(return_value={"ok": True, "requiresConfirmation": True}), Mock()
    monkeypatch.setattr(agent_mcp, "create_pending_operation", preview)
    monkeypatch.setattr(agent_mcp, "invoke_mcp_tool", remote)
    agent_mcp.invoke("u1", "c1", mcp_tool, {"to": "user@example.com"}, "发邮件")
    remote.assert_not_called()
    assert preview.call_args.kwargs["payload"]["arguments"] == {"to": "user@example.com"}


def test_mcp_model_cannot_add_confirmation(mcp_tool):
    with pytest.raises(Exception):
        agent_mcp.validate(mcp_tool, {"to": "u@x.com", "confirmed": True})


def test_mcp_changed_schema_invalidates_confirmation(mcp_tool):
    payload = {"toolId": 7, "arguments": {"to": "u@x.com"}, "fingerprint": "old"}
    result = agent_mcp.execute_confirmed("u1", {"payload": payload})
    assert not result["ok"]


def test_mcp_high_risk_requires_two_separate_approvals(mcp_tool, monkeypatch):
    mcp_tool["riskLevel"] = "write_high"
    preview = Mock(return_value={"ok": True, "requiresConfirmation": True})
    remote = Mock(return_value={"ok": True})
    monkeypatch.setattr(agent_mcp, "create_pending_operation", preview)
    monkeypatch.setattr(agent_mcp, "invoke_mcp_tool", remote)
    operation = {"id": 10, "toolName": "mcp_7", "summary": "send", "operationType": "mcp.invoke",
                 "payload": {"toolId": 7, "arguments": {"to": "u@x.com"}, "fingerprint": agent_mcp.fingerprint(mcp_tool), "serverFingerprint": "server"}}
    agent_mcp.execute_confirmed("u1", operation)
    remote.assert_not_called()
    assert preview.call_args.kwargs["operation_type"] == "mcp.invoke.second"
    agent_mcp.execute_confirmed("u1", {**operation, "operationType": "mcp.invoke.second"})
    assert remote.call_args.args[2]["secondConfirmed"] is True


def test_mcp_external_refs_never_resolved(mcp_tool):
    mcp_tool["inputSchema"] = {"$ref": "http://169.254.169.254/schema"}
    with pytest.raises(ValueError):
        agent_mcp.validate(mcp_tool, {})


def test_native_entry_integrated_in_chat(setup, monkeypatch):
    model, execute = setup
    from app.agents.tool_agent import run_tool_agent
    monkeypatch.setattr(loop, "available", lambda: True)
    model.invoke.side_effect = [call("work_summary"), AIMessage(content="完成")]
    result = run_tool_agent("u1", None, "检查项目", True)
    assert result["decision"]["router"] == "native_tool_loop"
    assert execute.call_count == 1


def test_real_chatopenai_tool_protocol_with_fake_transport(setup, monkeypatch):
    import httpx
    from langchain_openai import ChatOpenAI
    requests = []
    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        message = {"role": "assistant", "content": "完成"}
        if len(requests) == 1:
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "weather-call", "type": "function",
                "function": {"name": "weather", "arguments": '{"city":"上海"}'},
            }]}
        return httpx.Response(200, json={"id": "response-1", "object": "chat.completion",
            "created": 1, "model": "test", "choices": [{"index": 0, "message": message,
            "finish_reason": "tool_calls" if len(requests) == 1 else "stop"}]})
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = ChatOpenAI(model="test", api_key="test", http_client=client, max_retries=0)
        monkeypatch.setattr(loop, "build_intent_task_model", lambda: model)
        result = loop.run_native_loop("u1", None, "上海天气")
    assert result["decision"]["reason"] == "completed"
    assert requests[0]["tools"][0]["type"] == "function"
    assert any(item.get("tool_call_id") == "weather-call" for item in requests[1]["messages"])


def test_external_claim_committed_before_remote_io(monkeypatch):
    from app.tools import operation_confirmation as confirmation
    events = []
    class Connection:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def execute(self, sql, args): events.append("claim" if "'executing'" in sql and "SET status = 'executing'" in sql else "finish")
        def commit(self): events.append("commit")
    operation = {"id": 7, "status": "pending", "toolName": "mcp_7", "conversationId": "c1"}
    monkeypatch.setattr(confirmation, "postgres_connection", Connection)
    monkeypatch.setattr(confirmation, "_get_operation_for_update", lambda *args: operation)
    monkeypatch.setattr(confirmation, "get_pending_operation", lambda *args: {**operation, "status": "confirmed"})
    monkeypatch.setattr(confirmation, "audit_tool_call", Mock())
    def execute(*args):
        assert events[:2] == ["claim", "commit"]
        events.append("remote")
        return {"ok": True}
    monkeypatch.setattr(confirmation, "_execute_operation", execute)
    confirmation._confirm_external_operation("u1", 7)
    assert events == ["claim", "commit", "remote", "finish", "commit"]


@pytest.mark.parametrize("status", ["executing", "confirmed", "failed", "rejected"])
def test_external_claim_never_replayed(monkeypatch, status):
    from app.tools import operation_confirmation as confirmation
    connection = Mock()
    connection.__enter__ = Mock(return_value=connection)
    connection.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(confirmation, "postgres_connection", lambda: connection)
    monkeypatch.setattr(confirmation, "_get_operation_for_update", lambda *args: {"status": status})
    execute = Mock()
    monkeypatch.setattr(confirmation, "_execute_operation", execute)
    assert confirmation._confirm_external_operation("u1", 1)["idempotentReplay"]
    execute.assert_not_called()


def test_delete_classification_requires_confirmation():
    from app.mcp.client_manager import infer_tool_risk
    assert infer_tool_risk("delete_document") == "write_high"
    assert infer_tool_risk("batch_edit_files") == "write_high"
    assert infer_tool_risk("send_email") == "write_high"
    assert infer_tool_risk("change_permission") == "dangerous"


def test_weather_date_outside_available_forecast_is_not_fabricated(monkeypatch):
    from app.tools import weather_tool
    monkeypatch.setattr(weather_tool, "_get_weather", lambda city: {"ok": True, "weather": {"daily": {"time": ["2026-09-10"]}}})
    assert not weather_tool.get_weather("上海", "2026-12-01")["ok"]
