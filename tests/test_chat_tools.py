from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import json
from threading import Barrier
from unittest.mock import Mock

from langchain_core.messages import AIMessage
import pytest

from app.chat_tools import MAX_TOOL_CALLS, mcp_display_name, run_observed_tool, start_tool_call, tool_scope
from app.core.config import settings
from app.agents import native_tool_loop as loop
from app.tools import native_registry, tool_registry, agent_mcp


@pytest.mark.parametrize("result,status", [
    ({"ok": True}, "completed"), ({"ok": False}, "failed"),
    ({"ok": True, "requiresConfirmation": True}, "awaiting_confirmation"),
    ({"ok": False, "requiresConfirmation": True, "policyBlocked": True}, "failed"),
    ({"ok": False, "requiresConfirmation": True}, "failed"),
    ({"ok": False, "needsClarification": True}, "needs_clarification"),
    ({"ok": True, "operation": {"status": "failed"}}, "failed"),
])
def test_tool_events_report_actual_boundary_and_status_without_payload(result, status):
    events = []
    raw = {**result, "content": "private result api_key=secret", "summary": "untrusted summary"}

    def operation():
        assert [event["status"] for event in events] == ["running"]
        return raw

    with tool_scope(events.append):
        assert run_observed_tool("weather", operation) is raw
    assert [event["status"] for event in events] == ["running", status]
    assert events[0]["callId"] == events[1]["callId"]
    assert events[0]["title"] == "查询天气"
    assert events[1]["durationMs"] >= 0
    assert [event["sequence"] for event in events] == [1, 2]
    assert "private" not in json.dumps(events) and "untrusted" not in json.dumps(events)


def test_legacy_native_bridge_does_not_duplicate_and_repeated_calls_get_distinct_ids(monkeypatch):
    monkeypatch.setattr(tool_registry, "audit_tool_call", lambda **kwargs: None)
    monkeypatch.setattr(tool_registry, "_execute_tool", lambda *args: {"ok": True, "content": "private"})
    monkeypatch.setattr(tool_registry, "list_available_tools", lambda: [{"name": "health_check", "enabled": True}])
    events = []
    with tool_scope(events.append):
        for _ in range(2):
            native_registry.execute_native("health_check", {}, "u", "c", "secret question")
    assert [item["status"] for item in events] == ["running", "completed"] * 2
    assert len({item["callId"] for item in events}) == 2
    assert all(item["tool"] == "health_check" for item in events)


def test_direct_native_weather_and_validation_are_observed(monkeypatch):
    monkeypatch.setattr(tool_registry, "audit_tool_call", lambda **kwargs: None)
    monkeypatch.setattr(tool_registry, "list_available_tools", lambda: [{"name": "weather", "enabled": True}])
    monkeypatch.setattr("app.tools.weather_tool.get_weather", lambda *args: {"ok": True, "content": "private"})
    events = []
    with tool_scope(events.append):
        native_registry.execute_native("weather", {"city": "上海"}, "u", "c", "天气")
        with pytest.raises(ValueError):
            native_registry.execute_native("weather", {}, "u", "c", "天气")
    assert [item["status"] for item in events] == ["running", "completed", "running", "needs_clarification"]


def test_native_write_preview_is_awaiting_confirmation_not_success(monkeypatch):
    monkeypatch.setattr(tool_registry, "audit_tool_call", lambda **kwargs: None)
    monkeypatch.setattr(tool_registry, "list_available_tools", lambda: [{"name": "todo", "enabled": True}])
    monkeypatch.setattr(native_registry, "create_pending_operation", lambda **kwargs: {
        "ok": True, "requiresConfirmation": True, "operation": {"status": "pending"}, "content": "private preview"})
    events = []
    with tool_scope(events.append):
        native_registry.execute_native("todo", {"title": "private task"}, "u", "c", "private task")
    assert [item["status"] for item in events] == ["running", "awaiting_confirmation"]
    assert "等待确认" in events[-1]["summary"]
    assert "private" not in json.dumps(events)


def test_direct_mcp_bridge_and_confirmed_execution_are_observed(monkeypatch):
    monkeypatch.setattr(agent_mcp, "_invoke", lambda *args: {"ok": True, "requiresConfirmation": True})
    monkeypatch.setattr(agent_mcp, "_execute_confirmed", lambda *args: {"ok": True, "content": "private result"})
    events = []
    with tool_scope(events.append):
        agent_mcp.invoke("u", "c", {"id": 7, "toolName": "search.public_docs"}, {"private": "secret"}, "private question")
        agent_mcp.execute_confirmed("u", {"toolName": "mcp_7", "private": "secret"})
    assert [item["status"] for item in events] == ["running", "awaiting_confirmation", "running", "completed"]
    assert events[0]["callId"] != events[2]["callId"]
    assert events[0]["tool"] == "mcp_7:search.public_docs"
    assert all(item["title"] == "调用外部工具" for item in events)
    assert "private" not in json.dumps(events)


@pytest.mark.parametrize("name,expected", [
    ("lookup.public-docs:v2", "mcp_7:lookup.public-docs:v2"),
    ("lookup" * 40, ("mcp_7:" + "lookup" * 40)[:128]),
    ("unsafe\npassword=secret", "mcp_7"),
    ("查找", "mcp_7"), (None, "mcp_7"),
])
def test_mcp_display_names_only_include_safe_registered_identifiers(name, expected):
    assert mcp_display_name({"id": 7, "toolName": name, "description": "private"}) == expected


@pytest.mark.parametrize("error,status", [(RuntimeError("private"), "failed"), (GeneratorExit(), "interrupted")])
def test_scope_failure_closes_inflight_calls_and_resets_request_context(error, status):
    events = []
    with pytest.raises(type(error)), tool_scope(events.append):
        start_tool_call("weather")
        raise error
    assert [event["status"] for event in events] == ["running", status]
    assert run_observed_tool("weather", lambda: {"ok": True}) == {"ok": True}
    assert len(events) == 2
    later_events = []
    with tool_scope(later_events.append):
        run_observed_tool("weather", lambda: {"ok": True})
    assert [event["sequence"] for event in later_events] == [1, 2]
    assert later_events[0]["callId"] != events[0]["callId"]


def test_legacy_caught_error_and_uncaught_error_cannot_leak_exception(monkeypatch):
    monkeypatch.setattr(tool_registry, "audit_tool_call", lambda **kwargs: None)

    def failure(*args):
        raise RuntimeError("private exception password=secret")

    monkeypatch.setattr(tool_registry, "_execute_tool", failure)
    events = []
    with tool_scope(events.append):
        assert not tool_registry.execute_tool("weather", "u", "c", "private")["ok"]
        with pytest.raises(RuntimeError):
            run_observed_tool("knowledge_search", failure)
    assert [item["status"] for item in events] == ["running", "failed"] * 2
    assert "private" not in json.dumps(events) and "secret" not in json.dumps(events)


def test_shared_scope_orders_parallel_calls_and_separate_requests_are_isolated():
    boundary = Barrier(2)

    def request(tool):
        events = []
        with tool_scope(events.append):
            def operation():
                boundary.wait(timeout=5)
                return {"ok": True}
            run_observed_tool(tool, operation)
        return events

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(request, ["weather", "file_parse"]))
    assert {item["tool"] for item in first} == {"weather"}
    assert {item["tool"] for item in second} == {"file_parse"}
    assert [item["sequence"] for item in first] == [1, 2]
    assert [item["sequence"] for item in second] == [1, 2]

    events = []
    with tool_scope(events.append), ThreadPoolExecutor(max_workers=2) as pool:
        def operation():
            boundary.wait(timeout=5)
            return {"ok": True}
        futures = [pool.submit(copy_context().run, run_observed_tool, tool, operation)
                   for tool in ("weather", "web_search")]
        for future in futures:
            future.result(timeout=5)
    assert [item["sequence"] for item in events] == [1, 2, 3, 4]
    assert [item["status"] for item in events[:2]] == ["running", "running"]


def test_caps_only_display_not_execution_and_closes_abandoned_invocations():
    events, executed = [], []
    with tool_scope(events.append):
        abandoned = start_tool_call("web_search")
        for index in range(MAX_TOOL_CALLS + 2):
            run_observed_tool("weather", lambda: executed.append(True) or {"ok": True})
    assert len(executed) == MAX_TOOL_CALLS + 2
    assert len({item["callId"] for item in events}) == MAX_TOOL_CALLS
    assert len(events) == MAX_TOOL_CALLS * 2
    assert events[-1]["callId"] == abandoned.call_id and events[-1]["status"] == "interrupted"
    abandoned.finish({"ok": True})
    assert events[-1]["status"] == "interrupted"


def test_observer_failure_is_nonfatal_and_unsafe_names_are_not_forwarded():
    with tool_scope(lambda _: (_ for _ in ()).throw(RuntimeError("private observer"))):
        assert run_observed_tool("weather", lambda: {"ok": True})["ok"]
    events = []
    with tool_scope(events.append):
        run_observed_tool("private\npassword=secret", lambda: {"ok": False})
    assert all(item["tool"] == "unavailable_tool" for item in events)
    assert "private" not in json.dumps(events)


def test_native_knowledge_mcp_and_clarification_special_paths_emit_once(monkeypatch):
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(loop, "tool_schemas", lambda: [])
    selected = {"id": 7, "toolName": "search.public_docs", "description": "private description", "inputSchema": {"type": "object"}}
    monkeypatch.setattr(loop.agent_mcp, "catalog", lambda _: {"mcp_7": selected})
    monkeypatch.setattr(loop, "rank_mcp_tools", lambda tools, _: tools)
    monkeypatch.setattr("app.rag.retriever.retrieve_context_with_diagnostics", lambda *args: {"contexts": []})
    monkeypatch.setattr(agent_mcp, "_invoke", lambda *args: {"ok": True, "requiresConfirmation": True})
    model = Mock()
    model.bind_tools.return_value = model
    monkeypatch.setattr(loop, "build_intent_task_model", lambda: model)
    events = []
    model.invoke.side_effect = [
        AIMessage(content="", tool_calls=[{"name": "knowledge_search", "args": {"query": "private"}, "id": "one"}]),
        AIMessage(content="", tool_calls=[{"name": "mcp_7", "args": {}, "id": "two"}]),
    ]
    with tool_scope(events.append):
        result = loop.run_native_loop("u", None, "查询", ["authorized-kb"])
    assert result["decision"]["reason"] == "awaiting_confirmation"
    assert [(item["tool"], item["status"]) for item in events] == [
        ("knowledge_search", "running"), ("knowledge_search", "completed"),
        ("mcp_7:search.public_docs", "running"), ("mcp_7:search.public_docs", "awaiting_confirmation")]
    assert "private" not in json.dumps(events)

    monkeypatch.setattr(loop, "tool_schemas", lambda: [{"function": {"name": "request_clarification"}}])
    model.invoke.side_effect = [AIMessage(content="", tool_calls=[{
        "name": "request_clarification", "args": {"question": "private question"}, "id": "three"}])]
    events = []
    with tool_scope(events.append):
        loop.run_native_loop("u", None, "查询")
    assert [item["status"] for item in events] == ["running", "needs_clarification"]
    assert "private" not in json.dumps(events)
