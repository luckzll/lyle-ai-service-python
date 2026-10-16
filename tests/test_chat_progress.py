from concurrent.futures import ThreadPoolExecutor
import json
from threading import Barrier

import pytest

from app.chat_progress import emit_step, progress_scope, sanitize_timings
from app.graphs import multi_agent_graph as graph
from app.security.chat_debug import build_safe_chat_debug


def _supervisor(*, memory=True, rag=True, tools=False):
    return {"intent": "knowledge", "use_memory": memory, "use_rag": rag,
            "use_tools": tools, "next_agents": []}


def test_progress_scope_restores_callback_and_is_non_fatal():
    outer, inner = [], []
    with progress_scope(outer.append):
        emit_step("supervisor_agent", "running")
        with progress_scope(inner.append):
            emit_step("memory_agent", "running")
        emit_step("supervisor_agent", "completed")
        with progress_scope(lambda event: (_ for _ in ()).throw(RuntimeError("observer gone"))):
            emit_step("memory_agent", "running")
    emit_step("tool_agent", "running")
    assert [event["step"] for event in outer] == ["supervisor_agent"] * 2
    assert [event["step"] for event in inner] == ["memory_agent"]


def test_real_graph_reports_parallel_starts_completion_order_and_rag_timings(monkeypatch):
    events = []
    branch_barrier = Barrier(2)
    monkeypatch.setattr(graph, "run_supervisor_agent", lambda *_: _supervisor())

    def memory(*_):
        assert any(e["step"] == "memory_agent" and e["status"] == "running" for e in events)
        branch_barrier.wait(timeout=5)
        return {"memory": {}, "memory_context": "private memory"}

    def rag(*_):
        assert any(e["step"] == "rag_agent" and e["status"] == "running" for e in events)
        branch_barrier.wait(timeout=5)
        return {"contexts": [{"chunk": "private document"}], "queries": ["private query"],
                "sufficiency": {"sufficient": True}, "timings": {"embedding_ms": 12.5, "secret": "never copy"}}

    monkeypatch.setattr(graph, "run_memory_agent", memory)
    monkeypatch.setattr(graph, "run_rag_agent", rag)
    with progress_scope(events.append):
        result = graph.prepare_multi_agent_context("u", "c", "private query", ["kb"])

    assert result["memory_context"] == "private memory"
    assert result["contexts"] == [{"chunk": "private document"}]
    positions = {(e["step"], e["status"]): i for i, e in enumerate(events)}
    assert positions["supervisor_agent", "completed"] < positions["memory_agent", "running"]
    assert positions["supervisor_agent", "completed"] < positions["rag_agent", "running"]
    assert positions["gather_agent", "running"] > positions["memory_agent", "completed"]
    assert positions["gather_agent", "running"] > positions["rag_agent", "completed"]
    assert ("tool_agent", "running") not in positions
    assert ("tool_agent", "skipped") in positions
    rag_done = events[positions["rag_agent", "completed"]]
    assert rag_done["timings"] == {"embedding_ms": 12.5}
    assert rag_done["summary"] == "已检索 1 次，取得 1 个相关资料片段"
    assert all(event.get("summary") for event in events)
    assert not any(event.get("summary") == "步骤已完成" for event in events)
    assert next(t for t in result["agent_trace"] if t["agent"] == "rag_agent")["timings"] == rag_done["timings"]
    assert "private" not in json.dumps(events)
    assert "secret" not in json.dumps(events)


def test_overlapping_requests_keep_langgraph_worker_contexts_isolated(monkeypatch):
    monkeypatch.setattr(graph, 'load_conversation_context', lambda *_: {'memory': {}, 'memory_context': ''})
    supervisor_barrier = Barrier(2)
    rag_barrier = Barrier(2)

    def supervisor(question, _):
        supervisor_barrier.wait(timeout=5)
        return _supervisor(memory=question == "first", tools=question == "second")

    def rag(question, *_):
        rag_barrier.wait(timeout=5)
        count = 1 if question == "first" else 2
        return {"contexts": [{"chunk": "content"}] * count, "queries": [question],
                "timings": {"total_ms": count}}

    monkeypatch.setattr(graph, "run_supervisor_agent", supervisor)
    monkeypatch.setattr(graph, "run_memory_agent", lambda *_: {"memory": {}, "memory_context": ""})
    monkeypatch.setattr(graph, "run_tool_agent", lambda *_, **__: {"tool_results": [], "tool_context": ""})
    monkeypatch.setattr(graph, "run_rag_agent", rag)

    def run(question):
        events = []
        with progress_scope(events.append):
            graph.prepare_multi_agent_context(question, "conversation", question, ["kb"])
        return events

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(run, "first")
        second_future = pool.submit(run, "second")
        first, second = first_future.result(timeout=10), second_future.result(timeout=10)
    for events, count, skipped, completed in [
        (first, 1, "tool_agent", "memory_agent"),
        (second, 2, "memory_agent", "tool_agent"),
    ]:
        assert next(e for e in events if e["step"] == "rag_agent" and e["status"] == "completed")["timings"] == {"total_ms": count}
        assert [e["status"] for e in events if e["step"] == skipped] == ["skipped"]
        assert [e["status"] for e in events if e["step"] == completed] == ["running", "completed"]


def test_failed_node_reports_failure_without_exception_content(monkeypatch):
    events = []
    monkeypatch.setattr(graph, "run_supervisor_agent", lambda *_: _supervisor(memory=False, tools=False))

    def failed_rag(*_):
        raise RuntimeError("secret query token=provider-secret")

    monkeypatch.setattr(graph, "run_rag_agent", failed_rag)
    with progress_scope(events.append), pytest.raises(RuntimeError):
        graph.prepare_multi_agent_context("u", None, "q", ["kb"])
    assert [e["status"] for e in events if e["step"] == "rag_agent"] == ["running", "failed"]
    assert not any(e["step"] == "gather_agent" for e in events)
    assert "secret" not in json.dumps(events)


def test_no_dispatched_agents_and_decision_skip_are_explicit(monkeypatch):
    monkeypatch.setattr(graph, 'load_conversation_context', lambda *_: {'memory': {}, 'memory_context': ''})
    monkeypatch.setattr(graph, "run_supervisor_agent", lambda *_: _supervisor(memory=False, rag=False))
    events = []
    with progress_scope(events.append):
        result = graph.prepare_multi_agent_context("u", None, "q", [])
    assert result["memory_context"] == ""
    for agent in ("memory_agent", "rag_agent", "tool_agent"):
        assert [e for e in events if e["step"] == agent][-1]["status"] == "skipped"
        assert not any(e["step"] == agent and e["status"] == "completed" for e in events)


def test_empty_recall_and_failed_or_pending_tools_have_truthful_feedback(monkeypatch):
    events = []
    monkeypatch.setattr(graph, "run_supervisor_agent", lambda *_: _supervisor(tools=True))
    monkeypatch.setattr(graph, "run_memory_agent", lambda *_: {"memory": {}, "memory_context": ""})
    monkeypatch.setattr(graph, "run_rag_agent", lambda *_: {
        "contexts": [], "queries": ["private query"], "sufficiency": {},
    })
    monkeypatch.setattr(graph, "run_tool_agent", lambda *_, **__: {
        "tool_results": [
            {"ok": False, "error": "private provider error"},
            {"ok": True, "requiresConfirmation": True, "operation": {"status": "pending", "summary": "private operation"}},
        ],
        "tool_context": "private context",
    })
    with progress_scope(events.append):
        graph.prepare_multi_agent_context("u", "c", "q", ["kb"])
    summaries = {event["step"]: event["summary"] for event in events if event["status"] == "completed"}
    assert summaries["memory_agent"] == "未召回可用的历史上下文"
    assert summaries["rag_agent"] == "已检索 1 次，未命中相关资料片段"
    assert summaries["tool_agent"] == "已返回 2 个工具结果，其中 1 个失败、1 个待确认"
    assert "private" not in json.dumps(events)


@pytest.mark.parametrize(("tool_result", "detail"), [
    ({"ok": False, "requiresConfirmation": True}, "，其中 1 个失败"),
    ({"ok": False, "requiresConfirmation": True, "policyBlocked": True}, "，其中 1 个失败"),
    ({"ok": False, "requiresConfirmation": True, "operation": {"status": "pending"}}, "，其中 1 个失败"),
    ({"ok": True, "requiresConfirmation": True, "policyBlocked": True, "operation": {"status": "pending"}}, "，其中 1 个失败"),
    ({"ok": True, "requiresConfirmation": True}, ""),
    ({"ok": True, "requiresConfirmation": True, "operation": {"status": "confirmed"}}, ""),
    ({"ok": True, "requiresConfirmation": False, "operation": {"status": "failed"}}, "，其中 1 个失败"),
    ({"ok": True, "requiresConfirmation": True, "operation": {"status": "pending"}}, "，其中 1 个待确认"),
    ({"ok": False, "requiresConfirmation": True, "needsClarification": True}, ""),
])
def test_tool_feedback_only_reports_confirmation_for_an_actual_pending_operation(monkeypatch, tool_result, detail):
    events = []
    monkeypatch.setattr(graph, "run_tool_agent", lambda *_, **__: {
        "tool_results": [tool_result], "tool_context": "private context",
    })
    state = graph._initial_state("u", "c", "q", [])
    state["supervisor"] = {"use_tools": True}
    with progress_scope(events.append):
        graph.tool_agent_node(state)
    assert events[-1]["summary"] == "已返回 1 个工具结果" + detail
    assert "private" not in json.dumps(events)


@pytest.mark.parametrize(("approved", "skipped", "status", "summary"), [
    (True, True, "skipped", "本地规则检查完成，本轮无需模型复核"),
    (False, False, "failed", "回答内容检查未通过"),
])
def test_review_feedback_distinguishes_local_check_from_disabled_model_and_rejection(
    monkeypatch, approved, skipped, status, summary,
):
    calls, events = [], []
    monkeypatch.setattr(graph, "run_reviewer_agent", lambda *_, **__: calls.append(True) or {
        "approved": approved, "skipped": skipped, "reason": "private provider content",
    })
    with progress_scope(events.append):
        graph.reviewer_agent_node(graph._initial_state("u", "c", "q", []))
    assert calls == [True]
    assert events[-1]["status"] == status
    assert events[-1]["summary"] == summary
    assert "private" not in json.dumps(events)


def test_timing_whitelist_rejects_text_unknown_nonfinite_and_out_of_range():
    value = {"embedding_ms": 12.25, "query_1_embedding_ms": 5, "query_99_ms": 10,
             "query_rewrite_ms": "secret", "vector_search_ms": float("nan"),
             "keyword_search_ms": float("inf"), "merge_ms": -1, "rerank_ms": True,
             "recall_ms": 10 ** 400, "writer_first_token_ms": 100, "query": "private"}
    expected = {"embedding_ms": 12.25, "query_1_embedding_ms": 5, "writer_first_token_ms": 100}
    assert sanitize_timings(value) == expected
    debug = build_safe_chat_debug(
        knowledge_base_ids=[], contexts=[], agent_trace=[{"agent": "rag_agent", "timings": value}],
        tool_results=[], review={}, usage={}, used_memories=[], memory_exclusions={},
    )
    assert debug["agentTrace"][0]["timings"] == expected
