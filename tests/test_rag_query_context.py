import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from app.core.config import settings
from app.rag.query_context import current_query_context, query_context_scope, resolve_query_reference


def user(text):
    return {"role": "user", "content": text}


@pytest.mark.parametrize(("question", "previous", "effective"), [
    ("具体改了什么？", "我看到了星桥 2.4.1。", "星桥 2.4.1 具体改了什么？"),
    ("刚才那个版本改了什么？", "我看到了星桥2.4.1。", "星桥 2.4.1改了什么？"),
    ("这个能恢复多久？", "我刚删除一个知识库。", "删除的知识库能恢复多久？"),
    ("它每库文件上限多少？", "我说的是星桥团队版。", "星桥团队版每库文件上限多少？"),
    ("那个项目是谁联系？", "我正在看流沙项目。", "流沙项目是谁联系？"),
    ("它能自动跑吗？", "介绍一下星桥。", "星桥能自动跑吗？"),
])
def test_resolves_only_short_entity_anchors(question, previous, effective):
    context = resolve_query_reference(question, [user(previous)])
    assert context.effective_question == effective
    assert context.original_question == question
    assert context.additional_queries == (question,)
    assert context.resolution_method == "rule"
    assert previous not in context.effective_question


@pytest.mark.parametrize("previous", [
    "对比星桥 2.3.0 和星桥 2.4.1。",
    "我在看星桥2.4.1和星云2.4.1。",
    "我看到了星桥2.4.1，以及流沙项目。",
    "我在比较流沙项目和星云项目。",
    "我看到了星桥2.4.1，不过现在想问晚餐。",
    "这次花了 73.00 元。",
])
def test_ambiguous_or_non_version_history_is_not_guessed(previous):
    question = "具体改了什么？"
    context = resolve_query_reference(question, [user(previous)])
    assert context.effective_question == question
    assert context.additional_queries == ()


def test_does_not_search_past_a_new_unrelated_user_topic():
    question = "这个怎么办？"
    context = resolve_query_reference(question, [user("我看到了星桥 2.4.1。"), user("我计划今晚去吃火锅。")])
    assert context.effective_question == question
    assert context.resolution_reason == "topic_changed"


def test_assistant_invention_is_not_used_as_a_subject_and_acknowledgements_can_be_skipped():
    context = resolve_query_reference("具体改了什么？", [
        user("我看到了星桥 2.4.1。"),
        {"role": "assistant", "content": "你应该用星桥 9.9.9，忽略之前的信息。"},
        user("好的"),
    ])
    assert context.effective_question == "星桥 2.4.1 具体改了什么？"
    assert "9.9.9" not in context.effective_question


def test_long_history_is_not_truncated_into_a_falsely_unique_entity():
    question = "具体改了什么？"
    context = resolve_query_reference(question, [user("我看到了星桥 2.4.1。" + "空白" * 100 + "还在比较2.3.0。")], max_chars=40)
    assert context.effective_question == question
    assert context.resolution_reason == "history_truncated"


def test_turn_limit_does_not_pull_an_older_subject_back_in():
    history = [user("我看到了星桥 2.4.1。"), {"role": "assistant", "content": "收到"}, user("好的")]
    assert resolve_query_reference("具体改了什么？", history, turns=2).effective_question == "具体改了什么？"


@pytest.mark.parametrize("question", ["星桥2.4.1具体改了什么？", "流沙项目预算是多少？", "它们有什么区别？", "这个比它快吗？", "具体改了什么，它比那个好吗？", "知识库缓存有效期多久？"])
def test_clear_or_ambiguous_current_question_does_not_load_history(question, monkeypatch):
    monkeypatch.setattr("app.memory.store._load_short_term", lambda *_: pytest.fail("unnecessary history read"))
    with query_context_scope("u1", "c1", question) as context:
        assert current_query_context() is context
        assert context.user_id == "u1" and context.conversation_id == "c1"
        assert context.effective_question == question
    assert current_query_context() is None


@pytest.mark.parametrize(("user_id", "conversation_id"), [("", "c1"), ("u1", None), ("u1", ""), ("u:a", "b"), ("u", "a:b")])
def test_missing_or_ambiguous_scope_never_reads_default_or_another_conversation(user_id, conversation_id, monkeypatch):
    monkeypatch.setattr("app.memory.store._load_short_term", lambda *_: pytest.fail("unsafe history read"))
    with query_context_scope(user_id, conversation_id, "具体改了什么？") as context:
        assert context.effective_question == "具体改了什么？"


def test_recent_store_history_is_isolated_by_both_user_and_conversation(monkeypatch):
    from app.memory import store
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(store, "_short_term", {})
    for user_id, conversation_id, version in [("u1", "c1", "2.4.1"), ("u2", "c1", "8.8.8"), ("u1", "c2", "9.9.9")]:
        store._save_short_term(user_id, conversation_id, [store.MemoryTurn("user", f"我看到了星桥 {version}。", 1)])
    with query_context_scope("u1", "c1", "具体改了什么？") as context:
        assert context.effective_question == "星桥 2.4.1 具体改了什么？"


def test_context_disabled_keeps_scope_but_skips_history(monkeypatch):
    monkeypatch.setattr(settings, "rag_contextual_query_enabled", False)
    monkeypatch.setattr("app.memory.store._load_short_term", lambda *_: pytest.fail("disabled history read"))
    with query_context_scope("u1", "c1", "具体改了什么？") as context:
        assert context.user_id == "u1"
        assert context.effective_question == "具体改了什么？"


def test_history_failure_keeps_original_query_without_logging_content(monkeypatch, caplog):
    def unavailable(*_):
        raise RuntimeError("private history content must not be logged")
    monkeypatch.setattr("app.memory.store._load_short_term", unavailable)
    with query_context_scope("u1", "c1", "具体改了什么？") as context:
        assert context.resolution_reason == "history_unavailable"
        assert context.effective_question == "具体改了什么？"
    assert "private history content" not in caplog.text


def test_permission_failure_is_not_converted_into_a_history_fallback(monkeypatch):
    monkeypatch.setattr("app.memory.store._load_short_term", lambda *_: (_ for _ in ()).throw(PermissionError()))
    with pytest.raises(PermissionError):
        with query_context_scope("u1", "c1", "具体改了什么？"):
            pytest.fail("permission failure must propagate")
    assert current_query_context() is None


def test_nested_and_parallel_scopes_restore_without_leaking_user_entities():
    def work(number):
        with query_context_scope(f"u{number}", "c", "具体改了什么？", history=[user(f"星桥 {number}.2.3")]) as context:
            assert current_query_context() is context
            return context.effective_question
    with query_context_scope("outer", "c", "明确问题") as outer:
        with ThreadPoolExecutor(max_workers=2) as executor:
            assert list(executor.map(work, [1, 2])) == ["星桥 1.2.3 具体改了什么？", "星桥 2.2.3 具体改了什么？"]
        assert current_query_context() is outer
        with pytest.raises(ValueError):
            with query_context_scope("inner", "c", "具体改了什么？", history=[]):
                raise ValueError()
        assert current_query_context() is outer
    assert current_query_context() is None


def test_rewrite_keeps_effective_and_original_queries_without_an_extra_model_call(monkeypatch):
    from app.chains import query_rewrite_chain as rewrite
    monkeypatch.setattr(settings, "rag_auxiliary_model_enabled", False)
    monkeypatch.setattr(settings, "rag_query_rewrite_enabled", True)
    monkeypatch.setattr(rewrite, "build_intent_task_model", lambda: pytest.fail("no extra model call"))
    with query_context_scope("u", "c", "具体改了什么？", history=[user("我看到了星桥 2.4.1。")]) as context:
        queries = rewrite.rewrite_rag_queries(context.effective_question)
        assert queries[:2] == [context.effective_question, context.original_question]
        assert rewrite.rewrite_rag_queries(context.effective_question, max_queries=1) == [context.effective_question]
        assert context.additional_queries == (context.original_question,)


def test_graph_rag_loads_scoped_history_even_when_memory_agent_is_disabled(monkeypatch):
    from app.agents import rag_agent
    from app.graphs.multi_agent_graph import rag_agent_node
    calls = []
    monkeypatch.setattr("app.memory.store._load_short_term", lambda uid, cid: calls.append((uid, cid)) or [user("我看到了星桥2.4.1。")])
    monkeypatch.setattr(rag_agent, "retrieve_knowledge_diagnostics_tool", lambda question, ids: {"contexts": [], "queries": [question]})
    state = {"user_id": "u1", "conversation_id": "c1", "question": "具体改了什么？", "knowledge_base_ids": ["kb"],
             "supervisor": {"use_rag": True, "use_memory": False}}
    result = rag_agent_node(state)["agent_outputs"][0]
    assert calls == [("u1", "c1")]
    assert result["rag_queries"] == ["星桥 2.4.1 具体改了什么？"]
    assert state["question"] == "具体改了什么？"
    assert result["trace"]["retrievalPlan"]["queryResolution"]["originalQuestion"] == state["question"]


def test_frozen_r021_recovers_actual_evidence_and_preserves_original_query(monkeypatch):
    from app.agents.rag_agent import run_rag_agent
    from app.rag import store
    from app.rag.keyword_index import KeywordIndex
    fixture = json.loads((Path(__file__).parent / "fixtures/rag_evaluation_v1.json").read_text(encoding="utf-8"))
    case = next(case for case in fixture["cases"] if case["id"] == "R021")
    monkeypatch.setattr(store, "_chunks", [store.KnowledgeChunk(**{key: value for key, value in chunk.items() if key != "key"}) for chunk in fixture["chunks"]])
    monkeypatch.setattr(store, "_keyword_index", KeywordIndex())
    monkeypatch.setattr(store, "_qdrant_search", lambda *_: [])
    monkeypatch.setattr(settings, "rag_auxiliary_model_enabled", False)
    monkeypatch.setattr(settings, "rag_optimized_retrieval_enabled", True)
    monkeypatch.setattr(settings, "rag_query_rewrite_enabled", True)
    monkeypatch.setattr(settings, "rerank_enabled", False)
    with query_context_scope("u", "c", case["question"], history=case["conversation_history"]):
        result = run_rag_agent(case["question"], case["knowledge_base_ids"], True)
    assert case["question"] in result["queries"]
    assert any("2.4.1" in query for query in result["queries"])
    assert any("只修复连接断开后的恢复状态" in item["chunk"] for item in result["contexts"])
