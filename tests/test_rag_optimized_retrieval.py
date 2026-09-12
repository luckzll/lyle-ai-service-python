from threading import Barrier, Lock

import httpx
import pytest

from app.chat_progress import rag_phase
from app.core.config import settings
from app.core.runtime_config import current_rag_config, temporary_ai_config
from app.rag import retriever


@pytest.fixture(autouse=True)
def isolated_retrieval(monkeypatch):
    for name, value in {
        "rag_optimized_retrieval_enabled": True,
        "rag_adaptive_retrieval_enabled": False,
        "rag_parallel_workers": 2,
        "rag_retrieval_candidate_limit": 20,
        "rag_global_rerank_limit": 40,
        "rag_context_top_k": 5,
        "rag_multi_query_count": 4,
        "rag_query_rewrite_enabled": True,
        "rag_sufficiency_check_enabled": True,
        "rag_fusion_mode": "legacy",
        "rerank_enabled": True,
        "rerank_provider": "lexical",
    }.items():
        monkeypatch.setattr(settings, name, value)
    monkeypatch.setattr(retriever, "rewrite_rag_queries", lambda query: [query, "expanded query"])
    monkeypatch.setattr(retriever, "judge_rag_sufficiency", lambda *_: {"sufficient": True, "method": "model"})


def candidate(doc_id, text, score=0.7, **extra):
    return {
        "knowledge_base_id": "kb", "doc_id": doc_id, "title": "Source " + doc_id,
        "chunk": text, "score": str(score), "backend": "qdrant", "hitModes": "vector",
        "chunkIndex": 2, "parentTitle": "Chapter", "startChar": 100, "endChar": 200,
        "vector_rank": 1, **extra,
    }


def test_original_question_reranks_combined_pool_once_and_preserves_citations(monkeypatch):
    calls = []
    # Rewritten query can favor a distractor. Ranking by that raw score after
    # unified rerank would put it ahead of the answer to the actual question.
    answer = candidate("answer", "battery charging temperature", 0.1, fusion_score=0.016)
    distractor = candidate("other", "battery industry investments", 9.0)
    monkeypatch.setattr(retriever, "search_candidates", lambda q, kb, limit: calls.append((q, kb, limit)) or
                        ([answer] if q == "charging temperature?" else [distractor]))
    rerank_calls = []

    def rank(query, rows, text_key, top_k):
        rerank_calls.append((query, rows, text_key, top_k))
        assert {row["doc_id"] for row in rows} == {"answer", "other"}
        return [{**row, "rerank_score": 0.95 if row["doc_id"] == "answer" else 0.1,
                 "rerank_backend": "test-model"}
                for row in sorted(rows, key=lambda row: row["doc_id"] != "answer")]

    monkeypatch.setattr(retriever, "rerank_items", rank)
    result = retriever.retrieve_context_with_diagnostics("charging temperature?", ["kb"])
    assert len(rerank_calls) == 1
    assert rerank_calls[0][0] == "charging temperature?"
    assert all(kb == ["kb"] and limit == 20 for _, kb, limit in calls)
    assert [row["doc_id"] for row in result["contexts"]] == ["answer", "other"]
    source = result["contexts"][0]
    assert source["score"] == source["raw_score"] == "0.1"
    assert source["fusion_score"] == 0.1 and source["fusion_method"] == "legacy"
    assert source["rerank_score"] == 0.95 and source["rerank_backend"] == "test-model"
    assert source["query_ranks"] == [{"queryIndex": 1, "rank": 1, "retrievalFusionScore": 0.016}]
    assert {key: source[key] for key in ("knowledge_base_id", "title", "parentTitle", "chunkIndex", "startChar", "endChar")} == {
        "knowledge_base_id": "kb", "title": "Source answer", "parentTitle": "Chapter",
        "chunkIndex": "2", "startChar": "100", "endChar": "200",
    }
    assert result["retrievalPlan"]["rerankPasses"] == 1
    assert result["retrievalPlan"]["candidateCount"] == 2


def test_cosmetic_query_duplicates_are_not_searched_twice(monkeypatch):
    calls = []
    monkeypatch.setattr(retriever, "rewrite_rag_queries", lambda _: ["知识库检索", "知识库检索？", "知识库检索!", "向量召回"])
    monkeypatch.setattr(retriever, "search_candidates", lambda q, *_: calls.append(q) or [])
    result = retriever.retrieve_context_with_diagnostics("知识库检索?", ["kb"])
    assert result["queries"] == ["知识库检索?", "向量召回"]
    assert set(calls) == {"知识库检索?", "向量召回"} and len(calls) == 2
    assert retriever._unique_queries(["C++?", "C#?", "v1.2", "v1.2?", "Question.", "question?"], 6) == ["C++?", "C#?", "v1.2", "Question.", "question?"]


@pytest.mark.parametrize("queries", [
    ['解释 {"mode":"a"}', '解释 {"mode""a"}'],
    ['JSON {"a":1,"b":2}', 'JSON {"a":1;"b":2}'],
    ["配置 --retry=1", "配置 --retry:1"],
    ["版本 v1.2", "版本 v12"],
    ["path 'a,b'", "path 'ab'"],
    ["参数 {a}", "参数 [a]"],
])
def test_meaningful_internal_query_punctuation_is_preserved(queries):
    assert retriever._unique_queries(queries, 4) == queries


def test_fusion_deduplicates_sources_without_losing_scope_or_position():
    first = candidate("same-id", "same passage", 0.3, vector_rank=2)
    second = candidate("same-id", "same passage", 1.0, backend="memory", hitModes="keyword", keyword_rank=1)
    other_kb = candidate("same-id", "same passage", 0.9, knowledge_base_id="kb-2")
    rows = retriever._fuse_query_candidates([("q1", [first, first]), ("q2", [second, other_kb])], "rrf")
    assert len(rows) == 2
    matched = rows[0]
    assert matched["knowledge_base_id"] == "kb"
    assert matched["backend"] == "memory,qdrant" and matched["hitModes"] == "keyword,vector"
    assert matched["raw_score"] == "1.0"
    assert matched["fusion_score"] == pytest.approx(2 / 61)
    assert matched["fusion_method"] == "rrf"
    assert matched["keyword_rank"] == 1 and matched["vector_rank"] == 1
    assert len(matched["query_ranks"]) == 2  # The duplicate in one query contributes once.
    assert matched["startChar"] == 100 and matched["endChar"] == 200


@pytest.mark.parametrize("other_location", [
    {"chunkIndex": 3}, {"startChar": 300, "endChar": 400}, {"pageNumber": 2},
])
def test_same_passage_text_at_different_locations_keeps_both_citations(other_location):
    first = candidate("doc", "repeated boilerplate", 0.7, pageNumber=1)
    other = {**first, **other_location}
    rows = retriever._fuse_query_candidates([("q", [first, other]), ("q2", [first])], "rrf")
    assert len(rows) == 2
    assert len(rows[0]["query_ranks"]) == 2 and len(rows[1]["query_ranks"]) == 1
    assert rows[1]["chunk"] == rows[0]["chunk"]
    for key, value in other_location.items():
        assert rows[1][key] == value


def test_identical_numeric_and_string_location_fields_still_deduplicate():
    first = candidate("doc", "one passage", pageNumber=1)
    other = {**first, "chunkIndex": "2", "startChar": "100", "endChar": "200", "pageNumber": "1"}
    rows = retriever._fuse_query_candidates([("q", [first]), ("q2", [other])], "legacy")
    assert len(rows) == 1 and len(rows[0]["query_ranks"]) == 2


def test_sources_without_location_keep_legacy_deduplication_identity():
    first = {"knowledge_base_id": "kb", "doc_id": "doc", "chunk": "one passage", "score": 0.7}
    other = {**first, "chunkIndex": None, "startChar": "", "endChar": None}
    rows = retriever._fuse_query_candidates([("q", [first]), ("q2", [other])], "legacy")
    assert len(rows) == 1


def test_rrf_agreement_and_legacy_score_are_separate_ranking_modes():
    common = candidate("common", "answer shared by both retrieval queries", 0.1)
    huge_score = candidate("outlier", "irrelevant keyword count", 900)
    inputs = [("q1", [huge_score, common]), ("q2", [common])]
    assert retriever._fuse_query_candidates(inputs, "rrf")[0]["doc_id"] == "common"
    assert retriever._fuse_query_candidates(inputs, "legacy")[0]["doc_id"] == "outlier"


def test_parallel_queries_share_policy_and_each_emit_their_own_timing(monkeypatch):
    gate, lock = Barrier(2), Lock()
    running = maximum = 0
    seen = []
    monkeypatch.setattr(retriever, "rewrite_rag_queries", lambda _: ["one", "two", "three", "four"])

    def search(query, *_):
        nonlocal running, maximum
        with lock:
            running += 1
            maximum = max(maximum, running)
            seen.append(current_rag_config().rag_context_top_k)
        with rag_phase("keyword_search"):
            gate.wait(timeout=3)
        with lock:
            running -= 1
        return [candidate(query, "keyword source", 0.7)]

    monkeypatch.setattr(retriever, "search_candidates", search)
    with temporary_ai_config({"RAG_CONTEXT_TOP_K": "3", "RAG_PARALLEL_WORKERS": "2"}):
        result = retriever.retrieve_context_with_diagnostics("one", ["kb"])
    assert maximum == 2 and running == 0
    assert seen == [3] * 4
    assert len(result["contexts"]) == 3
    for index in range(1, 5):
        assert f"query_{index}_ms" in result["timings"]
        assert f"query_{index}_keyword_search_ms" in result["timings"]
    assert {"recall_ms", "merge_ms", "rerank_ms", "sufficiency_ms", "total_ms"} <= result["timings"].keys()
    assert not any(key.endswith("_rerank_ms") for key in result["timings"])


def test_failed_expansion_retains_original_evidence(monkeypatch):
    def search(query, *_):
        if query == "expanded query":
            raise httpx.ReadTimeout("upstream secret must not appear in diagnostics")
        return [candidate("original", "question evidence")]

    monkeypatch.setattr(retriever, "search_candidates", search)
    result = retriever.retrieve_context_with_diagnostics("question", ["kb"])
    assert [row["doc_id"] for row in result["contexts"]] == ["original"]
    assert result["retrievalPlan"]["failedQueryCount"] == 1
    assert "secret" not in str(result)


def test_all_query_failures_are_not_reported_as_an_empty_knowledge_base(monkeypatch):
    monkeypatch.setattr(retriever, "search_candidates", lambda *_: (_ for _ in ()).throw(httpx.ReadTimeout("offline")))
    with pytest.raises(RuntimeError, match="知识库检索暂时不可用"):
        retriever.retrieve_context_with_diagnostics("question", ["kb"])


@pytest.mark.parametrize("phase", ["recall", "rewrite"])
@pytest.mark.parametrize("error", [PermissionError("denied"), httpx.HTTPStatusError("forbidden", request=httpx.Request("GET", "https://example.test"), response=httpx.Response(403))])
def test_access_denial_never_degrades_to_another_source(monkeypatch, phase, error):
    def reject(*_):
        raise error

    monkeypatch.setattr(retriever, "search_candidates", reject if phase == "recall" else lambda *_: [])
    if phase == "rewrite":
        monkeypatch.setattr(retriever, "rewrite_rag_queries", reject)
    with pytest.raises(type(error)):
        retriever.retrieve_context_with_diagnostics("question", ["kb"])


def test_unavailable_rewrite_still_uses_original_query(monkeypatch):
    calls = []
    monkeypatch.setattr(retriever, "rewrite_rag_queries", lambda *_: (_ for _ in ()).throw(httpx.ReadTimeout("offline")))
    monkeypatch.setattr(retriever, "search_candidates", lambda q, *_: calls.append(q) or [candidate("one", "answer")])
    result = retriever.retrieve_context_with_diagnostics("question", ["kb"])
    assert result["queries"] == calls == ["question"]
    assert len(result["contexts"]) == 1


def test_default_nonadaptive_plan_keeps_rewrite_and_final_sufficiency(monkeypatch):
    events = []
    monkeypatch.setattr(retriever, "rewrite_rag_queries", lambda q: events.append("rewrite") or [q, "expanded query"])
    monkeypatch.setattr(retriever, "search_candidates", lambda q, *_: events.append(q) or [candidate(q, "answer")])
    monkeypatch.setattr(retriever, "judge_rag_sufficiency", lambda *_: events.append("judge") or {"sufficient": True, "method": "model"})
    result = retriever.retrieve_context_with_diagnostics("question", ["kb"])
    assert events[0] == "rewrite" and events[-1] == "judge" and events.count("judge") == 1
    assert result["retrievalPlan"]["adaptiveUsed"] is False
    assert result["queries"] == ["question", "expanded query"]


def test_adaptive_model_judgement_can_reuse_original_evidence(monkeypatch):
    monkeypatch.setattr(settings, "rag_adaptive_retrieval_enabled", True)
    monkeypatch.setattr(retriever, "rewrite_rag_queries", lambda *_: pytest.fail("sufficient original evidence must skip rewrite"))
    calls = []
    monkeypatch.setattr(retriever, "search_candidates", lambda q, *_: calls.append(q) or [candidate("one", "answer")])
    result = retriever.retrieve_context_with_diagnostics("question", ["kb"])
    assert calls == result["queries"] == ["question"]
    assert result["retrievalPlan"]["expansionSkipped"] is True
    assert result["retrievalPlan"]["rerankPasses"] == 1
    assert result["sufficiency"]["method"] == "model"


@pytest.mark.parametrize("judgement", [
    {"sufficient": True, "method": "rule"}, {"sufficient": True, "method": "disabled"},
    {"sufficient": True}, {"sufficient": "true", "method": "model"},
])
def test_adaptive_requires_explicit_model_coverage_not_score_or_rule_overlap(monkeypatch, judgement):
    monkeypatch.setattr(settings, "rag_adaptive_retrieval_enabled", True)
    calls = []
    monkeypatch.setattr(retriever, "search_candidates", lambda q, *_: calls.append(q) or [candidate(q, "very high score", 999)])
    monkeypatch.setattr(retriever, "judge_rag_sufficiency", lambda *_: judgement)
    result = retriever.retrieve_context_with_diagnostics("question", ["kb"])
    assert calls == ["question", "expanded query"]
    assert result["retrievalPlan"]["expansionSkipped"] is False
    assert result["retrievalPlan"]["rerankPasses"] == 2


@pytest.mark.parametrize("question", [
    "比较A和B", "如何配置？如何更新？", "Explain setup and rollback", "分别说明两种办法", "1. setup 2. rollback",
    "说明启动步骤\n解释回滚流程", "如何训练，部署注意什么", "What are its pros/cons?", "Explain differences between the protocols",
])
def test_compound_or_comparison_questions_never_take_adaptive_shortcut(monkeypatch, question):
    monkeypatch.setattr(settings, "rag_adaptive_retrieval_enabled", True)
    events = []
    monkeypatch.setattr(retriever, "rewrite_rag_queries", lambda q: events.append("rewrite") or [q, "expanded query"])
    monkeypatch.setattr(retriever, "search_candidates", lambda q, *_: events.append(q) or [candidate(q, "one fact")])
    result = retriever.retrieve_context_with_diagnostics(question, ["kb"])
    assert events[0] == "rewrite"
    assert result["retrievalPlan"]["adaptiveUsed"] is False
    assert result["retrievalPlan"]["rerankPasses"] == 1


def test_adaptive_insufficient_expansion_reuses_original_and_rechecks_final_pool(monkeypatch):
    monkeypatch.setattr(settings, "rag_adaptive_retrieval_enabled", True)
    searched, judged = [], []
    monkeypatch.setattr(retriever, "search_candidates", lambda q, *_: searched.append(q) or [candidate(q, q)])

    def judge(question, rows):
        judged.append([row["doc_id"] for row in rows])
        return {"sufficient": len(rows) == 2, "method": "model"}

    monkeypatch.setattr(retriever, "judge_rag_sufficiency", judge)
    result = retriever.retrieve_context_with_diagnostics("question", ["kb"])
    assert searched == ["question", "expanded query"]
    assert judged[0] == ["question"] and set(judged[1]) == {"question", "expanded query"}
    assert result["sufficiency"]["sufficient"] is True
    assert result["retrievalPlan"]["rerankPasses"] == 2


def test_adaptive_can_recover_initial_transient_failure_with_expansion(monkeypatch):
    monkeypatch.setattr(settings, "rag_adaptive_retrieval_enabled", True)
    calls = []

    def search(q, *_):
        calls.append(q)
        if q == "question":
            raise httpx.ReadTimeout("offline")
        return [candidate("expanded", "answer")]

    monkeypatch.setattr(retriever, "search_candidates", search)
    result = retriever.retrieve_context_with_diagnostics("question", ["kb"])
    assert calls == ["question", "expanded query"]
    assert [row["doc_id"] for row in result["contexts"]] == ["expanded"]


def test_disabled_sufficiency_cannot_enable_adaptive(monkeypatch):
    monkeypatch.setattr(settings, "rag_adaptive_retrieval_enabled", True)
    monkeypatch.setattr(settings, "rag_sufficiency_check_enabled", False)
    monkeypatch.setattr(retriever, "search_candidates", lambda *_: [candidate("one", "answer")])
    result = retriever.retrieve_context_with_diagnostics("question", ["kb"])
    assert result["retrievalPlan"]["adaptiveUsed"] is False
    assert result["queries"] == ["question", "expanded query"]


def test_disabled_reranker_does_not_claim_a_model_pass(monkeypatch):
    monkeypatch.setattr(settings, "rerank_enabled", False)
    monkeypatch.setattr(retriever, "search_candidates", lambda *_: [candidate("one", "answer")])
    result = retriever.retrieve_context_with_diagnostics("question", ["kb"])
    assert result["retrievalPlan"]["rerankEnabled"] is False
    assert result["retrievalPlan"]["rerankPasses"] == 0


def test_single_query_retriever_preserves_reranking_and_source_metadata(monkeypatch):
    row = candidate("one", "answer", rerank_score=0.95, rerank_backend="flag_embedding", raw_score="0.7", fusion_score=0.03)
    monkeypatch.setattr(retriever, "search", lambda *_: [row])
    context = retriever.retrieve_context("question", ["kb"])[0]
    for key in ("knowledge_base_id", "rerank_score", "rerank_backend", "raw_score", "fusion_score"):
        assert context[key] == row[key]
    assert context["startChar"] == "100" and context["chunk"] == "answer"


@pytest.mark.parametrize("enabled, expected_count", [(True, 40), (False, 60)])
def test_global_rerank_budget_preserves_fused_order_and_only_limits_model_work(monkeypatch, enabled, expected_count):
    monkeypatch.setattr(settings, "rerank_enabled", enabled)
    monkeypatch.setattr(retriever, "rewrite_rag_queries", lambda q: [q])
    pool = [candidate(str(index), "passage " + str(index), index / 100) for index in range(60)]
    monkeypatch.setattr(retriever, "search_candidates", lambda *_: pool)
    seen = []

    def rank(query, rows, text_key, top_k):
        seen.extend(rows)
        return rows[:top_k]

    monkeypatch.setattr(retriever, "rerank_items", rank)
    result = retriever.retrieve_context_with_diagnostics("question", ["kb"])
    assert len(seen) == expected_count
    assert [row["doc_id"] for row in seen] == [str(index) for index in range(59, 59 - expected_count, -1)]
    assert result["retrievalPlan"]["candidateCount"] == 60
    assert result["retrievalPlan"]["selectedCount"] == expected_count
    assert len(result["contexts"]) == 5
