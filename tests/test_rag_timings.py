import json
from itertools import count

import httpx
import pytest

from app import chat_progress
from app.agents.rag_agent import run_rag_agent
from app.chat_progress import rag_phase, rag_timing_scope
from app.core.config import settings
from app.rag import retriever, store


def test_retrieval_timings_preserve_queries_order_scores_and_result(monkeypatch):
    # This fixture pins the pre-optimization pipeline as a behavior baseline.
    monkeypatch.setattr(settings, "rag_optimized_retrieval_enabled", False)
    calls = []
    clock = count()
    monkeypatch.setattr(chat_progress, "perf_counter", lambda: next(clock) / 1000)
    monkeypatch.setattr(settings, "rag_context_top_k", 2)
    monkeypatch.setattr(settings, "qdrant_enabled", True)
    monkeypatch.setattr(store, "_qdrant_available", lambda: True)
    monkeypatch.setattr(store, "_ensure_qdrant_collection", lambda: calls.append("prepare"))
    monkeypatch.setattr(retriever, "rewrite_rag_queries", lambda q: calls.append("rewrite") or [q, "second query"])
    monkeypatch.setattr(store, "embed_query", lambda q: calls.append("embed:" + q) or [0.1, 0.2])
    monkeypatch.setattr(store, "_keyword_search", lambda q, *args: calls.append("keyword:" + q) or [])
    monkeypatch.setattr(store, "_expand_parent_child_contexts", lambda rows, *args: rows)
    monkeypatch.setattr(store, "rerank_items", lambda q, rows, *args: calls.append("rerank:" + q) or rows)

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, *, json):
            calls.append("vector")
            return httpx.Response(200, request=httpx.Request("POST", url), json={"result": [
                {"score": 0.91, "payload": {"doc_id": "d1", "title": "same document", "chunk": "same chunk"}},
            ]})

    monkeypatch.setattr(store.httpx, "Client", FakeClient)
    monkeypatch.setattr(retriever, "judge_rag_sufficiency", lambda q, rows: calls.append("sufficiency") or {"sufficient": True, "score": 1})
    result = run_rag_agent("original query", ["kb"], True)

    assert calls == ["rewrite", "prepare", "embed:original query", "vector", "keyword:original query", "rerank:original query",
                     "prepare", "embed:second query", "vector", "keyword:second query", "rerank:second query", "sufficiency"]
    assert result["queries"] == ["original query", "second query"]
    # Duplicate chunks still keep the original query at equal scores.
    assert result["contexts"] == [{
        "doc_id": "d1", "title": "same document", "chunk": "same chunk", "score": "0.91",
        "backend": "qdrant", "hitModes": "vector", "chunkIndex": "", "parentTitle": "",
        "startChar": "", "endChar": "", "query": "original query",
    }]
    assert result["sufficiency"] == {"sufficient": True, "score": 1}
    timings = result["timings"]
    assert {"query_rewrite_ms", "sufficiency_ms", "recall_ms", "query_1_ms", "query_2_ms", "total_ms"} <= timings.keys()
    for phase in ("qdrant_prepare", "embedding", "vector_search", "keyword_search", "merge", "parent_expansion", "rerank"):
        assert timings[f"{phase}_ms"] == 2
        assert timings[f"query_1_{phase}_ms"] == 1
        assert timings[f"query_2_{phase}_ms"] == 1
    assert "original query" not in json.dumps(timings)
    assert "same chunk" not in json.dumps(timings)


def test_disabled_vector_backend_does_not_claim_embedding_or_vector_duration(monkeypatch):
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "rag_query_rewrite_enabled", False)
    monkeypatch.setattr(settings, "rag_sufficiency_check_enabled", False)
    monkeypatch.setattr(settings, "rerank_enabled", False)
    monkeypatch.setattr(store, "_keyword_search", lambda *_: [])
    result = retriever.retrieve_context_with_diagnostics("q", ["kb"])
    assert result["contexts"] == []
    assert result["queries"] == ["q"]
    assert "embedding_ms" not in result["timings"]
    assert "vector_search_ms" not in result["timings"]
    assert result["timings"]["keyword_search_ms"] >= 0
    # Disabled phases measure their fast-path call; no model duration is invented.
    assert result["timings"]["query_rewrite_ms"] >= 0


def test_failed_timing_scope_records_elapsed_and_resets_between_calls(monkeypatch):
    clock = iter([1.0, 1.1, 2.0, 2.2])
    monkeypatch.setattr(chat_progress, "perf_counter", lambda: next(clock))
    with rag_timing_scope() as first:
        with pytest.raises(RuntimeError), rag_phase("embedding"):
            raise RuntimeError("provider unavailable")
    with rag_timing_scope() as second:
        with rag_phase("keyword_search"):
            pass
    assert first.snapshot() == {"embedding_ms": 100}
    assert second.snapshot() == {"keyword_search_ms": 200}


def test_rag_not_selected_skips_retrieval_without_misleading_timings(monkeypatch):
    monkeypatch.setattr("app.agents.rag_agent.retrieve_knowledge_diagnostics_tool", lambda *_: pytest.fail("must not retrieve"))
    result = run_rag_agent("q", ["kb"], False)
    assert result["skipped"] is True
    assert result.get("timings", {}) == {}
