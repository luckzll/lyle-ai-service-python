from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier, Event

import pytest

from app.core.config import settings
from app.core.runtime_config import current_rag_config
from app.rag import retriever, query_context
from app.rag.cache_scope import (
    allowed_document_ids, allowed_document_revisions, candidate_cache, current_retrieval_cache_scope,
    invalidate_candidate_cache, mark_candidate_uncacheable, retrieval_cache_scope,
)


@pytest.fixture(autouse=True)
def configure_cache(monkeypatch):
    invalidate_candidate_cache()
    monkeypatch.setattr(settings, "rag_candidate_cache_enabled", True)
    monkeypatch.setattr(settings, "rag_candidate_cache_max_entries", 128)
    monkeypatch.setattr(settings, "rag_candidate_cache_max_bytes", 8 * 1024 * 1024)
    monkeypatch.setattr(settings, "rag_candidate_cache_ttl_seconds", 30)
    monkeypatch.setattr(retriever, "get_knowledge_base_revisions", lambda kb: tuple((key, 1) for key in sorted(kb)))
    monkeypatch.setattr(retriever, "knowledge_base_update_in_progress", lambda _: False)
    yield
    invalidate_candidate_cache()


@contextmanager
def scope(user="u", conversation="c", kb="kb", version="v1", auth="a1", documents=None):
    ids = ["doc"] if documents is None else documents
    with retrieval_cache_scope(user, [kb], {kb: version}, auth, conversation,
                               {kb: ids}, {kb: {doc: "legacy" for doc in ids}}) as context:
        yield context


def lookup(question="question", kb="kb"):
    return retriever._search_candidates_cached(question, [kb], 20, current_rag_config())


def row(kb="kb"):
    return {"knowledge_base_id": kb, "doc_id": "doc", "title": "original title", "chunk": "original passage",
            "chunkIndex": "2", "startChar": "30", "endChar": "60", "pageNumber": 4,
            "score": "0.8", "vector_rank": 1, "retrieval_context_version": "v2", "nested": {"rank": [1]}}


def test_complete_scope_cache_preserves_original_citation_fields_and_copies(monkeypatch):
    calls = []
    monkeypatch.setattr(retriever, "search_candidates", lambda *args: calls.append(args) or [row()])
    with scope():
        assert allowed_document_ids("kb") == frozenset({"doc"})
        assert allowed_document_ids("other") == frozenset()
        first = lookup()
        first.value[0]["chunk"] = "mutated"
        first.value[0]["nested"]["rank"][0] = 999
        cached = lookup()
    assert first.status == "miss" and cached.status == "hit"
    assert cached.value == [row()]
    assert len(calls) == 1 and current_retrieval_cache_scope() is None


def test_large_scope_reuses_immutable_document_lookup_views():
    documents = [str(index) for index in range(600)]
    versions = {doc: "revision-one" for doc in documents}
    with retrieval_cache_scope("u", ["kb"], {"kb": "version"}, "owner", "conversation",
                               {"kb": documents}, {"kb": versions}) as context:
        ids = allowed_document_ids("kb")
        revisions = allowed_document_revisions("kb")
        fingerprint = context.fingerprint()
        for _ in range(600):
            assert allowed_document_ids("kb") is ids
            assert allowed_document_revisions("kb") is revisions
            assert context.fingerprint() is fingerprint
        with pytest.raises(TypeError):
            revisions["0"] = "changed"
        with pytest.raises(AttributeError):
            ids.add("outside")
        with pytest.raises(TypeError):
            context._document_ids_by_kb["outside"] = frozenset({"outside"})
        with pytest.raises(TypeError):
            context._document_revisions_by_kb["kb"] = {}
        documents.append("outside")
        versions["0"] = "changed"
        assert len(ids) == 600 and revisions["0"] == "revision-one"
        assert allowed_document_ids("outside") is allowed_document_ids("outside")
        empty_revisions = allowed_document_revisions("outside")
        assert empty_revisions == {} and empty_revisions is allowed_document_revisions("outside")
        with pytest.raises(TypeError):
            empty_revisions["outside"] = "forbidden"
    assert allowed_document_revisions("kb") is None


@pytest.mark.parametrize("metadata", [
    {}, {"revisions": {"kb": "v1"}},
    {"revisions": {"kb": "v1"}, "authorization_revision": "auth"},
    {"revisions": {"different": "v1"}, "authorization_revision": "auth", "allowed_document_ids": {"kb": ["doc"]}},
    {"revisions": {"kb": ""}, "authorization_revision": "auth", "allowed_document_ids": {"kb": ["doc"]}},
    {"revisions": {"kb": "v1"}, "authorization_revision": "auth", "allowed_document_ids": {"kb": "doc"}},
])
def test_missing_or_incomplete_trusted_metadata_bypasses_candidate_cache(monkeypatch, metadata):
    calls = []
    monkeypatch.setattr(retriever, "search_candidates", lambda *_: calls.append(1) or [row()])
    with retrieval_cache_scope("u", ["kb"], **metadata):
        assert current_retrieval_cache_scope() is None and allowed_document_ids("kb") is None
        assert lookup().status == lookup().status == "bypass"
    assert len(calls) == 2


def test_no_scope_never_reuses_candidate_results(monkeypatch):
    calls = []
    monkeypatch.setattr(retriever, "search_candidates", lambda *_: calls.append(1) or [row()])
    assert lookup().status == lookup().status == "bypass"
    assert len(calls) == 2


@pytest.mark.parametrize("changed", [
    {"user": "different-user"}, {"conversation": "different-conversation"}, {"kb": "kb-two"},
    {"version": "new-db-manifest"}, {"auth": "new-authorization"}, {"documents": ["doc", "new-doc"]},
])
def test_user_conversation_authorization_scope_and_persistent_version_partition_entries(monkeypatch, changed):
    calls = []
    monkeypatch.setattr(retriever, "search_candidates", lambda q, kb, limit: calls.append(1) or [row(kb[0])])
    with scope():
        assert lookup().status == "miss"
    with scope(**changed):
        assert lookup(kb=changed.get("kb", "kb")).status == "miss"
    with scope():
        assert lookup().status == "hit"
    assert len(calls) == 2


def test_retrieval_policy_and_backend_identity_partition_candidates(monkeypatch):
    calls = []
    monkeypatch.setattr(retriever, "search_candidates", lambda *_: calls.append(1) or [row()])
    with scope():
        lookup()
        monkeypatch.setattr(settings, "rag_fusion_mode", "rrf")
        assert lookup().status == "miss"
        monkeypatch.setattr(settings, "qdrant_endpoint", "https://other.invalid")
        assert lookup().status == "miss"
        monkeypatch.setattr(settings, "embedding_model", "changed-embedding")
        assert lookup().status == "miss"
    assert len(calls) == 4


def test_history_fingerprint_partitions_same_effective_query(monkeypatch):
    calls = []
    monkeypatch.setattr(retriever, "search_candidates", lambda *_: calls.append(1) or [row()])
    context = query_context.QueryContext("u", "c", "it?", "question", history_fingerprint="history-one")
    with scope():
        token = query_context._context.set(context)
        try:
            lookup()
            query_context._context.set(query_context.QueryContext("u", "c", "it?", "question", history_fingerprint="history-two"))
            assert lookup().status == "miss"
        finally:
            query_context._context.reset(token)
    assert len(calls) == 2


def test_partial_backend_failures_are_not_retained(monkeypatch):
    calls = []

    def load(*_):
        calls.append(1)
        mark_candidate_uncacheable()
        return [row()]

    monkeypatch.setattr(retriever, "search_candidates", load)
    with scope():
        assert lookup().status == lookup().status == "miss"
    assert len(calls) == 2 and candidate_cache.stats()["entries"] == 0


def test_empty_results_and_unauthorized_source_rows_are_not_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(retriever, "search_candidates", lambda *_: calls.append(1) or [])
    with scope(documents=[]):
        assert allowed_document_ids("kb") == frozenset()
        lookup()
        lookup()
        monkeypatch.setattr(retriever, "search_candidates", lambda *_: calls.append(1) or [row()])
        lookup()
        lookup()
    assert len(calls) == 4 and candidate_cache.stats()["entries"] == 0


def test_exceptions_are_not_cached_and_permission_errors_are_not_retried(monkeypatch):
    calls = []

    def load(*_):
        calls.append(1)
        raise PermissionError("revoked")

    monkeypatch.setattr(retriever, "search_candidates", load)
    with scope():
        for _ in range(2):
            with pytest.raises(PermissionError):
                lookup()
    assert len(calls) == 2 and candidate_cache.stats()["entries"] == 0


def test_local_generation_change_or_update_window_bypasses_old_entries(monkeypatch):
    generation = [1]
    calls = []
    monkeypatch.setattr(retriever, "get_knowledge_base_revisions", lambda kb: (("kb", generation[0]),))
    monkeypatch.setattr(retriever, "search_candidates", lambda *_: calls.append(1) or [row()])
    with scope():
        lookup()
        generation[0] += 1
        assert lookup().status == "miss"
        monkeypatch.setattr(retriever, "knowledge_base_update_in_progress", lambda _: True)
        assert lookup().status == "bypass"
    assert len(calls) == 3


def test_generation_changing_during_load_prevents_old_result_from_becoming_reusable(monkeypatch):
    generation = [1]
    monkeypatch.setattr(retriever, "get_knowledge_base_revisions", lambda kb: (("kb", generation[0]),))

    def load(*_):
        generation[0] += 1
        return [row()]

    monkeypatch.setattr(retriever, "search_candidates", load)
    with scope():
        assert lookup().status == "miss"
    assert candidate_cache.stats()["entries"] == 0


def test_simultaneous_candidate_requests_share_one_recall(monkeypatch):
    gate, started, release = Barrier(3), Event(), Event()
    calls = []

    def load(*_):
        calls.append(1)
        started.set()
        assert release.wait(3)
        return [row()]

    def run():
        with scope():
            gate.wait(3)
            return lookup()

    monkeypatch.setattr(retriever, "search_candidates", load)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run) for _ in range(2)]
        gate.wait(3)
        assert started.wait(3)
        release.set()
        assert [future.result(3).value for future in futures] == [[row()], [row()]]
    assert len(calls) == 1


def test_request_cache_switch_is_respected(monkeypatch):
    calls = []
    monkeypatch.setattr(settings, "rag_candidate_cache_enabled", False)
    monkeypatch.setattr(retriever, "search_candidates", lambda *_: calls.append(1) or [row()])
    with scope():
        assert lookup().status == lookup().status == "bypass"
    assert len(calls) == 2


@pytest.mark.parametrize("first_rows, expected_queries", [([], ["v1.2 更新内容", "它改了什么"]), ([row()], ["v1.2 更新内容"])])
def test_single_query_reference_resolution_only_falls_back_after_an_empty_result(monkeypatch, first_rows, expected_queries):
    monkeypatch.setattr(settings, "rag_optimized_retrieval_enabled", True)
    monkeypatch.setattr(settings, "rag_multi_query_count", 1)
    monkeypatch.setattr(settings, "rag_adaptive_retrieval_enabled", False)
    calls = []
    monkeypatch.setattr(retriever, "rewrite_rag_queries", lambda q: [q])
    monkeypatch.setattr(retriever, "search_candidates", lambda q, *_: calls.append(q) or (first_rows if q == "v1.2 更新内容" else [row()]))
    monkeypatch.setattr(retriever, "judge_rag_sufficiency", lambda *_: {"sufficient": True})
    context = query_context.QueryContext("u", "c", "它改了什么", "v1.2 更新内容", ("它改了什么",), "history", "rule")
    token = query_context._context.set(context)
    try:
        result = retriever.retrieve_context_with_diagnostics(context.effective_question, ["kb"])
    finally:
        query_context._context.reset(token)
    assert calls == result["queries"] == expected_queries
    assert result["retrievalPlan"]["actualQueryCount"] == len(expected_queries)
