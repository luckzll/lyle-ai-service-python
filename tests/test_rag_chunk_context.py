from collections import OrderedDict
from dataclasses import replace
from threading import Event
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from app.core.config import settings
from app.rag import store
from app.rag.cache_scope import candidate_cache_attempt, retrieval_cache_scope
from app.rag.chunk_context import CONTEXT_VERSION, LEGACY_CONTEXT_VERSION, build_chunk_context, document_metadata, embedding_text
from app.rag.document_loader import split_text_with_metadata
from app.rag.keyword_index import KeywordIndex, VersionedChunkList


@pytest.fixture
def isolated_store(monkeypatch):
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "rag_chunk_context_enabled", True)
    monkeypatch.setattr(settings, "rag_chunk_context_max_chars", 480)
    monkeypatch.setattr(settings, "vector_backend_failure_cooldown_seconds", 0)
    monkeypatch.setattr(store, "_chunks", VersionedChunkList(lock=store._lock))
    monkeypatch.setattr(store, "_keyword_index", KeywordIndex())
    monkeypatch.setattr(store, "_tasks", {})
    monkeypatch.setattr(store, "_revision_source", None)
    monkeypatch.setattr(store, "_revision_source_version", None)
    monkeypatch.setattr(store, "_revision_snapshot", {})
    monkeypatch.setattr(store, "_revision_clock", 0)
    monkeypatch.setattr(store, "_revision_floor", 0)
    monkeypatch.setattr(store, "_kb_revisions", OrderedDict())
    monkeypatch.setattr(store, "_kb_updates", {})
    monkeypatch.setattr(store, "_qdrant_disabled_until", 0)


def _scope(allowed, versions=None):
    return retrieval_cache_scope("user-a", list(allowed), {kb: "revision-1" for kb in allowed}, "auth-1",
                                 allowed_document_ids=allowed,
                                 document_revisions=versions or {kb: {doc: "legacy" for doc in docs} for kb, docs in allowed.items()})


def test_deterministic_context_only_copies_leading_explicit_metadata():
    text = "# 星桥说明\n项目名称：星桥\n版本号：v2.3\n\n# 历史记录\n版本号：v1.0\n竞争产品：猎户\n"
    metadata = document_metadata(text)
    assert metadata == ("项目名称：星桥", "版本号：v2.3")
    raw = "它将在 14 天后清理。"
    context = build_chunk_context("星桥说明.txt", "# 恢复规则", raw, metadata)
    assert context.version == CONTEXT_VERSION
    assert "星桥说明.txt" in context.retrieval_text
    assert "v2.3" in context.retrieval_text
    assert "v1.0" not in context.retrieval_text and "猎户" not in context.retrieval_text
    assert context.retrieval_text.endswith(raw)
    assert context.sources == ("document_title", "section_title", "document_metadata")


def test_metadata_does_not_promote_later_section_or_infer_entities():
    assert document_metadata("普通开头。\n版本号：v9.9\n") == ()
    assert document_metadata("# 文档\n# 旧版本\n版本号：v1\n") == ()
    assert document_metadata("它属于星桥，可能适用于新版。") == ()


def test_prefix_budget_preserves_entire_raw_chunk_and_never_truncates_version():
    raw = "原始正文" * 300
    context = build_chunk_context("说明.txt", "# 章节", raw, ("版本号：" + "v123" * 80,), max_chars=1200)
    prefix = context.retrieval_text.split("\n\n原文片段：\n")[0]
    assert len(prefix) <= 160
    assert "版本号" not in prefix
    assert context.retrieval_text.endswith(raw)
    assert build_chunk_context("名称", "章节", raw, enabled=False).retrieval_text == ""
    assert build_chunk_context("名称", "章节", raw, max_chars=0).version == LEGACY_CONTEXT_VERSION


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_chunks_keep_exact_source_offsets_whitespace_and_section_identity(newline):
    text = newline.join(["  # 新版 v2", "  首行正文  ", "\t次行正文", "", "# 旧版 v1", "很长正文" * 50, ""])
    chunks = split_text_with_metadata(text, chunk_size=45, overlap=8)
    assert all(text[chunk.start_char:chunk.end_char] == chunk.text for chunk in chunks)
    assert all(len(chunk.text) <= 45 for chunk in chunks)
    assert all("# 旧版 v1" not in chunk.text for chunk in chunks if chunk.parent_title == "# 新版 v2")
    assert any(chunk.parent_title == "# 旧版 v1" for chunk in chunks)


def test_ingest_enhances_index_and_preserves_original_citation(isolated_store, tmp_path):
    text = "项目名称：Orion\n版本号：v3.2\n\n# 恢复规则\n保留 14 天，之后清理。"
    source = tmp_path / "说明.txt"
    source.write_text(text, encoding="utf-8")
    result = store.add_document("doc-a", "kb-a", str(source), retrieval_revision="revision-new")
    rows = store._keyword_search("Orion v3.2", ["kb-a"], 20)
    recovery = next(row for row in rows if "保留 14 天" in row["chunk"])
    assert "Orion" not in recovery["chunk"]
    assert recovery["contextVersion"] == CONTEXT_VERSION
    assert recovery["retrieval_revision"] == "revision-new"
    assert text[int(recovery["startChar"]):int(recovery["endChar"])] == recovery["chunk"]
    assert result["vector_chunks"] == 0
    assert store.get_task(result["task_id"])["vector_status"] == "disabled"


def test_old_chunks_rebuild_keyword_index_without_claiming_new_vectors(isolated_store):
    legacy = store.KnowledgeChunk("old", "kb-a", "旧文档", "旧正文", 1, "旧章节", 0, 3)
    store._chunks.append(legacy)
    assert embedding_text(legacy) == "旧正文"
    store.rebuild_keyword_index()
    result = store._keyword_search("旧正文", ["kb-a"], 1)[0]
    assert result["contextVersion"] == LEGACY_CONTEXT_VERSION
    assert store._chunks[0].retrieval_text == ""


def test_revisions_detect_append_replace_delete_clear_and_leave_other_kb_stable(isolated_store):
    a = store.KnowledgeChunk("a", "kb-a", "a", "alpha")
    b = store.KnowledgeChunk("b", "kb-b", "b", "beta")
    store._chunks.extend([a, b])
    first = dict(store.get_knowledge_base_revisions(["kb-a", "kb-b"]))
    store._chunks[0] = replace(a, chunk="new alpha")
    second = dict(store.get_knowledge_base_revisions(["kb-a", "kb-b"]))
    assert second["kb-a"] > first["kb-a"] and second["kb-b"] == first["kb-b"]
    del store._chunks[0]
    third = dict(store.get_knowledge_base_revisions(["kb-a", "kb-b"]))
    assert third["kb-a"] > second["kb-a"]
    store._chunks.clear()
    assert dict(store.get_knowledge_base_revisions(["kb-b"]))["kb-b"] > third["kb-b"]
    store._chunks = [a]
    assert dict(store.get_knowledge_base_revisions(["kb-a"]))["kb-a"] > third["kb-a"]


def test_ingest_blocks_caching_during_network_and_bumps_revision_afterward(isolated_store, monkeypatch, tmp_path):
    entered, release = Event(), Event()
    def upload(*_):
        entered.set()
        assert release.wait(2)
        return False
    monkeypatch.setattr(store, "_qdrant_upsert", upload)
    source = tmp_path / "new.txt"
    source.write_text("new data", encoding="utf-8")
    before = store.get_knowledge_base_revisions(["kb-a"])
    with ThreadPoolExecutor(max_workers=1) as pool:
        work = pool.submit(store.add_document, "new", "kb-a", str(source))
        try:
            assert entered.wait(1)
            assert store.knowledge_base_update_in_progress(["kb-a"])
            during = store.get_knowledge_base_revisions(["kb-a"])
            assert during != before
        finally:
            release.set()
        work.result(timeout=2)
    assert not store.knowledge_base_update_in_progress(["kb-a"])
    assert store.get_knowledge_base_revisions(["kb-a"]) != during


def test_cleanup_failure_never_publishes_new_chunks_or_success(isolated_store, monkeypatch, tmp_path):
    old = store.KnowledgeChunk("doc", "kb-a", "old", "old text")
    store._chunks.append(old)
    source = tmp_path / "new.txt"
    source.write_text("new text", encoding="utf-8")
    def fail_cleanup(*_):
        raise RuntimeError("vector unavailable")
    monkeypatch.setattr(store, "_qdrant_delete_document", fail_cleanup)
    before = store.get_knowledge_base_revisions(["kb-a"])
    with pytest.raises(RuntimeError):
        store.add_document("doc", "kb-a", str(source))
    assert store._chunks == [old]
    assert store.get_task("ingest-doc") is None
    assert store.get_knowledge_base_revisions(["kb-a"]) != before
    assert not store.knowledge_base_update_in_progress(["kb-a"])


@pytest.fixture
def vector_backend(isolated_store, monkeypatch):
    monkeypatch.setattr(settings, "qdrant_enabled", True)
    monkeypatch.setattr(store, "_ensure_qdrant_collection", lambda: None)
    monkeypatch.setattr(store, "embed_text", lambda text: state.embedded.append(text) or [.1, .2])
    monkeypatch.setattr(store, "embed_query", lambda text: [.1, .2])
    state = SimpleNamespace(requests=[], embedded=[], points=[])
    class Client:
        def __init__(self, **_): pass
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def request(self, method, url, **kwargs):
            state.requests.append((method, url, kwargs))
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"result": state.points})
        def post(self, url, **kwargs): return self.request("POST", url, **kwargs)
        def put(self, url, **kwargs): return self.request("PUT", url, **kwargs)
    monkeypatch.setattr(store.httpx, "Client", Client)
    return state


def test_reingest_cleans_same_revision_tail_and_embeds_enhanced_text_with_version(vector_backend, tmp_path):
    source = tmp_path / "新版本.txt"
    source.write_text("产品名称：Orion\n版本号：v2\n\n# 使用\n新版本正文。", encoding="utf-8")
    result = store.add_document("doc-a", "kb-a", str(source), retrieval_revision="new-revision")
    method, url, request = vector_backend.requests[0]
    assert method == "POST" and url.endswith("/points/delete")
    assert request["json"]["filter"]["must"] == [
        {"key": "knowledge_base_id", "match": {"value": "kb-a"}}, {"key": "doc_id", "match": {"value": "doc-a"}},
        {"key": "retrieval_revision", "match": {"value": "new-revision"}},
    ]
    writes = [row[2]["json"]["points"][0] for row in vector_backend.requests if row[0] == "PUT"]
    assert result["vector_chunks"] == len(writes)
    for point, embedded in zip(writes, vector_backend.embedded):
        payload = point["payload"]
        assert payload["retrieval_text"] == embedded
        assert embedded.endswith(payload["chunk"])
        assert payload["embedding_context_version"] == payload["context_version"] == CONTEXT_VERSION
        assert payload["embedding_model"] == settings.embedding_model
        assert payload["retrieval_revision"] == "new-revision"


def test_concurrent_revisions_use_distinct_point_ids_and_cleanup_scopes(vector_backend):
    base = store.KnowledgeChunk("same-doc", "kb-a", "doc", "source text", retrieval_revision="new")
    assert store._qdrant_upsert(base, 1)
    assert store._qdrant_upsert(replace(base, retrieval_revision="old"), 1)
    point_ids = [request[2]["json"]["points"][0]["id"] for request in vector_backend.requests]
    assert len(set(point_ids)) == 2
    store._qdrant_delete_document("same-doc", "kb-a", "old")
    clauses = vector_backend.requests[-1][2]["json"]["filter"]["must"]
    assert {"key": "retrieval_revision", "match": {"value": "old"}} in clauses
    assert all(clause.get("match", {}).get("value") != "new" for clause in clauses)


def test_trusted_manifest_excludes_deleted_files_from_keyword_parent_and_vector(vector_backend):
    store._chunks.extend([
        store.KnowledgeChunk("active", "kb-a", "active", "needle alpha", 1, "section", 0, 12),
        store.KnowledgeChunk("active", "kb-a", "active", "neighbor", 2, "section", 13, 21),
        store.KnowledgeChunk("deleted", "kb-a", "deleted", "needle deleted", 1, "section"),
        store.KnowledgeChunk("active", "kb-b", "other", "needle other kb", 1, "section"),
    ])
    vector_backend.points = [{"score": .8, "payload": {"knowledge_base_id": kb, "doc_id": doc, "chunk": "needle"}}
                             for kb, doc in [("kb-a", "active"), ("kb-a", "deleted"), ("kb-b", "active")]]
    with _scope({"kb-a": ["active"], "kb-b": []}):
        rows = store._keyword_search("needle", ["kb-a", "kb-b"], 10)
        assert [(row["knowledge_base_id"], row["doc_id"]) for row in rows] == [("kb-a", "active")]
        expanded = store._expand_parent_child_contexts(rows, ["kb-a", "kb-b"], 10)
        assert all(row["knowledge_base_id"] == "kb-a" and row["doc_id"] == "active" for row in expanded)
        vectors = store._qdrant_search("needle", ["kb-a", "kb-b"], 10)
        assert len(vectors) == 1 and vectors[0]["doc_id"] == "active"
        assert vectors[0]["embeddingContextVersion"] == LEGACY_CONTEXT_VERSION
    query_filter = vector_backend.requests[-1][2]["json"]["filter"]
    assert query_filter == {"should": [{"must": [
        {"key": "knowledge_base_id", "match": {"value": "kb-a"}}, {"key": "doc_id", "match": {"any": ["active"]}},
        {"should": [{"key": "retrieval_revision", "match": {"value": "legacy"}}, {"is_empty": {"key": "retrieval_revision"}}]},
    ]}]}
    with _scope({"kb-a": []}):
        count = len(vector_backend.requests)
        assert store._keyword_search("", ["kb-a"], 10) == []
        assert store._qdrant_search("needle", ["kb-a"], 10) == []
        assert len(vector_backend.requests) == count


def test_configured_vector_failure_marks_candidate_uncacheable_but_disabled_does_not(isolated_store, monkeypatch):
    with candidate_cache_attempt() as disabled:
        assert store._qdrant_search("q", ["kb-a"], 5) == []
        assert disabled.healthy
    monkeypatch.setattr(settings, "qdrant_enabled", True)
    monkeypatch.setattr(store, "_qdrant_available", lambda: False)
    with candidate_cache_attempt() as unavailable:
        assert store._qdrant_search("q", ["kb-a"], 5) == []
        assert not unavailable.healthy


def test_document_revision_filters_stale_worker_chunks_and_vector_rows(vector_backend):
    old = store.KnowledgeChunk("doc", "kb-a", "doc", "needle old worker", retrieval_revision="old")
    new = replace(old, chunk="needle new version", retrieval_revision="new")
    store._chunks.extend([old, new])
    vector_backend.points = [{"score": .8, "payload": {
        "knowledge_base_id": "kb-a", "doc_id": "doc", "chunk": f"needle {revision}",
        **({"retrieval_revision": revision} if revision is not None else {}),
    }} for revision in ("old", "new", None)]
    with _scope({"kb-a": ["doc"]}, {"kb-a": {"doc": "new"}}):
        assert [row["retrieval_revision"] for row in store.list_chunks("kb-a")] == ["new"]
        rows = store._keyword_search("needle", ["kb-a"], 5)
        assert [row["retrieval_revision"] for row in rows] == ["new"]
        vectors = store._qdrant_search("needle", ["kb-a"], 5)
        assert [row["retrieval_revision"] for row in vectors] == ["new"]
        assert store._expand_parent_child_contexts([{
            "knowledge_base_id": "kb-a", "doc_id": "doc", "chunk": "needle old worker", "retrieval_revision": "old",
        }], ["kb-a"], 5) == []
    query_filter = vector_backend.requests[-1][2]["json"]["filter"]
    assert query_filter["should"][0]["must"][-1] == {"key": "retrieval_revision", "match": {"value": "new"}}
    with _scope({"kb-a": ["doc"]}):
        assert store._keyword_search("needle", ["kb-a"], 5) == []
        assert [row["retrieval_revision"] for row in store._qdrant_search("needle", ["kb-a"], 5)] == ["legacy"]


def test_embedding_timeout_preserves_keyword_fallback_and_prevents_caching(vector_backend, monkeypatch):
    store._chunks.append(store.KnowledgeChunk("active", "kb-a", "needle", "needle evidence"))
    def timeout(_):
        raise TimeoutError("private provider URL")
    monkeypatch.setattr(store, "embed_query", timeout)
    with candidate_cache_attempt() as attempt:
        results = store.search_candidates("needle", ["kb-a"], 5)
        assert results and results[0]["chunk"] == "needle evidence"
        assert not attempt.healthy
    assert vector_backend.requests == []


@pytest.mark.parametrize("status", [401, 403])
def test_embedding_authorization_failure_does_not_downgrade_to_keyword(vector_backend, monkeypatch, status):
    class AuthorizationError(Exception):
        status_code = status
    def denied(_):
        raise AuthorizationError("private provider details")
    monkeypatch.setattr(store, "embed_query", denied)
    with candidate_cache_attempt() as attempt:
        with pytest.raises(AuthorizationError):
            store._qdrant_search("needle", ["kb-a"], 5)
        assert not attempt.healthy
