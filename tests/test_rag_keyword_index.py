from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event, Barrier

import pytest

from app.core.runtime_config import current_rag_config, temporary_ai_config
from app.rag import store
from app.rag.document_loader import TextChunk
from app.rag.keyword_index import KeywordIndex, VersionedChunkList


@pytest.fixture
def corpus(monkeypatch):
    rows = [
        store.KnowledgeChunk("doc-1", "kb-a", "MVP v6.2", "海蓝星项目用于知识库验收。", 1, "项目", 0, 18),
        store.KnowledgeChunk("doc-1", "kb-a", "计划", "MVP 阶段包含聊天和检索，版本 v6.2。", 2, "项目", 18, 40),
        store.KnowledgeChunk("doc-2", "kb-a", "预算", "预算需要结合实际调用量。", 1, "费用", 0, 14),
        store.KnowledgeChunk("doc-1", "kb-b", "MVP v6.2", "海蓝星属于另一个知识库，不可串入。", 1, "项目", 0, 20),
        store.KnowledgeChunk("empty", "kb-a", "", "", 1),
    ]
    monkeypatch.setattr(store, "_chunks", VersionedChunkList(rows, lock=store._lock))
    monkeypatch.setattr(store, "_keyword_index", KeywordIndex())
    return rows


def _legacy_expected(question, scope, top_k):
    # Independent old algorithm retained in store: scan + per-row statistics.
    terms = store._keyword_terms(question)
    rows = [row for row in store._chunks if row.knowledge_base_id in scope]
    scored = [(store._keyword_score(terms, row), row) for row in rows]
    scored = [(score, row) for score, row in scored if score > 0]
    if not scored and not terms:
        scored = [(0.01, row) for row in rows[:top_k]]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [(row.knowledge_base_id, row.doc_id, row.chunk_index, str(round(score, 4)), row.chunk) for score, row in scored[:top_k]]


def _search_tuple(question, scope, top_k=50):
    return [(row["knowledge_base_id"], row["doc_id"], int(row["chunkIndex"]), row["score"], row["chunk"])
            for row in store._keyword_search(question, scope, top_k)]


@pytest.mark.parametrize("question", ["海蓝星", "MVP v6.2", "实际调用量", "不存在abcdef", "", "？！"])
@pytest.mark.parametrize("scope", [["kb-a"], ["kb-b"], ["kb-a", "kb-b"], []])
def test_index_matches_old_scores_ties_and_scope(corpus, question, scope):
    assert _search_tuple(question, scope) == _legacy_expected(question, scope, 50)


def test_warm_lookup_only_tokenizes_query_and_scores_posting_candidates(corpus, monkeypatch):
    real_tokenize = store._keyword_terms
    tokenized, scored = [], []
    monkeypatch.setattr(store, "_keyword_terms", lambda text: tokenized.append(text) or real_tokenize(text))
    real_score = store.score_candidate
    monkeypatch.setattr(store, "score_candidate", lambda terms, candidate: scored.append(candidate.chunk.doc_id) or real_score(terms, candidate))
    store._keyword_search("MVP", ["kb-a"], 5)
    assert len(tokenized) == len(corpus) + 1
    tokenized.clear()
    scored.clear()
    store._keyword_search("预算", ["kb-a"], 5)
    assert tokenized == ["预算"]
    assert scored == ["doc-2"]


def test_mutations_clear_reorder_replacement_and_duplicates_invalidate(corpus):
    for mutate in [
        lambda: store._chunks.append(corpus[0]),
        lambda: store._chunks.reverse(),
        lambda: store._chunks.__setitem__(slice(1, 3), [replace(corpus[2], chunk="MVP 新资料")]),
        lambda: store._chunks.pop(),
        lambda: store._chunks.extend([corpus[1]]),
        lambda: store._chunks.clear(),
    ]:
        store._keyword_search("MVP", ["kb-a"], 20)
        version = store._chunks.version
        mutate()
        assert store._chunks.version > version
        for question in ("MVP", "", "新资料"):
            assert _search_tuple(question, ["kb-a"]) == _legacy_expected(question, ["kb-a"], 50)


def test_whole_list_replacement_is_wrapped_and_stale_terms_are_removed(corpus, monkeypatch):
    assert store._keyword_search("海蓝星", ["kb-a"], 5)
    replacement = replace(corpus[0], chunk="唯一新词 new-token")
    monkeypatch.setattr(store, "_chunks", [replacement])
    assert store._keyword_search("海蓝星", ["kb-a"], 5) == []
    assert store._keyword_search("new-token", ["kb-a"], 5)
    assert isinstance(store._chunks, VersionedChunkList)
    store._chunks[:] = []
    assert store._keyword_search("new-token", ["kb-a"], 5) == []
    assert not store._keyword_index._postings


def test_document_replace_reuses_unmodified_tokenization(corpus, monkeypatch):
    original = store._keyword_terms
    calls = []
    monkeypatch.setattr(store, "_keyword_terms", lambda text: calls.append(text) or original(text))
    store._keyword_search("MVP", ["kb-a"], 5)
    calls.clear()
    store._chunks[0] = replace(corpus[0], chunk="更改内容")
    store._keyword_search("MVP", ["kb-a"], 5)
    assert len(calls) == 2  # Query + one changed document; other rows reused.


def test_ingest_network_does_not_block_readers_and_batch_publishes_atomically(corpus, monkeypatch):
    entered, release = Event(), Event()
    monkeypatch.setattr(store, "_qdrant_delete_document", lambda *_: False)
    monkeypatch.setattr(store, "load_document_text", lambda _: ("replacement", "unused"))
    monkeypatch.setattr(store, "split_text_with_metadata", lambda _: [
        TextChunk("替换后的内容一", 1, "新资料", 0, 7), TextChunk("替换后的内容二", 2, "新资料", 7, 14),
    ])
    def upload(row, index):
        entered.set()
        assert release.wait(3)
    monkeypatch.setattr(store, "_qdrant_upsert", upload)
    before = _search_tuple("海蓝星", ["kb-a"])
    with ThreadPoolExecutor(max_workers=2) as pool:
        ingest = pool.submit(store.add_document, "doc-1", "kb-a", "fake")
        try:
            assert entered.wait(1)
            assert pool.submit(_search_tuple, "海蓝星", ["kb-a"]).result(timeout=1) == before
            assert store._keyword_search("替换后", ["kb-a"], 5) == []
        finally:
            release.set()
        ingest.result(timeout=2)
    assert store._keyword_search("海蓝星", ["kb-a"], 5) == []
    assert len(store._keyword_search("替换后", ["kb-a"], 5)) == 2
    assert store._keyword_search("海蓝星", ["kb-b"], 5)


def test_query_snapshot_remains_consistent_when_corpus_updates(corpus):
    with store._lock:
        snapshot = store._ensure_keyword_index_locked().candidates(["mvp"], ["kb-a"], 5)
    old_scores = [store.score_candidate(["mvp"], row) for row in snapshot]
    store._chunks.clear()
    assert store._keyword_search("MVP", ["kb-a"], 5) == []
    assert [store.score_candidate(["mvp"], row) for row in snapshot] == old_scores


def test_same_document_ingests_are_serialized_without_interleaved_batches(corpus, monkeypatch):
    first_entered, second_entered, release_first = Event(), Event(), Event()
    monkeypatch.setattr(store, "_qdrant_delete_document", lambda *_: False)
    monkeypatch.setattr(store, "load_document_text", lambda path: (path, path))
    monkeypatch.setattr(store, "split_text_with_metadata", lambda text: [TextChunk(text, 1, "", 0, len(text))])
    def upload(row, index):
        if row.chunk == "first-version":
            first_entered.set()
            assert release_first.wait(3)
        else:
            second_entered.set()
    monkeypatch.setattr(store, "_qdrant_upsert", upload)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(store.add_document, "doc-1", "kb-a", "first-version")
        try:
            assert first_entered.wait(1)
            second = pool.submit(store.add_document, "doc-1", "kb-a", "second-version")
            assert not second_entered.wait(.05)
        finally:
            release_first.set()
        first.result(timeout=2)
        second.result(timeout=2)
    assert second_entered.is_set()
    same_doc = [row for row in store._chunks if row.doc_id == "doc-1" and row.knowledge_base_id == "kb-a"]
    assert [row.chunk for row in same_doc] == ["second-version"]


def test_legacy_switch_disables_parallel_and_rank_fusion(monkeypatch):
    calls = []
    monkeypatch.setattr(store, "_qdrant_search", lambda *_: calls.append("vector") or [_hit("high", .99), _hit("both", .2)])
    monkeypatch.setattr(store, "_keyword_search", lambda *_: calls.append("keyword") or [_hit("both", .6)])
    monkeypatch.setattr(store, "_expand_parent_child_contexts", lambda rows, *_: rows)
    config = replace(current_rag_config(), rag_optimized_retrieval_enabled=False, rag_parallel_workers=8, rag_fusion_mode="rrf")
    monkeypatch.setattr(store, "current_rag_config", lambda: config)
    ranked = store.search_candidates("q", ["kb-a"], 5)
    assert calls == ["vector", "keyword"]
    assert [row["doc_id"] for row in ranked] == ["high", "both"]
    assert all("fusion_score" not in row for row in ranked)


def test_explicit_rebuild_preserves_results_and_drops_derived_state(corpus):
    before = _search_tuple("MVP", ["kb-a", "kb-b"])
    old_index = store._keyword_index
    store.rebuild_keyword_index()
    assert store._keyword_index is not old_index
    assert _search_tuple("MVP", ["kb-a", "kb-b"]) == before


def _hit(doc, score, kb="kb-a", index=1):
    return {"doc_id": doc, "knowledge_base_id": kb, "title": doc,
            "chunk": f"{doc} chunk {index}", "chunkIndex": str(index), "parentTitle": "",
            "score": str(score), "backend": "qdrant", "hitModes": "vector"}


def test_rrf_uses_rank_agreement_and_keeps_raw_score_separate():
    vector = [_hit("high", .99), _hit("both", .2)]
    keyword = [{**_hit("both", .6), "backend": "keyword", "hitModes": "keyword"}]
    legacy = store._merge_rag_results(vector, keyword, mode="legacy")
    ranked = store._merge_rag_results(vector, keyword, mode="rrf")
    assert [row["doc_id"] for row in legacy] == ["high", "both"]
    assert [row["doc_id"] for row in ranked] == ["both", "high"]
    assert ranked[0]["score"] == "0.6"
    assert ranked[0]["fusion_score"] == pytest.approx(1 / 62 + 1 / 61)
    assert ranked[0]["vector_rank"] == 2
    assert ranked[0]["keyword_rank"] == 1
    assert set(ranked[0]["hitModes"].split(",")) == {"vector", "keyword"}
    assert "fusion_score" not in legacy[0]


def test_rrf_parent_expansion_does_not_restore_raw_score_order(corpus):
    rows = [
        {**_hit("doc-1", .2, index=1), "chunk": corpus[0].chunk, "fusion_score": .03},
        {**_hit("doc-2", .99), "chunk": corpus[2].chunk, "fusion_score": .01},
    ]
    with temporary_ai_config({"RAG_FUSION_MODE": "rrf"}):
        expanded = store._expand_parent_child_contexts(rows, ["kb-a", "kb-b"], 5)
    assert expanded[0]["doc_id"] == "doc-1"
    assert all(row["knowledge_base_id"] == "kb-a" for row in expanded)
    assert all(float(a["fusion_score"]) >= float(b["fusion_score"]) for a, b in zip(expanded, expanded[1:]))


def test_two_recall_branches_are_parallel_but_results_still_in_source_order(monkeypatch):
    barrier = Barrier(2)
    def vector(*args):
        barrier.wait(timeout=2)
        return [_hit("vector", .8)]
    def keyword(*args):
        barrier.wait(timeout=2)
        return [_hit("keyword", .4)]
    monkeypatch.setattr(store, "_qdrant_search", vector)
    monkeypatch.setattr(store, "_keyword_search", keyword)
    monkeypatch.setattr(store, "_expand_parent_child_contexts", lambda rows, *_: rows)
    config = replace(current_rag_config(), rag_optimized_retrieval_enabled=True, rag_parallel_workers=2, rag_fusion_mode="legacy")
    monkeypatch.setattr(store, "current_rag_config", lambda: config)
    assert [row["doc_id"] for row in store.search_candidates("q", ["kb-a"], 5)] == ["vector", "keyword"]


def test_candidates_do_not_rerank_and_legacy_search_still_does(monkeypatch):
    calls = []
    monkeypatch.setattr(store, "_qdrant_search", lambda *_: [_hit("vector", .8)])
    monkeypatch.setattr(store, "_keyword_search", lambda *_: [])
    monkeypatch.setattr(store, "_expand_parent_child_contexts", lambda rows, *_: rows)
    monkeypatch.setattr(store, "rerank_items", lambda query, rows, key, top_k: calls.append(query) or rows[:top_k])
    assert store.search_candidates("q", ["kb-a"], 5)
    assert calls == []
    assert store.search("q", ["kb-a"], 1)
    assert calls == ["q"]


@pytest.mark.parametrize("mode", ["legacy", "rrf"])
def test_fusion_preserves_distinct_citation_locations_and_combines_same_location(mode):
    first = {**_hit("same-doc", .7), "chunk": "重复出现的相同正文", "startChar": "0", "endChar": "10", "pageNumber": "1"}
    second = {**first, "startChar": "100", "endChar": "110", "pageNumber": "2", "score": ".6"}
    # The same citation from another channel uses numeric metadata, not strings.
    keyword = {**first, "chunkIndex": 1, "startChar": 0, "endChar": 10, "page": 1,
               "pageNumber": None, "backend": "keyword", "hitModes": "keyword", "score": ".8"}
    ranked = store._merge_rag_results([first, second], [keyword], mode=mode)
    assert len(ranked) == 2
    matched = next(row for row in ranked if str(row["startChar"]) == "0")
    assert set(matched["hitModes"].split(",")) == {"vector", "keyword"}
    assert {str(row["startChar"]) for row in ranked} == {"0", "100"}
    if mode == "rrf":
        assert matched["fusion_score"] == pytest.approx(2 / 61)


def test_parent_expansion_keeps_repeated_text_at_distinct_positions(corpus):
    store._chunks[:] = [
        store.KnowledgeChunk("same-doc", "kb-a", "同一文档", "重复正文", 1, "同一节", 0, 4),
        store.KnowledgeChunk("same-doc", "kb-a", "同一文档", "重复正文", 2, "同一节", 10, 14),
        store.KnowledgeChunk("same-doc", "kb-b", "另一知识库", "重复正文", 2, "同一节", 10, 14),
    ]
    source = {**_hit("same-doc", .8), "chunk": "重复正文", "parentTitle": "同一节",
              "startChar": "0", "endChar": "4", "fusion_score": .03}
    expanded = store._expand_parent_child_contexts([source], ["kb-a"], 5)
    assert [(row["chunkIndex"], row["startChar"]) for row in expanded] == [("1", "0"), ("2", "10")]
    assert all(row["knowledge_base_id"] == "kb-a" for row in expanded)
    assert all(row["chunk"] == "重复正文" for row in expanded)
