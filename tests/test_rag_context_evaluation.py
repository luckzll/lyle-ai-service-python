"""Contracts for the local-vector experiment, using a deterministic fake model."""
import importlib.util
from pathlib import Path

import pytest


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_rag_context_v2.py"
    spec = importlib.util.spec_from_file_location("context_evaluation_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture():
    text = "项目名称：Atlas\n版本号：v2\n\n# 章节\nneedle alpha"
    raw = "needle alpha"
    start = text.index(raw)
    return {"documents": [{"knowledge_base_id": "kb-a", "doc_id": "doc", "title": "手册", "text": text}],
            "chunks": [{"key": "kb-a/doc#1", "knowledge_base_id": "kb-a", "doc_id": "doc", "title": "手册 #1",
                        "chunk": raw, "chunk_index": 1, "parent_title": "# 章节", "start_char": start, "end_char": len(text)}],
            "cases": [{"id": "test", "category": "contract", "question": "needle", "knowledge_base_ids": ["kb-a"],
                       "answerable": True, "evidence": [{"quote": "needle", "acceptable_chunk_keys": ["kb-a/doc#1"],
                                                         "review_only": "GOLD_MUST_NOT_ENTER_MODEL"}],
                       "expected": "EXPECTED_MUST_NOT_ENTER_MODEL"}]}


class FakeEmbedding:
    def __init__(self):
        self.inputs = []

    def embed(self, texts, **_kwargs):
        for text in texts:
            self.inputs.append(text)
            yield [1.0, 0.25] if "needle" in text else [0.1, 1.0]


def test_population_preserves_fixture_raw_spans_and_only_enhances_search_text(monkeypatch):
    evaluation = _script()
    fixture = _fixture()
    store = evaluation.load_module(evaluation.ROOT / "app/rag/store.py", "evaluation_contract_store")
    before, raw_texts = evaluation.populate_context(store, fixture, enhanced=False)
    after, contextual_texts = evaluation.populate_context(store, fixture, enhanced=True)
    assert raw_texts == ["needle alpha"]
    assert "Atlas" in contextual_texts[0] and "v2" in contextual_texts[0]
    assert after[0].chunk == before[0].chunk == "needle alpha"
    assert (after[0].start_char, after[0].end_char) == (before[0].start_char, before[0].end_char)
    assert "retrieval_text" not in fixture["chunks"][0]


def test_real_pipeline_interface_does_not_feed_gold_to_embeddings_or_reranker(monkeypatch):
    evaluation = _script()
    fixture = _fixture()
    store = evaluation.load_module(evaluation.ROOT / "app/rag/store.py", "evaluation_contract_scoring_store")
    records, texts = evaluation.populate_context(store, fixture, enhanced=True)
    model = FakeEmbedding()
    vectors = evaluation.embed_documents(model, texts)
    rerank_inputs = []
    def rerank(query, candidates, key, top_k):
        rerank_inputs.append((query, [item.get("retrieval_text") or item[key] for item in candidates]))
        return [{**item, "rerank_backend": "fastembed"} for item in candidates[:top_k]]
    with evaluation.configured(qdrant_enabled=False, rag_parent_child_expand_enabled=False, rag_parallel_workers=1):
        result = evaluation.run_variant(store, fixture, model, vectors, records, reranker=rerank, repeats=2)
    assert result["aggregate"]["recall_at_5"] == 1
    assert result["aggregate"]["source_span_validity"] == 1
    assert result["actual_rerank_input_min"] == result["actual_rerank_input_max"] == 1
    assert len(rerank_inputs) == 3  # First query plus two warm repetitions.
    assert all(query == "needle" for query, _ in rerank_inputs)
    captured = str(model.inputs) + str(rerank_inputs)
    assert "GOLD_MUST_NOT_ENTER_MODEL" not in captured
    assert "EXPECTED_MUST_NOT_ENTER_MODEL" not in captured


def test_cross_encoder_group_rejects_lexical_fallback():
    evaluation = _script()
    fixture = _fixture()
    store = evaluation.load_module(evaluation.ROOT / "app/rag/store.py", "evaluation_contract_fallback_store")
    records, texts = evaluation.populate_context(store, fixture, enhanced=True)
    model = FakeEmbedding()
    vectors = evaluation.embed_documents(model, texts)

    def fallback(_query, candidates, _key, top_k):
        return [{**item, "rerank_backend": "lexical"} for item in candidates[:top_k]]

    with evaluation.configured(qdrant_enabled=False, rag_parent_child_expand_enabled=False, rag_parallel_workers=1):
        with pytest.raises(ValueError, match="must not use a reranker fallback"):
            evaluation.run_variant(store, fixture, model, vectors, records, reranker=fallback, repeats=1)


def test_vector_replay_retains_scope_and_exact_location():
    evaluation = _script()
    fixture = _fixture()
    store = evaluation.load_module(evaluation.ROOT / "app/rag/store.py", "evaluation_contract_vector_store")
    records, texts = evaluation.populate_context(store, fixture, enhanced=False)
    model = FakeEmbedding()
    search = evaluation.vector_search(model, records, evaluation.embed_documents(model, texts))
    assert search("needle", ["other-kb"], 5) == []
    row = search("needle", ["kb-a"], 5)[0]
    assert row["backend"] == "local_fastembed_cosine"
    assert evaluation.score_case(fixture["cases"][0], [row], fixture["chunks"])["source_span_validity"] == 1


def test_append_requires_matching_provenance_and_preserves_existing_measurements():
    from copy import deepcopy

    evaluation = _script()
    planned = {"fixture_sha256": "fixture", "source_sha256": {"store": "code"}, "configuration": {"limit": 18},
               "embedding_model": "local-model", "threads": 2, "offline_only": True, "qdrant_network": False}
    existing = {**deepcopy(planned), "variants": {
        "before_batch": {"rows": ["original"]}, "context": {}, "context_fastembed_rerank": {}}}
    original = deepcopy(existing)
    evaluation.validate_append_results(existing, planned)
    assert existing == original

    changed = deepcopy(existing)
    changed["source_sha256"]["store"] = "different code"
    with pytest.raises(ValueError, match="provenance"):
        evaluation.validate_append_results(changed, planned)
    changed = deepcopy(existing)
    changed["variants"]["raw_fastembed_rerank"] = {}
    with pytest.raises(ValueError, match="refuse to overwrite"):
        evaluation.validate_append_results(changed, planned)
