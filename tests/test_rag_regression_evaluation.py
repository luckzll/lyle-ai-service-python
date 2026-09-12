"""Protect the frozen evaluation and its evidence-based scoring from optimistic mistakes."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("rag_evaluation_benchmark", ROOT / "scripts/benchmark_rag_optimization.py")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


@pytest.fixture
def corpus():
    return benchmark.load_fixture()


def test_frozen_corpus_has_verifiable_sources_and_no_gold_outside_authorized_kb(corpus):
    assert len(corpus["cases"]) == 50
    assert len(corpus["chunks"]) == 40
    assert len(corpus["documents"]) == 12
    assert sum(case["answerable"] for case in corpus["cases"]) == 44
    assert hashlib.sha256(benchmark.FIXTURE.read_bytes()).hexdigest() == "12cd7ae7265158ec60aa5adc54ca75567fda420da286a7cc1e1cf32c77cf8f15"


def test_frozen_corpus_rejects_invented_quote_even_with_a_real_document(corpus, tmp_path):
    corpus["cases"][0]["evidence"][0]["quote"] = "A fact that is not in this document."
    invalid_fixture = tmp_path / "invalid.json"
    invalid_fixture.write_text(json.dumps(corpus), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid manual evidence"):
        benchmark.load_fixture(invalid_fixture)


def test_exact_source_requires_offsets_not_only_a_matching_title(corpus):
    chunk = corpus["chunks"][0]
    result = benchmark.result_for(chunk, 1)
    assert benchmark.exact_key(result, corpus["chunks"], [chunk["knowledge_base_id"]]) == (chunk["key"], False)
    result["startChar"] = str(chunk["start_char"] + 1)
    assert benchmark.exact_key(result, corpus["chunks"], [chunk["knowledge_base_id"]]) == (None, False)


def test_vector_candidate_replay_preserves_complete_real_source_identity(corpus):
    chunk = corpus["chunks"][0]
    result = benchmark.result_for(chunk, .8)
    assert result["knowledge_base_id"] == chunk["knowledge_base_id"]
    assert result["doc_id"] == chunk["doc_id"]
    assert result["chunkIndex"] == str(chunk["chunk_index"])
    assert result["startChar"] == str(chunk["start_char"])
    assert result["endChar"] == str(chunk["end_char"])


def test_retrieved_source_from_another_kb_is_counted_as_a_leak(corpus):
    chunk = next(item for item in corpus["chunks"] if item["knowledge_base_id"] == "kb-east")
    case = next(item for item in corpus["cases"] if item["id"] == "R043")
    score = benchmark.score_case(case, [benchmark.result_for(chunk, 1)], corpus["chunks"])
    assert score["kb_leak_count"] == 1
    assert score["source_span_validity"] == 0
    assert score["recall_at_5"] == 0


def test_explicit_kb_identity_is_checked_even_when_text_and_document_ids_collide(corpus):
    chunk = corpus["chunks"][0]
    duplicate = {**chunk, "key": "kb-private/collision#1", "knowledge_base_id": "kb-private"}
    result = {**benchmark.result_for(duplicate, 1), "knowledge_base_id": "kb-private"}
    assert benchmark.exact_key(result, [chunk, duplicate], [chunk["knowledge_base_id"]]) == (None, True)


def test_duplicate_source_locations_satisfy_one_evidence_group(corpus):
    case = next(item for item in corpus["cases"] if item["id"] == "R048")
    alternatives = case["evidence"][0]["acceptable_chunk_keys"]
    assert len(alternatives) == 2
    contexts = [benchmark.result_for(chunk, 1) for chunk in corpus["chunks"] if chunk["key"] in alternatives]
    score = benchmark.score_case(case, contexts, corpus["chunks"])
    assert score["recall_at_5"] == 1
    assert score["all_required_evidence_supported"] is True


def test_multi_fact_recall_does_not_claim_full_support_after_one_good_hit(corpus):
    case = next(item for item in corpus["cases"] if item["id"] == "R033")
    target = next(chunk for chunk in corpus["chunks"] if chunk["key"] in case["evidence"][0]["acceptable_chunk_keys"])
    irrelevant = next(chunk for chunk in corpus["chunks"] if chunk["knowledge_base_id"] == "kb-guide"
                      and all(chunk["key"] not in evidence["acceptable_chunk_keys"] for evidence in case["evidence"]))
    score = benchmark.score_case(case, [benchmark.result_for(irrelevant, 1), benchmark.result_for(target, .5)], corpus["chunks"])
    assert score["recall_at_5"] == 1 / len(case["evidence"])
    assert score["all_required_evidence_supported"] is False
    assert score["mrr"] == .5


def test_no_answer_cases_are_not_credited_with_recall_or_answer_correctness(corpus):
    case = next(item for item in corpus["cases"] if item["id"] == "R038")
    score = benchmark.score_case(case, [benchmark.result_for(corpus["chunks"][0], 1)], corpus["chunks"])
    assert score["recall_at_5"] is None
    assert score["mrr"] is None
    assert score["all_required_evidence_supported"] is None
    assert score["no_answer_returned_contexts"] is True


def test_generation_callback_never_receives_gold_and_unsupported_quotes_fail(corpus, monkeypatch):
    case = corpus["cases"][0]
    context = benchmark.result_for(corpus["chunks"][0], 1)
    observed = []

    def generate(**kwargs):
        observed.append(kwargs)
        return {"answer": "This is a test of the adapter, not a generated benchmark answer.",
                "citations": [{"doc_id": context["doc_id"], "chunkIndex": context["chunkIndex"],
                               "quote": "Unsupported invented quotation"}]}

    monkeypatch.setattr(benchmark.importlib, "import_module", lambda name: SimpleNamespace(generate=generate))
    rows = benchmark.external_generation("test_adapter:generate", [case], {"rows": [{"case_id": case["id"], "contexts": [context]}]})
    assert set(observed[0]) == {"question", "history", "contexts"}
    assert rows[0]["citation_quotes_exist_in_retrieved_sources"] is False
    assert rows[0]["answer_correctness"].startswith("NOT_AUTOMATICALLY_JUDGED")


def test_paired_generation_export_keeps_gold_separate_for_review(corpus):
    retrieval = {"rows": [{"case_id": case["id"], "contexts": []} for case in corpus["cases"]]}
    paired = benchmark.paired_generation_inputs(corpus, retrieval, retrieval)
    assert len(paired["cases"]) == 9
    assert len({case["category"] for case in paired["cases"]}) == 6
    assert all("expected_evidence" in case["review_only"] for case in paired["cases"])
    assert all("expected_evidence" not in case for case in paired["cases"])


def test_keyword_index_keeps_frozen_baseline_top_five_on_larger_corpus(corpus, monkeypatch):
    from app.rag import store
    from app.rag.keyword_index import KeywordIndex, VersionedChunkList

    monkeypatch.setattr(store, "_chunks", VersionedChunkList(lock=store._lock))
    monkeypatch.setattr(store, "_keyword_index", KeywordIndex())
    benchmark.populate(store, benchmark.scaled_chunks(corpus["chunks"], 300))
    actual = store._keyword_search("星桥知识库文件备份保留几天", ["kb-guide"], 5)
    baseline = json.loads((ROOT / "tests/fixtures/rag_baseline_results_v1.json").read_text(encoding="utf-8"))
    expected = baseline["keyword_cpu"]["baseline"][-1]["result_keys"]
    assert [[row["doc_id"], row["chunkIndex"]] for row in actual] == expected
