"""Reproducible offline RAG comparison. No provider calls or model downloads by default.

Run with the project Python and an immutable pre-change app/ snapshot:
  python scripts/benchmark_rag_optimization.py --baseline-root C:/Temp/lyle-rag-baseline-... --output tests/fixtures/rag_results_v1.json

The optional generator receives questions and actual retrieved contexts, never the gold answers:
  --generator package.module:function --allow-external-generation
It must return {"answer": str, "citations": [{"doc_id": str, "chunkIndex": str, "quote": str}]};
answer correctness still requires human or independently configured model evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import logging
import math
from pathlib import Path
import platform
import re
import statistics
import sys
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIXTURE = ROOT / "tests/fixtures/rag_evaluation_v1.json"


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    location = (len(ordered) - 1) * quantile
    lower = math.floor(location)
    upper = math.ceil(location)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower), 3)


def load_fixture(path: Path = FIXTURE) -> dict:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    documents = {(doc["knowledge_base_id"], doc["doc_id"]): doc for doc in fixture["documents"]}
    chunks = {chunk["key"]: chunk for chunk in fixture["chunks"]}
    if len(chunks) != len(fixture["chunks"]):
        raise ValueError("duplicate fixture chunk key")
    for chunk in chunks.values():
        source = documents[(chunk["knowledge_base_id"], chunk["doc_id"])]["text"]
        if source[chunk["start_char"]:chunk["end_char"]] != chunk["chunk"]:
            raise ValueError(f"invalid exact source offsets: {chunk['key']}")
    for case in fixture["cases"]:
        if bool(case["evidence"]) != case["answerable"]:
            raise ValueError(f"inconsistent answerability: {case['id']}")
        for evidence in case["evidence"]:
            for key in evidence["acceptable_chunk_keys"]:
                if key not in chunks or evidence["quote"] not in chunks[key]["chunk"]:
                    raise ValueError(f"invalid manual evidence: {case['id']} {key}")
                if chunks[key]["knowledge_base_id"] not in case["knowledge_base_ids"]:
                    raise ValueError(f"gold is outside authorized KB: {case['id']} {key}")
    return fixture


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"module cannot be loaded: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def configured(**overrides):
    from app.core.config import settings
    previous = {key: (key in settings.__dict__, settings.__dict__.get(key)) for key in overrides}
    settings.__dict__.update(overrides)
    try:
        yield
    finally:
        for key, (present, value) in previous.items():
            if present:
                settings.__dict__[key] = value
            else:
                settings.__dict__.pop(key, None)


def result_for(chunk: dict, score: float, backend: str = "tfidf_replay") -> dict[str, str]:
    return {"doc_id": chunk["doc_id"], "knowledge_base_id": chunk["knowledge_base_id"], "title": chunk["title"], "chunk": chunk["chunk"],
            "score": str(score), "backend": backend, "hitModes": "vector" if backend == "tfidf_replay" else backend,
            "chunkIndex": str(chunk["chunk_index"]), "parentTitle": chunk["parent_title"],
            "startChar": str(chunk["start_char"]), "endChar": str(chunk["end_char"])}


def tfidf_vectors(chunks: list[dict], questions: list[str]):
    """Lexical candidate substitute, deliberately independent of the manual gold labels."""
    def terms(text: str):
        text = text.lower()
        return re.findall(r"[a-z0-9_]+", text) + re.findall(r"[\u4e00-\u9fff]", text) + [
            text[index:index + 2] for index in range(len(text) - 1)
            if re.fullmatch(r"[\u4e00-\u9fff]{2}", text[index:index + 2])]
    counters = [Counter(terms(item["chunk"])) for item in chunks]
    df = Counter(term for counter in counters for term in counter)
    def vector(counter):
        values = {term: (1 + math.log(count)) * (1 + math.log((len(chunks) + 1) / (df[term] + 1))) for term, count in counter.items()}
        norm = math.sqrt(sum(value * value for value in values.values())) or 1
        return {term: value / norm for term, value in values.items()}
    documents = [vector(counter) for counter in counters]
    replay = {}
    for question in questions:
        query = vector(Counter(terms(question)))
        ranked = sorted([(sum(value * doc.get(term, 0) for term, value in query.items()), index)
                         for index, doc in enumerate(documents)], key=lambda pair: (-pair[0], pair[1]))
        replay[question] = [(score, chunks[index]) for score, index in ranked if score > 0]
    return replay


def populate(store, chunks: list[dict]):
    records = [store.KnowledgeChunk(**{key: value for key, value in chunk.items() if key != "key"}) for chunk in chunks]
    with store._lock:
        store._chunks[:] = records
    # A new process/module is used for each cold build. Query-time snapshot validation must see this corpus.
    return records


def exact_key(result: dict, chunks: list[dict], knowledge_base_ids: list[str]) -> tuple[str | None, bool]:
    matches = [item for item in chunks if item["doc_id"] == result.get("doc_id")
               and item["chunk"] == result.get("chunk")
               and (not result.get("knowledge_base_id") or item["knowledge_base_id"] == str(result["knowledge_base_id"]))
               and str(item["chunk_index"]) == str(result.get("chunkIndex"))
               and str(item["start_char"]) == str(result.get("startChar"))
               and str(item["end_char"]) == str(result.get("endChar"))]
    authorized = [item for item in matches if item["knowledge_base_id"] in knowledge_base_ids]
    if len(authorized) == 1:
        return authorized[0]["key"], False
    return None, bool(matches and not authorized)


def score_case(case: dict, contexts: list[dict], chunks: list[dict]) -> dict:
    keys = []
    invalid = leaks = 0
    for context in contexts[:5]:
        key, leaked = exact_key(context, chunks, case["knowledge_base_ids"])
        keys.append(key)
        invalid += int(key is None)
        leaks += int(leaked)
    gold = case["evidence"]
    supported = [any(key in evidence["acceptable_chunk_keys"] and evidence["quote"] in context.get("chunk", "")
                     for key, context in zip(keys, contexts[:5])) for evidence in gold]
    relevant = set(key for evidence in gold for key in evidence["acceptable_chunk_keys"])
    first_rank = next((index for index, key in enumerate(keys, 1) if key in relevant), None)
    # Recall is evidence-group recall: duplicated alternative locations satisfy one fact only.
    return {"recall_at_5": sum(supported) / len(gold) if gold else None,
            "mrr": 1 / first_rank if first_rank else 0.0 if gold else None,
            "all_required_evidence_supported": all(supported) if gold else None,
            "evidence_precision_at_5": sum(key in relevant for key in keys) / len(keys) if keys and gold else None,
            "source_span_validity": (len(keys) - invalid) / len(keys) if keys else 1.0,
            "invalid_source_count": invalid, "kb_leak_count": leaks,
            "no_answer_returned_contexts": bool(contexts) if not gold else None,
            "retrieved_keys": keys, "missing_evidence": [evidence for evidence, found in zip(gold, supported) if not found]}


def aggregate(rows: list[dict]) -> dict:
    def mean(key):
        values = [row[key] for row in rows if row.get(key) is not None]
        return round(statistics.mean(values), 6) if values else None
    return {"cases": len(rows), "answerable_cases": sum(row["recall_at_5"] is not None for row in rows),
            "recall_at_5": mean("recall_at_5"), "mrr": mean("mrr"),
            "all_required_evidence_supported": mean("all_required_evidence_supported"),
            "evidence_precision_at_5": mean("evidence_precision_at_5"),
            "source_span_validity": mean("source_span_validity"),
            "kb_leak_count": sum(row["kb_leak_count"] for row in rows),
            "no_answer_returned_context_rate": mean("no_answer_returned_contexts"),
            "latency_ms_p50": percentile([row["latency_ms"] for row in rows], .5),
            "latency_ms_p95": percentile([row["latency_ms"] for row in rows], .95)}


def evaluate(store, chunks: list[dict], cases: list[dict], replay: dict | None, retriever=None):
    populate(store, chunks)
    store._qdrant_search = lambda query, kb_ids, limit: [result_for(chunk, score) for score, chunk in (replay or {}).get(query, [])
                                                       if chunk["knowledge_base_id"] in kb_ids][:limit]
    rows = []
    for case in cases:
        start = perf_counter()
        diagnostics = None
        if retriever is None:
            contexts = store.search(case["question"], case["knowledge_base_ids"], top_k=5)
        else:
            diagnostics = retriever.retrieve_context_with_diagnostics(case["question"], case["knowledge_base_ids"])
            contexts = diagnostics["contexts"]
        elapsed = (perf_counter() - start) * 1000
        rows.append({"case_id": case["id"], "category": case["category"], "question": case["question"],
                     "latency_ms": round(elapsed, 3), **score_case(case, contexts, chunks), "contexts": contexts,
                     **({"queries": diagnostics["queries"], "retrieval_plan": diagnostics.get("retrievalPlan", {})}
                        if diagnostics is not None else {})})
    badcases = [row for row in rows if row["recall_at_5"] not in {None, 1.0} or row["invalid_source_count"]
                or row["kb_leak_count"] or row["no_answer_returned_contexts"]]
    return {"aggregate": aggregate(rows), "by_category": {category: aggregate([r for r in rows if r["category"] == category])
            for category in sorted({row["category"] for row in rows})}, "badcase_ids": [row["case_id"] for row in badcases], "rows": rows}


def scaled_chunks(chunks: list[dict], count: int) -> list[dict]:
    result = list(chunks)
    while len(result) < count:
        number = len(result)
        text = f"测试附录 {number}：本段记录设备巡检、会议归档、文件批次与备用说明。编号 AUX-{number:04d}。"
        result.append({"key": f"kb-guide/noise-{number}#1", "doc_id": f"noise-{number}", "knowledge_base_id": "kb-guide",
                       "title": f"附录 {number}", "chunk": text, "chunk_index": 1,
                       "parent_title": "测试附录", "start_char": 0, "end_char": len(text)})
    return result[:count]


def benchmark_keyword(loader, chunks, sizes, repeats):
    rows = []
    for size in sizes:
        store = loader()
        populate(store, scaled_chunks(chunks, size))
        question = "星桥知识库文件备份保留几天"
        start = perf_counter()
        cold_result = store._keyword_search(question, ["kb-guide"], 5)
        cold_ms = (perf_counter() - start) * 1000
        samples = []
        for _ in range(repeats):
            start = perf_counter()
            warm = store._keyword_search(question, ["kb-guide"], 5)
            samples.append(round((perf_counter() - start) * 1000, 3))
            if warm != cold_result:
                raise AssertionError("warm and cold keyword results differ")
        build_ms = None
        if hasattr(store, "rebuild_keyword_index"):
            build_store = loader()
            populate(build_store, scaled_chunks(chunks, size))
            start = perf_counter()
            build_store.rebuild_keyword_index()
            build_ms = round((perf_counter() - start) * 1000, 3)
        rows.append({"chunks": size, "question": question, "cold_query_including_build_ms": round(cold_ms, 3),
                     "separate_cold_index_build_ms": build_ms,
                     "warm_samples_ms": samples, "warm_p50_ms": percentile(samples, .5), "warm_p95_ms": percentile(samples, .95),
                     "result_keys": [(item.get("doc_id"), item.get("chunkIndex")) for item in cold_result]})
    return rows


def replay_queries(question: str) -> list[str]:
    """A fixed duplicate-rewrite replay, not an oracle or a substitute for useful LLM rewrites."""
    return [question, question.rstrip("?？。") + "？", "  " + question + "  "]


def configure_retriever_replay(retriever, store, rewrite=replay_queries):
    retriever.search = store.search
    if hasattr(store, "search_candidates"):
        retriever.search_candidates = store.search_candidates
    retriever.rewrite_rag_queries = rewrite
    retriever.judge_rag_sufficiency = lambda question, contexts: {
        "sufficient": False, "method": "offline_not_judged", "reason": "No auxiliary model called in this replay."}
    return retriever


def paired_generation_inputs(fixture: dict, baseline: dict, optimized: dict) -> dict:
    """Gold is separate review-only data and must never be passed to the generation callback."""
    selected = {"R011", "R013", "R014", "R020", "R021", "R024", "R033", "R038", "R043"}
    before = {row["case_id"]: row for row in baseline["rows"]}
    after = {row["case_id"]: row for row in optimized["rows"]}
    return {"fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
            "selection": "Initial six category representatives plus R013/R014 ranking regressions and R021 known missing-history evidence, selected after retrieval results for diagnostic sensitivity. Not a random quality estimate.",
            "instructions": "Generate using only question, history and the selected contexts. review_only is for independent assessment after generation; never put it in the generation prompt.",
            "cases": [{"case_id": case["id"], "category": case["category"], "question": case["question"],
                       "history": case.get("conversation_history", []),
                       "baseline_contexts": before[case["id"]]["contexts"],
                       "optimized_contexts": after[case["id"]]["contexts"],
                       "review_only": {"answerable": case["answerable"], "expected_evidence": case["evidence"],
                                       "expected_answer_facts": [evidence["quote"] for evidence in case["evidence"]]}}
                      for case in fixture["cases"] if case["id"] in selected]}


def external_generation(generator_spec: str, cases: list[dict], retrieval: dict) -> list[dict]:
    module, separator, function = generator_spec.partition(":")
    if not separator:
        raise ValueError("generator must be module:function")
    generator = getattr(importlib.import_module(module), function)
    rows = []
    by_case = {row["case_id"]: row for row in retrieval["rows"]}
    for case in cases:
        contexts = by_case[case["id"]]["contexts"]
        start = perf_counter()
        # Never supply gold evidence or an expected answer to the generation function.
        output = generator(question=case["question"], history=case.get("conversation_history", []), contexts=contexts)
        if not isinstance(output, dict) or not isinstance(output.get("answer"), str):
            raise TypeError("generator must return an answer string and optional source citations")
        citations = output.get("citations", [])
        if not isinstance(citations, list):
            raise TypeError("generator citations must be a list")
        valid = [isinstance(citation, dict) and isinstance(citation.get("quote"), str) and bool(citation["quote"])
                 and any(context.get("doc_id") == citation.get("doc_id")
                         and str(context.get("chunkIndex")) == str(citation.get("chunkIndex"))
                         and citation["quote"] in context.get("chunk", "") for context in contexts) for citation in citations]
        rows.append({"case_id": case["id"], "output": output, "latency_ms": round((perf_counter() - start) * 1000, 3),
                     "citation_quotes_exist_in_retrieved_sources": all(valid) if valid else None,
                     "answer_correctness": "NOT_AUTOMATICALLY_JUDGED; independent human/model assessment required"})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 300])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--generator")
    parser.add_argument("--allow-external-generation", action="store_true")
    parser.add_argument("--paired-generation-output", type=Path)
    args = parser.parse_args()
    if args.generator and not args.allow_external_generation:
        parser.error("external generation is opt-in; supply --allow-external-generation only after authorizing the provider/cost")
    fixture = load_fixture(args.fixture)
    logging.disable(logging.INFO)
    serial = 0
    def baseline():
        nonlocal serial
        serial += 1
        base = args.baseline_root / "app/rag"
        reranker = load_module(base / "reranker.py", f"rag_eval_baseline_reranker_{serial}")
        store = load_module(base / "store.py", f"rag_eval_baseline_store_{serial}")
        store.rerank_items = reranker.rerank_items
        return store
    def optimized():
        nonlocal serial
        serial += 1
        return load_module(ROOT / "app/rag/store.py", f"rag_eval_current_store_{serial}")
    def pipeline(store, is_baseline: bool, *, rules: bool = False):
        base = args.baseline_root if is_baseline else ROOT
        retriever = load_module(base / "app/rag/retriever.py", f"rag_eval_retriever_{serial}")
        if not is_baseline:
            retriever.rerank_items = store.rerank_items
        rewrite = (lambda question: rule_modules[is_baseline]._rule_queries(question, 3)) if rules else replay_queries
        return configure_retriever_replay(retriever, store, rewrite)
    rule_modules = {is_baseline: load_module(base / "app/chains/query_rewrite_chain.py", f"rag_eval_rules_{is_baseline}")
                    for is_baseline, base in [(True, args.baseline_root), (False, ROOT)]}
    query_set = [query for case in fixture["cases"] for query in replay_queries(case["question"])]
    query_set.extend(query for rules in rule_modules.values() for case in fixture["cases"]
                     for query in rules._rule_queries(case["question"], 3))
    replay = tfidf_vectors(fixture["chunks"], query_set)
    result: dict[str, Any] = {"fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_sha256": {name: hashlib.sha256((args.baseline_root / f"app/rag/{name}.py").read_bytes()).hexdigest()
                            for name in ["store", "retriever", "reranker"]},
        "current_source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in [ROOT / "app/rag" / name for name in
                                               ["store.py", "retriever.py", "reranker.py", "keyword_index.py", "concurrency.py", "config.py", "identity.py"]]
                                  + [ROOT / "app/chains/query_rewrite_chain.py"]},
        "python": platform.python_version(), "platform": platform.platform(), "corpus_chunks": len(fixture["chunks"]),
        "evaluation_scope": "Offline retrieval CPU replay. TF-IDF candidate substitute is lexical, not semantic embeddings. No model answer-quality validation.",
        "timing_scope": "Local search/merge/parent expansion/lexical rerank only; vector candidates precomputed; excludes embeddings, provider/network latency, rewrite and sufficiency models.",
        "pipeline_rewrite_scope": "Fixed original question plus punctuation/whitespace duplicates. Measures deduplication and global rerank with unchanged information, not model rewrite quality. Ambiguous history is not resolved by this replay.",
        "rule_pipeline_scope": "Actual old/current _rule_queries helper, limit 3, with no model calls. rule_no_rerank variants retain the existing disabled-reranker default; rule_pipeline variants explicitly enable lexical reranking.",
        "variants": {}, "keyword_cpu": {}, "generation": "NOT_RUN"}
    common = dict(qdrant_enabled=False, rerank_enabled=True, rerank_provider="lexical", rerank_candidate_limit=18,
                  rag_parent_child_expand_enabled=True, rag_parent_child_sibling_limit=2,
                  rag_query_rewrite_enabled=False, rag_sufficiency_check_enabled=False,
                  rag_auxiliary_model_enabled=False, rag_context_top_k=5, rag_multi_query_count=3,
                  rag_retrieval_candidate_limit=18, rag_parallel_workers=4, rag_global_rerank_limit=40,
                  rag_adaptive_retrieval_enabled=False)
    result["configuration"] = common
    with configured(**common, rag_optimized_retrieval_enabled=False, rag_fusion_mode="legacy"):
        result["variants"]["baseline_keyword"] = evaluate(baseline(), fixture["chunks"], fixture["cases"], None)
        result["variants"]["baseline_tfidf_hybrid"] = evaluate(baseline(), fixture["chunks"], fixture["cases"], replay)
        store = baseline()
        result["variants"]["baseline_pipeline_replay"] = evaluate(store, fixture["chunks"], fixture["cases"], replay, pipeline(store, True))
        store = baseline()
        result["variants"]["baseline_rule_pipeline"] = evaluate(store, fixture["chunks"], fixture["cases"], replay, pipeline(store, True, rules=True))
        with configured(rerank_enabled=False):
            store = baseline()
            result["variants"]["baseline_rule_no_rerank_pipeline"] = evaluate(store, fixture["chunks"], fixture["cases"], replay, pipeline(store, True, rules=True))
        result["keyword_cpu"]["baseline"] = benchmark_keyword(baseline, fixture["chunks"], args.sizes, args.repeats)
    if not args.baseline_only:
        for mode in ["legacy", "rrf"]:
            with configured(**common, rag_optimized_retrieval_enabled=True, rag_fusion_mode=mode):
                if mode == "legacy":
                    result["variants"]["optimized_keyword"] = evaluate(optimized(), fixture["chunks"], fixture["cases"], None)
                    result["keyword_cpu"]["optimized"] = benchmark_keyword(optimized, fixture["chunks"], args.sizes, args.repeats)
                result["variants"][f"optimized_{mode}_tfidf_hybrid"] = evaluate(optimized(), fixture["chunks"], fixture["cases"], replay)
                store = optimized()
                result["variants"][f"optimized_{mode}_pipeline_replay"] = evaluate(store, fixture["chunks"], fixture["cases"], replay, pipeline(store, False))
                store = optimized()
                result["variants"][f"optimized_{mode}_rule_pipeline"] = evaluate(store, fixture["chunks"], fixture["cases"], replay, pipeline(store, False, rules=True))
                with configured(rerank_enabled=False):
                    store = optimized()
                    result["variants"][f"optimized_{mode}_rule_no_rerank_pipeline"] = evaluate(store, fixture["chunks"], fixture["cases"], replay, pipeline(store, False, rules=True))
    if args.paired_generation_output:
        if args.baseline_only:
            parser.error("paired generation input requires an optimized comparison")
        paired = paired_generation_inputs(fixture, result["variants"]["baseline_rule_no_rerank_pipeline"],
                                          result["variants"]["optimized_legacy_rule_no_rerank_pipeline"])
        paired["retrieval_variants"] = ["baseline_rule_no_rerank_pipeline", "optimized_legacy_rule_no_rerank_pipeline"]
        args.paired_generation_output.parent.mkdir(parents=True, exist_ok=True)
        args.paired_generation_output.write_text(json.dumps(paired, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.generator:
        chosen = result["variants"].get("optimized_legacy_tfidf_hybrid", result["variants"]["baseline_tfidf_hybrid"])
        result["generation"] = external_generation(args.generator, fixture["cases"], chosen)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "variants": {key: value["aggregate"] for key, value in result["variants"].items()},
                      "keyword_cpu": result["keyword_cpu"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
