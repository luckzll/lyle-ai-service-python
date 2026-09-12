"""Repeatable offline cache benchmark; no LLM, network or production documents.

Requires an already downloaded BAAI/bge-small-zh-v1.5 model in --cache-dir.
The embedding measurements use actual ONNX inference. Candidate measurements
use the actual keyword store with Qdrant disabled, isolating the candidate cache.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import statistics
import sys
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _offline(*_args, **_kwargs):
    raise RuntimeError("Network access is disabled for this benchmark; pre-download model assets")


def _timed(function):
    started = perf_counter()
    value = function()
    return value, round((perf_counter() - started) * 1000, 4)


def _summary(samples):
    values = sorted(sample["milliseconds"] for sample in samples)
    position = (len(values) - 1) * 0.95
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return {
        "samples": samples,
        "medianMs": round(statistics.median(values), 4),
        "p95Ms": round(values[lower] + (values[upper] - values[lower]) * (position - lower), 4),
    }


def run(cache_dir: Path, samples: int, corpus_size: int):
    # These hooks also catch accidental future network use in the store/model.
    with ExitStack() as stack:
        stack.enter_context(patch.object(socket.socket, "connect", _offline))
        stack.enter_context(patch.object(socket, "create_connection", _offline))
        stack.enter_context(patch.object(socket, "getaddrinfo", _offline))
        from fastembed import TextEmbedding
        from app.core.config import settings
        from app.core.runtime_config import current_rag_config
        from app.embedding import client as embedding
        from app.rag import retriever, store
        from app.rag.cache_scope import (
            allowed_document_ids, allowed_document_revisions, invalidate_candidate_cache, retrieval_cache_scope,
        )

        for field, value in {
            "embedding_provider": "fastembed", "embedding_model": "BAAI/bge-small-zh-v1.5", "embedding_dimension": 512,
            "embedding_cache_enabled": True, "qdrant_enabled": False, "rerank_enabled": False,
            "rag_optimized_retrieval_enabled": True, "rag_candidate_cache_enabled": True,
            "rag_query_rewrite_enabled": False, "rag_sufficiency_check_enabled": False,
            "rag_adaptive_retrieval_enabled": False, "rag_parent_child_expand_enabled": False,
            "rag_retrieval_candidate_limit": 20, "rag_context_top_k": 5,
        }.items():
            stack.enter_context(patch.object(settings, field, value))
        model, model_init_ms = _timed(lambda: TextEmbedding(
            model_name=settings.embedding_model, cache_dir=str(cache_dir), threads=2, local_files_only=True,
        ))
        stack.enter_context(patch.object(embedding, "_fastembed_model", lambda: model))
        # Exclude model loading and one ONNX kernel warmup from cache-miss cost.
        embedding.embed_text("缓存基准预热")
        original_embed = embedding._embed_with_fastembed
        embed_calls = [0]

        def measured_embed(text):
            embed_calls[0] += 1
            return original_embed(text)

        stack.enter_context(patch.object(embedding, "_embed_with_fastembed", measured_embed))
        question = "RAG 检索缓存如何根据知识库版本失效？"
        embedding_samples = {"cold": [], "warm": []}
        with retrieval_cache_scope("benchmark-user", [], conversation_id="benchmark-conversation"):
            for index in range(samples):
                embedding.clear_embedding_query_cache()
                before = embed_calls[0]
                cold, cold_ms = _timed(lambda: embedding.embed_query(question))
                embedding_samples["cold"].append({"sample": index + 1, "milliseconds": cold_ms, "modelCalls": embed_calls[0] - before})
                before = embed_calls[0]
                warm, warm_ms = _timed(lambda: embedding.embed_query(question))
                embedding_samples["warm"].append({"sample": index + 1, "milliseconds": warm_ms, "modelCalls": embed_calls[0] - before})
                assert cold == warm and len(cold) == 512

        kb = "benchmark-kb"
        chunks = [store.KnowledgeChunk(
            str(index), kb, f"缓存说明 {index}",
            f"RAG 检索缓存使用知识库版本识别资料更新。第 {index} 项记录删除与重建后的版本变化。",
            start_char=0, end_char=75, retrieval_revision="benchmark-revision",
        ) for index in range(corpus_size)]
        stack.enter_context(patch.object(store, "_chunks", chunks))
        _, index_init_ms = _timed(store.rebuild_keyword_index)
        documents = [item.doc_id for item in chunks]
        original_search = retriever.search_candidates
        recall_calls = [0]

        def measured_search(*args):
            recall_calls[0] += 1
            return original_search(*args)

        stack.enter_context(patch.object(retriever, "search_candidates", measured_search))

        def scope(version, ids):
            return retrieval_cache_scope("benchmark-user", [kb], {kb: version}, "benchmark-owner", "benchmark-conversation",
                                         {kb: ids}, {kb: {doc: "benchmark-revision" for doc in ids}})

        authorization_samples = []
        scope_samples = []

        def build_scope():
            with scope("manifest-one", documents) as context:
                return context

        def authorize_each_document():
            for doc in documents:
                assert doc in allowed_document_ids(kb)
                assert allowed_document_revisions(kb)[doc] == "benchmark-revision"

        for index in range(samples):
            _, build_ms = _timed(build_scope)
            scope_samples.append({"sample": index + 1, "milliseconds": build_ms})
            with scope("manifest-one", documents):
                ids = allowed_document_ids(kb)
                revisions = allowed_document_revisions(kb)
                _, authorization_ms = _timed(authorize_each_document)
                assert ids is allowed_document_ids(kb) and revisions is allowed_document_revisions(kb)
                authorization_samples.append({"sample": index + 1, "milliseconds": authorization_ms,
                                              "documentChecks": len(documents)})

        candidate_samples = {"cold": [], "warm": []}
        with scope("manifest-one", documents):
            for index in range(samples):
                invalidate_candidate_cache()
                before = recall_calls[0]
                cold, cold_ms = _timed(lambda: retriever.retrieve_context_with_diagnostics(question, [kb]))
                candidate_samples["cold"].append({"sample": index + 1, "milliseconds": cold_ms, "recallCalls": recall_calls[0] - before,
                                                   "cache": cold["retrievalPlan"]["candidateCache"]})
                before = recall_calls[0]
                warm, warm_ms = _timed(lambda: retriever.retrieve_context_with_diagnostics(question, [kb]))
                candidate_samples["warm"].append({"sample": index + 1, "milliseconds": warm_ms, "recallCalls": recall_calls[0] - before,
                                                   "cache": warm["retrievalPlan"]["candidateCache"]})
                assert cold["contexts"] == warm["contexts"] and cold["contexts"]
        removed_document = warm["contexts"][0]["doc_id"]
        remaining = [doc for doc in documents if doc != removed_document]
        with scope("manifest-after-delete", remaining):
            before = recall_calls[0]
            changed, changed_ms = _timed(lambda: retriever.retrieve_context_with_diagnostics(question, [kb]))
            changed_calls = recall_calls[0] - before
            assert changed_calls == 1 and removed_document not in {row["doc_id"] for row in changed["contexts"]}
            before = recall_calls[0]
            changed_warm, changed_warm_ms = _timed(lambda: retriever.retrieve_context_with_diagnostics(question, [kb]))
            assert changed_warm["contexts"] == changed["contexts"] and recall_calls[0] == before
        policy = asdict(current_rag_config())
        embedding.clear_embedding_query_cache()
        invalidate_candidate_cache()

    sources = ["app/rag/cache.py", "app/rag/cache_scope.py", "app/embedding/client.py", "app/rag/retriever.py", "app/rag/store.py"]
    return {
        "benchmark": "rag-cache-v2", "network": "blocked", "llmCalls": 0, "samplesPerState": samples,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "fastembed": importlib.metadata.version("fastembed"), "onnxruntime": importlib.metadata.version("onnxruntime")},
        "embedding": {"model": "BAAI/bge-small-zh-v1.5", "dimension": 512, "threads": 2, "modelInitMs": model_init_ms,
                      "cacheDirectory": str(cache_dir), "cold": _summary(embedding_samples["cold"]), "warm": _summary(embedding_samples["warm"]),
                      "vectorsExactlyEqual": True},
        "candidates": {"backend": "actual keyword store; Qdrant disabled", "corpusChunks": corpus_size, "indexInitMs": index_init_ms,
                       "cold": _summary(candidate_samples["cold"]), "warm": _summary(candidate_samples["warm"]),
                       "citationFieldsExactlyEqual": True,
                       "manifestChange": {"milliseconds": changed_ms, "recallCalls": changed_calls, "cache": changed["retrievalPlan"]["candidateCache"],
                                          "deletedDocumentExcluded": True, "nextWarmMs": changed_warm_ms, "nextWarmRecallCalls": 0}},
        "authorization": {"documents": corpus_size, "immutableViewsReused": True,
                          "scopeBuild": _summary(scope_samples), "allDocumentLookups": _summary(authorization_samples)},
        "limits": "Cache misses exclude one-time model/index initialization. This isolates cache work and is not end-to-end production RAG latency.",
        "policy": policy,
        "sourceSha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sources},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "lyle-rag-models")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--corpus-size", type=int, default=600)
    parser.add_argument("--output", type=Path, default=ROOT / "tests" / "fixtures" / "rag_cache_results_v2.json")
    args = parser.parse_args()
    if args.samples < 2 or args.corpus_size < 10:
        parser.error("Use at least two samples and ten synthetic chunks")
    result = run(args.cache_dir.resolve(), args.samples, args.corpus_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "embeddingColdMedianMs": result["embedding"]["cold"]["medianMs"],
                      "embeddingWarmMedianMs": result["embedding"]["warm"]["medianMs"],
                      "candidateColdMedianMs": result["candidates"]["cold"]["medianMs"],
                      "candidateWarmMedianMs": result["candidates"]["warm"]["medianMs"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
