"""Compare frozen RAG evidence with real local vectors, without model downloads.

This is a retrieval experiment, not a generated-answer quality benchmark. It
keeps the original 50 questions/chunks/gold evidence and isolates deterministic
chunk context and optional local cross-encoder reranking. No Qdrant, database,
chat model, query rewrite, history resolution or application billing is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_rag_optimization import FIXTURE, aggregate, configured, load_fixture, load_module, percentile, result_for, score_case


def populate_context(store, fixture: dict, *, enhanced: bool):
    from app.rag.chunk_context import build_chunk_context, document_metadata

    documents = {(doc["knowledge_base_id"], doc["doc_id"]): doc for doc in fixture["documents"]}
    records, texts = [], []
    for source in fixture["chunks"]:
        values = {key: value for key, value in source.items() if key != "key"}
        if enhanced:
            doc = documents[(source["knowledge_base_id"], source["doc_id"])]
            context = build_chunk_context(doc["title"], source["parent_title"], source["chunk"],
                                          document_metadata(doc["text"]), max_chars=480)
            values.update(retrieval_text=context.retrieval_text, context_version=context.version,
                          document_title=doc["title"], context_sources=context.sources)
        record = store.KnowledgeChunk(**values)
        records.append(record)
        texts.append(getattr(record, "retrieval_text", "") or record.chunk)
    with store._lock:
        store._chunks[:] = records
    return records, texts


def embed_documents(embedder, texts: list[str]):
    import numpy as np

    vectors = np.asarray(list(embedder.embed(texts, batch_size=16)), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts) or not np.isfinite(vectors).all():
        raise ValueError("Invalid local document embeddings")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("Empty local document embedding")
    return vectors / norms


def vector_search(embedder, records, vectors):
    """Actual query embedding plus local cosine; the scope filter mirrors Qdrant."""
    import numpy as np

    def search(question, knowledge_base_ids, limit):
        query = np.asarray(next(iter(embedder.embed([question], batch_size=1))), dtype=np.float32)
        norm = np.linalg.norm(query)
        if not np.isfinite(query).all() or norm <= 0:
            raise ValueError("Invalid local query embedding")
        scores = vectors @ (query / norm)
        allowed = set(knowledge_base_ids)
        rows = []
        for index in np.argsort(-scores, kind="stable"):
            record = records[int(index)]
            if record.knowledge_base_id not in allowed:
                continue
            source = {"doc_id": record.doc_id, "knowledge_base_id": record.knowledge_base_id,
                      "title": record.title, "chunk": record.chunk, "chunk_index": record.chunk_index,
                      "parent_title": record.parent_title, "start_char": record.start_char, "end_char": record.end_char}
            row = result_for(source, float(scores[index]), "local_fastembed_cosine")
            row.update(hitModes="vector", retrieval_text=getattr(record, "retrieval_text", ""),
                       contextVersion=getattr(record, "context_version", "raw-v1"))
            rows.append(row)
            if len(rows) >= limit:
                break
        return rows
    return search


def run_variant(store, fixture, embedder, vectors, records, *, reranker=None, repeats=2):
    store._qdrant_search = vector_search(embedder, records, vectors)
    last_selection = {}

    def retrieve(question, kb_ids):
        candidates = store.search_candidates(question, kb_ids, 18)
        last_selection.update(candidate_count=len(candidates), rerank_input_count=min(40, len(candidates)) if reranker else 0)
        if reranker is None:
            return candidates[:5]
        selected = reranker(question, candidates[:40], "chunk", 5)
        if any(row.get("rerank_backend") != "fastembed" for row in selected):
            raise ValueError("Local cross-encoder evaluation must not use a reranker fallback")
        return selected

    first = fixture["cases"][0]
    started = perf_counter()
    retrieve(first["question"], first["knowledge_base_ids"])
    first_query_ms = round((perf_counter() - started) * 1000, 3)
    rows = []
    for case in fixture["cases"]:
        samples = []
        contexts = None
        for _ in range(max(1, min(5, repeats))):
            started = perf_counter()
            current = retrieve(case["question"], case["knowledge_base_ids"])
            samples.append(round((perf_counter() - started) * 1000, 3))
            if contexts is not None and [(row["doc_id"], row.get("chunkIndex"), row["chunk"]) for row in contexts] != [
                    (row["doc_id"], row.get("chunkIndex"), row["chunk"]) for row in current]:
                raise ValueError("Unstable deterministic local retrieval")
            contexts = current
        # Expected evidence is used only after retrieval for scoring.
        rows.append({"case_id": case["id"], "category": case["category"], "question": case["question"],
                     "latency_ms": percentile(samples, .5), "latency_samples_ms": samples,
                     **last_selection,
                     **score_case(case, contexts, fixture["chunks"]), "contexts": contexts})
    return {"aggregate": aggregate(rows), "first_query_ms": first_query_ms,
            "warm_repeats_per_case": max(1, min(5, repeats)),
            "candidate_limit_per_channel": 18, "rerank_input_cap": 40 if reranker else 0,
            "actual_rerank_input_min": min(row["rerank_input_count"] for row in rows),
            "actual_rerank_input_max": max(row["rerank_input_count"] for row in rows),
            "by_category": {category: aggregate([row for row in rows if row["category"] == category])
                            for category in sorted({row["category"] for row in rows})}, "rows": rows}


def render_report(result: dict) -> str:
    lines = ["# 分块上下文与真实本地向量评测", "",
             "使用原有 50 题冻结证据集，问题和原始片段保持不变；期望证据只在检索完成后用于评分。", "",
             "| 方案 | Recall@5 | MRR | 引用位置有效率 | 热查询 p50 / p95（ms） | 首次查询（ms） |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, variant in result.get("variants", {}).items():
        metrics = variant["aggregate"]
        lines.append(f"| {name} | {metrics['recall_at_5']:.4f} | {metrics['mrr']:.4f} | {metrics['source_span_validity']:.4f} | "
                     f"{metrics['latency_ms_p50']:.2f} / {metrics['latency_ms_p95']:.2f} | {variant['first_query_ms']:.2f} |")
    lines.extend(["", "before_batch：本批开始前代码与原文；context：当前代码与上下文增强；raw_fastembed_rerank：当前代码、原文与重排；context_fastembed_rerank：当前代码、增强与重排。",
                  "模型：BAAI/bge-small-zh-v1.5；可选重排：BAAI/bge-reranker-base。模型仅从本地缓存加载。",
                  "使用真实模型向量和本地余弦计算代替 Qdrant 网络调用，关键词召回、融合、相邻片段扩展调用对应版本的实际代码。",
                  "热查询耗时包含每次真实问题向量化和本地召回，重排组还包含真实模型重排；未开启跨请求缓存。",
                  "before_batch、context、raw_fastembed_rerank 的首次查询包含各自关键词索引初建；组合重排组复用 context 的索引和文档向量。模型加载与整库向量化另记，不能解释为线上冷启动或线上延迟。",
                  "固定原有片段是为了隔离上下文增强的效果；新分块器的原文位置修复另由合同测试验证。",
                  "这些组没有使用多轮补全、辅助模型改写或答案生成，不能据此判断最终回答正确率或线上提速。", ""])
    lines.append(f"本次模型初始化 {result.get('embedding_model_load_ms', 0):.2f} ms；整库向量化：" + "、".join(
        f"{name} {variant['corpus_embedding_ms']:.2f} ms" for name, variant in result.get("variants", {}).items() if "corpus_embedding_ms" in variant) + "。")
    for name, variant in result.get("variants", {}).items():
        if variant.get("rerank_input_cap"):
            lines.append(f"{name}：重排预算上限 {variant['rerank_input_cap']}，本数据集实际每题输入 {variant['actual_rerank_input_min']}–{variant['actual_rerank_input_max']} 个候选；所有返回项已核对 rerank_backend=fastembed。")
    if "rerank_model_load_ms" in result:
        lines.append(f"重排模型加载 {result['rerank_model_load_ms']:.2f} ms，单独统计。")
    if result.get("raw_rerank_append"):
        appended = result["raw_rerank_append"]
        lines.append(f"原文重排组为同一份模型、数据和源码的追加消融，前三组测量保持原样；追加进程向量模型加载 {appended['embedding_model_load_ms']:.2f} ms、重排模型加载 {appended.get('rerank_model_load_ms', 0):.2f} ms。")
    repeat_counts = sorted({variant.get("warm_repeats_per_case", 1) for variant in result.get("variants", {}).values()})
    lines.append("每题计时重复次数：" + "、".join(map(str, repeat_counts)) + "；表中 p50 / p95 跨 50 题汇总，小数据集不代表线上负载与尾延迟。")
    if result.get("not_run"):
        lines.append("尚未完成的组：" + ", ".join(f"{name}（{reason}）" for name, reason in result["not_run"].items()))
    if "before_batch" in result.get("variants", {}) and "context" in result["variants"]:
        before = {row["case_id"]: row for row in result["variants"]["before_batch"]["rows"]}
        regressions = [row["case_id"] for row in result["variants"]["context"]["rows"]
                       if row["recall_at_5"] is not None and row["recall_at_5"] < before[row["case_id"]]["recall_at_5"]]
        lines.extend(["", "上下文增强组证据召回下降的题目：" + ("、".join(regressions) if regressions else "无") + "。",
                      "保留这些回退样本；MRR 上升不代表每个问题的证据覆盖都改善。", ""])
        for name, variant in result["variants"].items():
            if name == "before_batch":
                continue
            for row in variant["rows"]:
                if row["recall_at_5"] is None or row["recall_at_5"] >= before[row["case_id"]]["recall_at_5"]:
                    continue
                missing = "；".join(item["quote"] for item in row["missing_evidence"])
                lines.append(f"- {name} / {row['case_id']}：{row['question']} 缺少证据“{missing}”。")
    lines.extend(["", "## 本次发布取舍", "",
                  "继续保持 rag_chunk_context_enabled=False、rerank_enabled=False。本次 50 题证据覆盖尚未证明优于基线，不足以支持默认开启。",
                  "上下文增强改变了候选排序；重排可能补回某些复合问题的证据，也可能把缺少上文的短追问排错。本次没有把期望答案写回问题，也没有根据回退样本调整参数。",
                  "重排使用 2 个 CPU 线程，增加的耗时来自真实 cross-encoder 推理。raw_fastembed_rerank 使用当前 store 与原文片段，单独测重排；context_fastembed_rerank 则测增强和重排组合，两个结果需要分别判断。",
                  "Recall / MRR 只统计 44 道可回答题；无答案题是否返回上下文记录在 JSON，不能仅凭检索指标认定具有拒答能力。引用位置有效只代表片段能对回原文，不代表答案正确。",
                  "启用前还需要独立的真实业务保留集、多轮问题补全与多证据覆盖检查，以及目标服务器上的负载和延迟验证。"])
    if "raw_fastembed_rerank" in result.get("variants", {}) and "before_batch" in result["variants"]:
        raw_metrics = result["variants"]["raw_fastembed_rerank"]["aggregate"]
        baseline_metrics = result["variants"]["before_batch"]["aggregate"]
        lines.extend(["", f"单独重排的 Recall@5 为 {raw_metrics['recall_at_5']:.4f}，MRR 相比基线变化 "
                      f"{(raw_metrics['mrr'] - baseline_metrics['mrr']) * 100:+.2f} 个百分点；"
                      f"本机 p50 为 {raw_metrics['latency_ms_p50'] / 1000:.2f} 秒。"
                      "这一结果支持保留按需重排选项；组合组的退化不能直接归因给单独重排。"])
    lines.extend(["", "## 复现", "",
                  "基线 store.py 与 keyword_index.py 均显式从本批开始前的快照加载，文件 SHA-256 与原始 50 题数据集 SHA-256 保存在结果 JSON。模型只读本地缓存，不下载、不调用生成 API、不写业务库。",
                  "运行命令（在 Python 服务目录）：", "", "```powershell",
                  ".venv-win/Scripts/python.exe scripts/evaluate_rag_context_v2.py --baseline-root C:/Temp/lyle-rag-context-baseline-gfiindva --cache-dir C:/Users/zhl20/.cache/lyle-rag-models --output tests/fixtures/rag_context_results_v2.json --report docs/RAG_CONTEXT_EVALUATION_V2.md --repeats 1 --include-reranker",
                  "```"])
    lines.extend(["", "## 数据版本与部署边界", "",
                  "增强只对重新入库的片段生成新检索文本；旧片段与旧向量保持 raw-v1，不会因为重建内存关键词索引就自动变成新向量。",
                  "Qdrant 点 ID 包含文档 revision，重试只清理相同 revision，避免跨进程的旧任务删除或覆盖新版本。旧 revision 的物理向量需后续 GC，当前只通过可信 manifest 屏蔽，仍占用存储。",
                  "Java 每次从数据库确认权限、已完成文档名单与对应 revision；Python HTTP 知识库入口必须验证完整 manifest。",
                  "发布与回滚须保持 Java/Python 的 manifest 协议同步。不得把不发送 manifest 的旧 Java 与依赖版本隔离的新 Python 混用。",
                  "无 scope 的底层 store helpers 仅用于可信内部工具、兼容测试，不能充当外部 HTTP 回退路径。", ""])
    return "\n".join(lines) + "\n"


def validate_append_results(existing: dict, planned: dict) -> None:
    """Require the same measured code, data and retrieval settings before appending."""
    for field in ("fixture_sha256", "source_sha256", "configuration", "embedding_model", "threads", "offline_only", "qdrant_network"):
        if existing.get(field) != planned[field]:
            raise ValueError("Existing evaluation provenance does not match this append run")
    if not {"before_batch", "context", "context_fastembed_rerank"}.issubset(existing.get("variants", {})):
        raise ValueError("Append requires the completed original three variants")
    if "raw_fastembed_rerank" in existing["variants"]:
        raise ValueError("Raw reranker variant already exists; refuse to overwrite measured results")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--include-reranker", action="store_true")
    parser.add_argument("--append-raw-rerank", action="store_true",
                        help="Append only the raw-chunk reranker ablation to matching existing three-group results")
    args = parser.parse_args()
    # Offline mode is a second guard in addition to FastEmbed's local-only flag.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    logging.disable(logging.INFO)
    fixture = load_fixture(args.fixture)
    from fastembed import TextEmbedding
    started = perf_counter()
    embedder = TextEmbedding(model_name="BAAI/bge-small-zh-v1.5", threads=2,
                             cache_dir=str(args.cache_dir), local_files_only=True)
    model_load_ms = round((perf_counter() - started) * 1000, 3)
    result = {"schema_version": 2, "measured_at_utc": datetime.now(timezone.utc).isoformat(),
              "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
              "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [
                  args.baseline_root / "app/rag/store.py", args.baseline_root / "app/rag/keyword_index.py",
                  ROOT / "app/rag/store.py", ROOT / "app/rag/keyword_index.py", ROOT / "app/rag/chunk_context.py",
                  ROOT / "app/rag/document_loader.py", ROOT / "app/rag/reranker.py"]},
              "python": platform.python_version(), "platform": platform.platform(),
              "embedding_model": "BAAI/bge-small-zh-v1.5", "embedding_model_load_ms": model_load_ms,
              "threads": 2, "offline_only": True, "qdrant_network": False, "generation": "NOT_RUN",
              "variants": {}, "not_run": {"context_fastembed_rerank": "not requested"}}
    common = dict(postgres_enabled=False, redis_enabled=False, qdrant_enabled=False,
                  rag_candidate_cache_enabled=False, embedding_cache_enabled=False,
                  rag_optimized_retrieval_enabled=True, rag_parallel_workers=1,
                  rag_parent_child_expand_enabled=True, rag_parent_child_sibling_limit=2,
                  rag_fusion_mode="legacy", rerank_enabled=False, rag_context_top_k=5,
                  rag_retrieval_candidate_limit=18, rag_global_rerank_limit=40)
    result["configuration"] = common
    if args.append_raw_rerank:
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        validate_append_results(existing, result)
        existing["raw_rerank_append"] = {"measured_at_utc": result["measured_at_utc"], "embedding_model_load_ms": model_load_ms}
        result = existing

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(render_report(result), encoding="utf-8")

    with configured(**common):
        current = None
        variants = [] if args.append_raw_rerank else [("before_batch", False, args.baseline_root / "app/rag/store.py"),
                                                     ("context", True, ROOT / "app/rag/store.py")]
        for name, enhanced, path in variants:
            module = load_module(path, "context_v2_store_" + name)
            if not enhanced:
                # Load the actual pre-batch index, not today's compatible module
                # imported transitively by the snapshot store.
                index_module = load_module(args.baseline_root / "app/rag/keyword_index.py", "context_v2_baseline_keyword_index")
                module.KeywordIndex, module.VersionedChunkList = index_module.KeywordIndex, index_module.VersionedChunkList
                module.score_candidate = index_module.score_candidate
                module._keyword_index = index_module.KeywordIndex()
                module._chunks = index_module.VersionedChunkList(lock=module._lock)
            records, texts = populate_context(module, fixture, enhanced=enhanced)
            started = perf_counter()
            vectors = embed_documents(embedder, texts)
            build_ms = round((perf_counter() - started) * 1000, 3)
            variant = run_variant(module, fixture, embedder, vectors, records, repeats=args.repeats)
            variant["corpus_embedding_ms"] = build_ms
            variant["configuration"] = {"rag_chunk_context_enabled": enhanced, "rerank_enabled": False}
            result["variants"][name] = variant
            if enhanced:
                current = (module, records, vectors)
            save()
            print(json.dumps({"variant": name, "aggregate": variant["aggregate"]}), flush=True)
        if args.include_reranker or args.append_raw_rerank:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
                started = perf_counter()
                rerank_model = TextCrossEncoder(model_name="BAAI/bge-reranker-base", threads=2,
                                               cache_dir=str(args.cache_dir), local_files_only=True)
                rerank_load_ms = round((perf_counter() - started) * 1000, 3)
                if args.append_raw_rerank:
                    result["raw_rerank_append"]["rerank_model_load_ms"] = rerank_load_ms
                else:
                    result["rerank_model_load_ms"] = rerank_load_ms
                reranker = load_module(ROOT / "app/rag/reranker.py", "context_v2_reranker")
                reranker._fastembed_reranker_model = lambda: rerank_model
                reranker.begin_external_call = lambda **_: None
                reranker.finish_external_call = lambda *_args, **_kwargs: None
                with configured(rerank_enabled=True, rerank_provider="fastembed", rerank_model="BAAI/bge-reranker-base",
                                rerank_fallback_enabled=False, rerank_threads=2, rerank_batch_size=16,
                                rerank_cache_dir=str(args.cache_dir), rerank_local_files_only=True):
                    rerank_configuration = {
                        "rerank_enabled": True, "rerank_provider": "fastembed",
                        "rerank_model": "BAAI/bge-reranker-base", "rerank_fallback_enabled": False,
                        "rerank_threads": 2, "rerank_batch_size": 16, "rerank_local_files_only": True}
                    if current is not None:
                        module, records, vectors = current
                        result["variants"]["context_fastembed_rerank"] = run_variant(
                            module, fixture, embedder, vectors, records, reranker=reranker.rerank_items, repeats=args.repeats)
                        result["variants"]["context_fastembed_rerank"]["configuration"] = {
                            **rerank_configuration, "rag_chunk_context_enabled": True}
                        result["not_run"].pop("context_fastembed_rerank", None)
                        save()
                    raw = load_module(ROOT / "app/rag/store.py", "context_v2_store_raw_rerank")
                    raw_records, raw_texts = populate_context(raw, fixture, enhanced=False)
                    started = perf_counter()
                    raw_vectors = embed_documents(embedder, raw_texts)
                    raw_build_ms = round((perf_counter() - started) * 1000, 3)
                    result["variants"]["raw_fastembed_rerank"] = run_variant(
                        raw, fixture, embedder, raw_vectors, raw_records, reranker=reranker.rerank_items, repeats=args.repeats)
                    result["variants"]["raw_fastembed_rerank"].update(
                        configuration={**rerank_configuration, "rag_chunk_context_enabled": False},
                        corpus_embedding_ms=raw_build_ms)
                    result["not_run"].pop("raw_fastembed_rerank", None)
            except Exception as error:
                missing_variant = "context_fastembed_rerank" if "context_fastembed_rerank" not in result["variants"] else "raw_fastembed_rerank"
                result["not_run"][missing_variant] = type(error).__name__
            save()
            print(json.dumps({"rerank_completed": "context_fastembed_rerank" in result["variants"], "not_run": result["not_run"]}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"kind": "error", "error_type": type(error).__name__}), flush=True)
        sys.exit(1)
