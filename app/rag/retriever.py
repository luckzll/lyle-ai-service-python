import logging
import hashlib
import json
import math
import re
import unicodedata
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from app.core.runtime_config import current_rag_config
from app.core.config import settings
from app.embedding.client import embedding_identity, query_cache_identity
from app.chains.query_rewrite_chain import rewrite_rag_queries
from app.chains.rag_sufficiency_chain import judge_rag_sufficiency
from app.rag.store import search, search_candidates, get_knowledge_base_revisions, knowledge_base_update_in_progress
from app.rag.reranker import rerank_items
from app.rag.concurrency import bounded_map
from app.rag.identity import citation_identity
from app.rag.cache import CacheResult
from app.rag.cache_scope import candidate_cache, candidate_cache_attempt, current_retrieval_cache_scope
from app.chat_progress import rag_phase, rag_query, rag_timing_scope

logger = logging.getLogger(__name__)


def retrieve_context(question: str, knowledge_base_ids: list[str]) -> list[dict[str, Any]]:
    retriever = LyleKnowledgeRetriever(knowledge_base_ids=knowledge_base_ids, top_k=current_rag_config().rag_context_top_k)
    documents = retriever.invoke(question)
    return [
        _context_fields({
            **document.metadata,
            "chunk": document.page_content,
        })
        for document in documents
    ]


def retrieve_context_with_diagnostics(question: str, knowledge_base_ids: list[str]) -> dict[str, object]:
    config = current_rag_config()
    plan: dict[str, object] = {"mode": "legacy", "adaptiveUsed": False}
    with rag_timing_scope() as timings:
        with rag_phase("total"):
            if config.rag_optimized_retrieval_enabled:
                contexts, queries, sufficiency, plan = _retrieve_optimized(question, knowledge_base_ids, config)
            else:
                with rag_phase("query_rewrite"):
                    queries = rewrite_rag_queries(question)
                with rag_phase("recall"):
                    contexts = _retrieve_multi_query_contexts(queries, knowledge_base_ids, config.rag_context_top_k)
                with rag_phase("sufficiency"):
                    sufficiency = judge_rag_sufficiency(question, contexts)
            with rag_phase("diagnostics"):
                attribution = _diagnose_rag_badcase(question, knowledge_base_ids, queries, contexts, sufficiency)
                citation_quality = _score_citation_quality(contexts)
    return {
        "contexts": contexts,
        "queries": queries,
        "sufficiency": sufficiency,
        "attribution": attribution,
        "citationQuality": citation_quality,
        "timings": timings.snapshot(),
        "retrievalPlan": plan,
    }


def _normalise_query(query: str) -> str:
    # Normalize ordinary phrasing, preserving all internal punctuation: JSON,
    # filenames and version/command parameters may differ only in punctuation.
    value = unicodedata.normalize("NFKC", query).casefold().strip()
    value = re.sub(r"[?？!！。]+$", "", value).strip()
    return re.sub(r"\s+", " ", value)


def _unique_queries(queries: list[str], limit: int) -> list[str]:
    unique: dict[str, str] = {}
    for query in queries:
        value = str(query).strip()
        normalised = _normalise_query(value)
        if normalised and normalised not in unique:
            unique[normalised] = value
        if len(unique) >= limit:
            break
    return list(unique.values())


def _adaptive_single_intent(question: str) -> bool:
    """An optional fast path must not infer multi-part coverage from a single matching passage."""
    comparison_or_composition = re.compile(
        r"对比|比较|区别|差异|异同|优缺点|优劣|相同|不同|分别|各自|两者|哪个|以及|同时|然后|接着|另外|此外|和|与|及|并|或|、|[;；，,\n\r]|"
        r"\b(?:compare|comparison|versus|vs\.?|differences?|both|and|or|also|then|pros|cons|better|worse)\b|"
        r"(?:^|\s)[1-9一二三四][.)、]|[（(][一二三四1-9][）)]",
        re.IGNORECASE,
    )
    return not comparison_or_composition.search(question) and len(re.findall(r"[?？]", question)) <= 1


def _authorization_error(error: Exception) -> bool:
    # Authorization failures are not a transient missing query. Never continue
    # with another retrieval source after an explicit access denial.
    current: BaseException | None = error
    for _ in range(4):
        if current is None:
            break
        if isinstance(current, PermissionError):
            return True
        status = getattr(current, "status_code", None) or getattr(getattr(current, "response", None), "status_code", None)
        if status in (401, 403):
            return True
        current = current.__cause__ or current.__context__
    return False


def _candidate_backend_identity() -> str:
    return hashlib.sha256(json.dumps([
        embedding_identity(), settings.qdrant_enabled, settings.qdrant_endpoint,
        settings.qdrant_collection, settings.qdrant_api_key,
        settings.rag_chunk_context_enabled, settings.rag_chunk_context_max_chars,
    ], separators=(",", ":")).encode("utf-8")).hexdigest()


def _search_candidates_cached(question: str, knowledge_base_ids: list[str], limit: int, config: Any) -> CacheResult:
    scope = current_retrieval_cache_scope()
    principal = query_cache_identity()
    if (not config.rag_candidate_cache_enabled or scope is None or not question or principal is None
            or principal[0] != scope.user_id or set(knowledge_base_ids) != set(scope.knowledge_base_ids)):
        return CacheResult(search_candidates(question, knowledge_base_ids, limit), "bypass")
    generation = get_knowledge_base_revisions(knowledge_base_ids)
    backend = _candidate_backend_identity()
    key = (scope.fingerprint(), principal, generation, backend, config, limit,
           hashlib.sha256(question.encode("utf-8")).hexdigest())
    candidate_cache.configure(max_entries=config.rag_candidate_cache_max_entries, max_bytes=config.rag_candidate_cache_max_bytes)

    def valid() -> bool:
        return (not knowledge_base_update_in_progress(knowledge_base_ids)
                and get_knowledge_base_revisions(knowledge_base_ids) == generation
                and _candidate_backend_identity() == backend)

    with candidate_cache_attempt() as attempt:
        allowed = {kb: frozenset(ids) for kb, ids in scope.allowed_document_ids}
        versions = {kb: dict(rows) for kb, rows in scope.document_revisions}

        def cacheable(rows) -> bool:
            return bool(rows) and attempt.healthy and all(
                str(row.get("doc_id", "")) in allowed.get(str(row.get("knowledge_base_id", "")), frozenset())
                and str(row.get("retrieval_revision") or "legacy") == versions.get(str(row.get("knowledge_base_id", "")), {}).get(str(row.get("doc_id", "")))
                for row in rows
            )

        return candidate_cache.get_or_compute(
            key, lambda: search_candidates(question, knowledge_base_ids, limit),
            ttl_seconds=config.rag_candidate_cache_ttl_seconds, cacheable=cacheable, valid=valid,
        )


def _retrieve_optimized(question: str, knowledge_base_ids: list[str], config: Any):
    limit = max(1, min(4, config.rag_multi_query_count))
    top_k = config.rag_context_top_k
    candidate_limit = max(top_k, config.rag_retrieval_candidate_limit)
    cache: dict[str, tuple[list[dict[str, Any]], bool]] = {}
    query_indexes: dict[str, int] = {}
    plan: dict[str, object] = {
        "mode": "optimized", "fusionMode": config.rag_fusion_mode, "adaptiveUsed": False,
        "expansionSkipped": False, "failedQueryCount": 0, "rerankPasses": 0,
        "rerankEnabled": config.rerank_enabled, "candidateCount": 0, "selectedCount": 0,
        "parallelWorkers": config.rag_parallel_workers,
        "candidateCache": {"hit": 0, "miss": 0, "shared": 0, "bypass": 0}, "actualQueryCount": 0,
    }

    def collect(queries: list[str]) -> None:
        missing = [query for query in queries if _normalise_query(query) not in cache]
        for query in missing:
            query_indexes[_normalise_query(query)] = len(query_indexes) + 1

        def run(query: str):
            try:
                with rag_query(query_indexes[_normalise_query(query)]):
                    result = _search_candidates_cached(query, knowledge_base_ids, candidate_limit, config)
                    return (result.value, False), result.status
            except Exception as error:
                if _authorization_error(error):
                    raise
                # Failed expansion must not discard a useful earlier query, or expose provider details.
                logger.warning("RAG candidate query unavailable. error_type=%s", type(error).__name__)
                return ([], True), "bypass"

        with rag_phase("recall"):
            results = bounded_map(run, missing, max_workers=config.rag_parallel_workers)
        for query, (result, cache_status) in zip(missing, results):
            cache[_normalise_query(query)] = result
            plan["candidateCache"][cache_status] += 1
        plan["failedQueryCount"] = sum(int(failed) for _, failed in cache.values())
        plan["actualQueryCount"] = len(cache)

    def rank(queries: list[str], *, allow_initial_failure: bool = False) -> list[dict[str, Any]]:
        with rag_phase("merge"):
            candidates = _fuse_query_candidates(
                [(query, cache[_normalise_query(query)][0]) for query in queries], config.rag_fusion_mode,
            )
        if not allow_initial_failure and not candidates and cache and all(failed for _, failed in cache.values()):
            raise RuntimeError("知识库检索暂时不可用，请稍后重试。")
        plan["candidateCount"] = len(candidates)
        if config.rerank_enabled:
            candidates = candidates[:max(top_k, config.rag_global_rerank_limit)]
        plan["selectedCount"] = len(candidates)
        with rag_phase("rerank"):
            ranked = rerank_items(question, candidates, "chunk", top_k)
        plan["rerankPasses"] = int(plan["rerankPasses"]) + int(config.rerank_enabled and bool(candidates))
        # The returned ordering is authoritative. Never sort it again by the incomparable raw retrieval score.
        return [_context_fields(item) for item in ranked]

    first = _unique_queries([question], 1)
    adaptive = (config.rag_adaptive_retrieval_enabled and config.rag_sufficiency_check_enabled
                and config.rag_query_rewrite_enabled and limit > 1 and _adaptive_single_intent(question))
    initial_contexts = None
    initial_sufficiency = None
    if adaptive:
        plan["adaptiveUsed"] = True
        collect(first)
        initial_contexts = rank(first, allow_initial_failure=True)
        with rag_phase("sufficiency"):
            initial_sufficiency = judge_rag_sufficiency(question, initial_contexts)
        if (initial_contexts and initial_sufficiency.get("sufficient") is True
                and initial_sufficiency.get("method") == "model"):
            plan["expansionSkipped"] = True
            return initial_contexts, first, initial_sufficiency, plan

    with rag_phase("query_rewrite"):
        try:
            rewritten = rewrite_rag_queries(question)
        except Exception as error:
            if _authorization_error(error):
                raise
            logger.warning("RAG query rewrite unavailable; retaining original query. error_type=%s", type(error).__name__)
            rewritten = [question]
        queries = _unique_queries([question, *rewritten], limit)
    collect(queries)
    if initial_contexts is not None and queries == first:
        if not initial_contexts and cache and all(failed for _, failed in cache.values()):
            raise RuntimeError("知识库检索暂时不可用，请稍后重试。")
        return initial_contexts, queries, initial_sufficiency, plan
    contexts = rank(queries)
    if limit == 1 and not contexts:
        from app.rag.query_context import current_query_context

        query_context = current_query_context()
        if (query_context is not None and query_context.effective_question == question
                and query_context.original_question in query_context.additional_queries
                and _normalise_query(query_context.original_question) != _normalise_query(question)):
            # A single-query policy still has a conservative escape hatch when
            # reference resolution found nothing. No extra query runs on a hit
            # or after a failed/forbidden retrieval.
            queries = [*queries, query_context.original_question]
            collect(queries)
            contexts = rank(queries)
    with rag_phase("sufficiency"):
        sufficiency = judge_rag_sufficiency(question, contexts)
    return contexts, queries, sufficiency, plan


def _fuse_query_candidates(query_results: list[tuple[str, list[dict[str, Any]]]], mode: str) -> list[dict[str, Any]]:
    merged: dict[tuple[str, ...], dict[str, Any]] = {}
    for query_index, (query, rows) in enumerate(query_results, start=1):
        seen: set[tuple[str, ...]] = set()
        for rank, source in enumerate(rows, start=1):
            key = citation_identity(source)
            if key in seen:
                continue
            seen.add(key)
            candidate = dict(source)
            candidate["raw_score"] = source.get("raw_score", source.get("score", ""))
            previous = merged.get(key)
            if previous is None:
                candidate["query"] = query
                candidate["query_ranks"] = []
                candidate["query_rrf_score"] = 0.0
                merged[key] = candidate
            else:
                candidate = previous
                if _safe_float(source.get("score")) > _safe_float(previous.get("score")):
                    protected = {field: previous[field] for field in ("query_ranks", "query_rrf_score")}
                    candidate = {**previous, **{field: value for field, value in source.items() if value not in (None, "")}, **protected,
                                 "raw_score": source.get("raw_score", source.get("score", "")), "query": query}
                    merged[key] = candidate
                for field in ("backend", "hitModes"):
                    candidate[field] = ",".join(sorted({value for item in (previous, source) for value in str(item.get(field, "")).split(",") if value}))
                for field in ("vector_rank", "keyword_rank"):
                    values = [item[field] for item in (previous, source) if isinstance(item.get(field), (int, float))]
                    if values:
                        candidate[field] = min(values)
                for field, value in source.items():
                    if candidate.get(field) in (None, "") and value not in (None, ""):
                        candidate[field] = value
            candidate["query_ranks"].append({"queryIndex": query_index, "rank": rank, "retrievalFusionScore": source.get("fusion_score")})
            candidate["query_rrf_score"] += 1.0 / (60 + rank)
    for item in merged.values():
        item["fusion_method"] = mode
        item["fusion_score"] = item["query_rrf_score"] if mode == "rrf" else _safe_float(item.get("score"))
    if mode == "rrf":
        return sorted(merged.values(), key=lambda item: item["fusion_score"], reverse=True)
    return sorted(merged.values(), key=lambda item: _safe_float(item.get("score")), reverse=True)


def _context_fields(item: dict[str, Any]) -> dict[str, Any]:
    result = dict(item)
    for key in ("doc_id", "title", "chunk", "score", "backend", "hitModes", "chunkIndex", "parentTitle", "startChar", "endChar"):
        result[key] = str(item.get(key, ""))
    return result


def _retrieve_multi_query_contexts(
    queries: list[str],
    knowledge_base_ids: list[str],
    top_k: int,
) -> list[dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    for index, query in enumerate(queries, start=1):
        retriever = LyleKnowledgeRetriever(knowledge_base_ids=knowledge_base_ids, top_k=top_k)
        with rag_query(index):
            documents = retriever.invoke(query)
        for document in documents:
            doc_id = str(document.metadata.get("doc_id", ""))
            chunk = document.page_content
            key = f"{doc_id}:{chunk}"
            score = _safe_float(document.metadata.get("score"))
            existing = merged.get(key)
            if existing and _safe_float(existing.get("score")) >= score:
                continue
            merged[key] = {
                "doc_id": doc_id,
                "title": str(document.metadata.get("title", "")),
                "chunk": chunk,
                "score": str(score),
                "backend": str(document.metadata.get("backend", "")),
                "hitModes": str(document.metadata.get("hitModes", "")),
                "chunkIndex": str(document.metadata.get("chunkIndex", "")),
                "parentTitle": str(document.metadata.get("parentTitle", "")),
                "startChar": str(document.metadata.get("startChar", "")),
                "endChar": str(document.metadata.get("endChar", "")),
                "query": query,
            }
    return sorted(
        merged.values(),
        key=lambda item: _safe_float(item.get("score")),
        reverse=True,
    )[:top_k]


def _safe_float(value: object) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _diagnose_rag_badcase(
    question: str,
    knowledge_base_ids: list[str],
    queries: list[str],
    contexts: list[dict[str, str]],
    sufficiency: dict[str, object],
) -> dict[str, object]:
    top_score = max((_safe_float(item.get("score")) for item in contexts), default=0.0)
    reasons: list[str] = []
    suggestions: list[str] = []

    if not knowledge_base_ids:
        reasons.append("未选择知识库")
        suggestions.append("先选择至少一个知识库再提问。")
    if not contexts:
        reasons.append("没有命中文档片段")
        suggestions.append("确认文件已完成入库，或换一个更贴近文档原文的问法。")
    elif top_score < 0.45:
        reasons.append("最高召回分数偏低")
        suggestions.append("尝试补充实体名、文档标题或关键词；必要时重建索引。")
    if contexts and not bool(sufficiency.get("sufficient", True)):
        reasons.append("命中文档不足以回答问题")
        suggestions.append("补充相关资料，或调整文档分块让完整段落能被一起召回。")
    if len(queries) <= 1:
        suggestions.append("可开启或增强 Query Rewrite，让同一问题生成多个检索 query。")

    if not reasons:
        reasons.append("召回结果基本可用")
        suggestions.append("优先检查 Writer 是否正确引用了命中文档。")

    return {
        "status": "ok" if reasons == ["召回结果基本可用"] else "needs_attention",
        "reasons": reasons,
        "suggestions": suggestions,
        "queryCount": len(queries),
        "contextCount": len(contexts),
        "topScore": round(top_score, 4),
        "sufficiencyScore": _safe_float(sufficiency.get("score")),
    }


def _score_citation_quality(contexts: list[dict[str, str]]) -> dict[str, object]:
    if not contexts:
        return {
            "level": "empty",
            "score": 0.0,
            "reason": "没有可引用片段。",
        }
    scores = [_safe_float(item.get("score")) for item in contexts]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    if avg_score >= 0.75:
        level = "high"
        reason = "命中文档平均分较高，引用质量较好。"
    elif avg_score >= 0.45:
        level = "medium"
        reason = "命中文档有一定相关性，建议人工确认关键片段。"
    else:
        level = "low"
        reason = "命中文档相关性偏低，回答容易跑偏。"
    return {
        "level": level,
        "score": round(avg_score, 4),
        "reason": reason,
    }


class LyleKnowledgeRetriever(BaseRetriever):
    knowledge_base_ids: list[str] = Field(default_factory=list)
    top_k: int = 5

    def _get_relevant_documents(self, query: str) -> list[Document]:
        results = search(query, self.knowledge_base_ids, self.top_k)
        return [
            Document(
                page_content=item.get("chunk", ""),
                metadata={
                    **{key: value for key, value in item.items() if key != "chunk"},
                    "doc_id": item.get("doc_id", ""),
                    "title": item.get("title", ""),
                    "score": item.get("score", ""),
                    "backend": item.get("backend", ""),
                    "hitModes": item.get("hitModes", ""),
                    "chunkIndex": item.get("chunkIndex", ""),
                    "parentTitle": item.get("parentTitle", ""),
                    "startChar": item.get("startChar", ""),
                    "endChar": item.get("endChar", ""),
                },
            )
            for item in results
        ]
