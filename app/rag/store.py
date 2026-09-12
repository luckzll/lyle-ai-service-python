from collections import OrderedDict
from dataclasses import dataclass
import logging
import math
import re
from threading import RLock
from time import time
from urllib.parse import urlparse
from uuid import NAMESPACE_URL, uuid5

import httpx

from app.core.config import settings
from app.core.runtime_config import current_rag_config
from app.core.qdrant import qdrant_headers
from app.embedding.client import EMBEDDING_DIMENSION, embed_query, embed_text
from app.rag.document_loader import StorageObjectRef, load_document_text, split_text_with_metadata
from app.rag.reranker import rerank_items
from app.rag.keyword_index import KeywordIndex, VersionedChunkList, score_candidate
from app.rag.identity import citation_identity
from app.rag.chunk_context import LEGACY_CONTEXT_VERSION, build_chunk_context, document_metadata, embedding_text, keyword_text
from app.rag.cache_scope import allowed_document_ids, allowed_document_revisions, current_retrieval_cache_scope, invalidate_candidate_cache, mark_candidate_uncacheable
from app.chat_progress import rag_phase


@dataclass(frozen=True)
class KnowledgeChunk:
    doc_id: str
    knowledge_base_id: str
    title: str
    chunk: str
    chunk_index: int = 1
    parent_title: str = ""
    start_char: int = 0
    end_char: int = 0
    retrieval_text: str = ""
    context_version: str = LEGACY_CONTEXT_VERSION
    document_title: str = ""
    context_sources: tuple[str, ...] = ()
    retrieval_revision: str = "legacy"


_lock = RLock()
_chunks: list[KnowledgeChunk] = VersionedChunkList(lock=_lock)
_keyword_index = KeywordIndex()
_revision_source = None
_revision_source_version = None
_revision_snapshot: dict[str, tuple[KnowledgeChunk, ...]] = {}
_revision_clock = 0
_revision_floor = 0
_kb_revisions: OrderedDict[str, int] = OrderedDict()
_kb_updates: dict[str, int] = {}
# Fixed striped locks bound memory while serializing updates to the same document.
_document_locks = tuple(RLock() for _ in range(64))
_tasks: dict[str, dict[str, str]] = {}
_storage_object_owners: dict[tuple[str, str], str] = {}
_qdrant_disabled_until = 0.0
logger = logging.getLogger(__name__)


def add_document(
    file_id: str,
    knowledge_base_id: str,
    storage_path: str,
    user_id: str | None = None,
    storage_ref: StorageObjectRef | None = None,
    retrieval_revision: str = "legacy",
) -> dict[str, object]:
    task_id = f"ingest-{file_id}"
    title, text = load_document_text(storage_ref or storage_path)
    chunks = split_text_with_metadata(text)
    metadata = document_metadata(text)
    logger.info("Document ingest started. file_id=%s knowledge_base_id=%s chunks=%s", file_id, knowledge_base_id, len(chunks))

    replacements = []
    for index, chunk in enumerate(chunks, start=1):
        context = build_chunk_context(title, chunk.parent_title, chunk.text, metadata,
                                      enabled=settings.rag_chunk_context_enabled,
                                      max_chars=settings.rag_chunk_context_max_chars)
        replacements.append(KnowledgeChunk(
                doc_id=file_id,
                knowledge_base_id=knowledge_base_id,
                title=f"{title} #{index}",
                chunk=chunk.text,
                chunk_index=chunk.index,
                parent_title=chunk.parent_title,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                retrieval_text=context.retrieval_text,
                context_version=context.version,
                document_title=title,
                context_sources=context.sources,
                retrieval_revision=str(retrieval_revision or "legacy"),
            ))
    document_lock = _document_locks[hash((knowledge_base_id, file_id)) % len(_document_locks)]
    with document_lock:
        # Keep ingest synchronous, but do not stop all readers while embedding or
        # contacting Qdrant. The complete local batch is published atomically.
        _begin_knowledge_base_update(knowledge_base_id)
        try:
            # Retries may replace a shorter batch, but must never delete another
            # process's newer revision. Superseded revisions require separate GC.
            _qdrant_delete_document(file_id, knowledge_base_id, str(retrieval_revision or "legacy"))
            vector_count = 0
            for index, item in enumerate(replacements, start=1):
                try:
                    if _qdrant_upsert(item, index):
                        vector_count += 1
                except Exception as exception:
                    _mark_qdrant_failure(exception)
                    _log_qdrant_failure("upsert", exception, knowledge_base_id, file_id)
            with _lock:
                _chunks[:] = [
                    item for item in _chunks
                    if not (item.doc_id == file_id and item.knowledge_base_id == knowledge_base_id)
                ] + replacements
                _ensure_keyword_index_locked()
                _tasks[task_id] = {
                    "task_id": task_id,
                    "file_id": file_id,
                    "knowledge_base_id": knowledge_base_id,
                    "status": "completed",
                    "message": f"Document ingest completed. chunks={len(chunks)}",
                    "context_version": replacements[0].context_version if replacements else LEGACY_CONTEXT_VERSION,
                    "vector_chunks": vector_count,
                    "vector_status": "disabled" if not settings.qdrant_enabled else "completed" if vector_count == len(replacements) else "partial",
                }
                if user_id and storage_ref:
                    _remember_storage_owner(user_id, storage_ref)
        finally:
            _end_knowledge_base_update(knowledge_base_id)

    candidate_memories = _extract_document_candidate_memories(user_id, knowledge_base_id, title, text)
    return {
        "task_id": task_id,
        "candidate_memories": candidate_memories,
        "context_version": replacements[0].context_version if replacements else LEGACY_CONTEXT_VERSION,
        "vector_chunks": vector_count,
    }


def get_task(task_id: str) -> dict[str, str] | None:
    with _lock:
        task = _tasks.get(task_id)
        return dict(task) if task else None


def _bump_knowledge_base_revision_locked(knowledge_base_id: str) -> None:
    global _revision_clock, _revision_floor
    _revision_clock += 1
    _kb_revisions[knowledge_base_id] = _revision_clock
    _kb_revisions.move_to_end(knowledge_base_id)
    invalidate_candidate_cache([knowledge_base_id])
    if len(_kb_revisions) > 4096:
        _kb_revisions.popitem(last=False)
        # Evicted IDs cannot resume an old revision if later requested again.
        _revision_floor = _revision_clock


def _check_corpus_revisions_locked() -> None:
    global _chunks, _revision_source, _revision_source_version, _revision_snapshot
    if not isinstance(_chunks, VersionedChunkList):
        _chunks = VersionedChunkList(_chunks, lock=_lock)
    if _revision_source is _chunks and _revision_source_version == _chunks.version:
        return
    grouped: dict[str, list[KnowledgeChunk]] = {}
    for chunk in _chunks:
        grouped.setdefault(chunk.knowledge_base_id, []).append(chunk)
    snapshot = {kb: tuple(values) for kb, values in grouped.items()}
    for kb in snapshot.keys() | _revision_snapshot.keys():
        if snapshot.get(kb) != _revision_snapshot.get(kb):
            _bump_knowledge_base_revision_locked(kb)
    _revision_source, _revision_source_version = _chunks, _chunks.version
    _revision_snapshot = snapshot


def get_knowledge_base_revisions(knowledge_base_ids: list[str]) -> tuple[tuple[str, int], ...]:
    """Process-local generations; trusted persistent manifests remain required."""
    with _lock:
        _check_corpus_revisions_locked()
        return tuple((kb, max(_kb_revisions.get(kb, 0), _revision_floor)) for kb in sorted(set(knowledge_base_ids)))


def knowledge_base_update_in_progress(knowledge_base_ids: list[str]) -> bool:
    with _lock:
        return any(_kb_updates.get(kb, 0) for kb in knowledge_base_ids)


def invalidate_knowledge_bases(knowledge_base_ids: list[str]) -> None:
    """Hook for external deletion/rebuild protocols, without claiming vector cleanup."""
    with _lock:
        for kb in set(knowledge_base_ids):
            _bump_knowledge_base_revision_locked(kb)


def _begin_knowledge_base_update(kb: str) -> None:
    with _lock:
        _kb_updates[kb] = _kb_updates.get(kb, 0) + 1
        _bump_knowledge_base_revision_locked(kb)


def _end_knowledge_base_update(kb: str) -> None:
    with _lock:
        remaining = _kb_updates.get(kb, 1) - 1
        if remaining:
            _kb_updates[kb] = remaining
        else:
            _kb_updates.pop(kb, None)
        _bump_knowledge_base_revision_locked(kb)


def storage_object_belongs_to_user(user_id: str, storage_path: str) -> bool:
    parsed = urlparse(storage_path)
    if parsed.scheme != "s3":
        return False
    bucket = parsed.netloc
    object_key = parsed.path.lstrip("/")
    if not bucket or not object_key:
        return False
    with _lock:
        return _storage_object_owners.get((bucket, object_key)) == user_id


def remember_storage_object_owner(user_id: str, bucket: str, object_key: str) -> None:
    if not user_id or not bucket or not object_key:
        return
    with _lock:
        _storage_object_owners[(bucket, object_key)] = user_id


def list_chunks(knowledge_base_id: str | None = None, limit: int = 50) -> list[dict[str, str]]:
    with _lock:
        items = [
            item for item in _chunks
            if (knowledge_base_id is None or item.knowledge_base_id == knowledge_base_id)
            and _document_allowed(item.knowledge_base_id, item.doc_id, item.retrieval_revision)
        ][:max(1, min(limit, 200))]
    return [
        {
            "doc_id": item.doc_id,
            "knowledge_base_id": item.knowledge_base_id,
            "title": item.title,
            "chunk": item.chunk,
            "chunkIndex": str(item.chunk_index),
            "parentTitle": item.parent_title,
            "startChar": str(item.start_char),
            "endChar": str(item.end_char),
            "backend": "memory",
            "contextVersion": item.context_version,
            "retrieval_revision": item.retrieval_revision,
        }
        for item in items
    ]


def _remember_storage_owner(user_id: str, ref: StorageObjectRef) -> None:
    provider = (ref.storage_provider or "").upper()
    if provider in {"MINIO", "S3"} and ref.bucket and ref.object_key:
        _storage_object_owners[(ref.bucket, ref.object_key)] = user_id
        return

    parsed = urlparse(ref.storage_path)
    if parsed.scheme == "s3" and parsed.netloc and parsed.path:
        _storage_object_owners[(parsed.netloc, parsed.path.lstrip("/"))] = user_id


def search(question: str, knowledge_base_ids: list[str], top_k: int = 5) -> list[dict[str, str]]:
    if not knowledge_base_ids:
        return []

    candidates = search_candidates(question, knowledge_base_ids, _candidate_limit(top_k))
    with rag_phase("rerank"):
        return rerank_items(question, candidates, "chunk", top_k)


def search_candidates(question: str, knowledge_base_ids: list[str], limit: int = 20) -> list[dict[str, str]]:
    """Return fused/expanded candidates without performing a per-query rerank."""
    if not knowledge_base_ids or limit <= 0:
        return []
    config = current_rag_config()
    candidate_limit = max(1, limit)

    def recall(mode):
        if mode == "vector":
            return _qdrant_search(question, knowledge_base_ids, candidate_limit)
        with rag_phase("keyword_search"):
            return _keyword_search(question, knowledge_base_ids, candidate_limit)

    from app.rag.concurrency import bounded_map
    qdrant_results, keyword_results = bounded_map(
        recall, ["vector", "keyword"],
        max_workers=config.rag_parallel_workers if config.rag_optimized_retrieval_enabled else 1,
    )
    if len(set(knowledge_base_ids)) == 1:
        scope_id = knowledge_base_ids[0]
        qdrant_results = [{**row, "knowledge_base_id": row.get("knowledge_base_id") or scope_id} for row in qdrant_results]
    qdrant_results = [row for row in qdrant_results if _document_allowed(row.get("knowledge_base_id", ""), row.get("doc_id", ""), row.get("retrieval_revision") or "legacy")]
    keyword_results = [row for row in keyword_results if _document_allowed(row.get("knowledge_base_id", ""), row.get("doc_id", ""), row.get("retrieval_revision") or "legacy")]
    with rag_phase("merge"):
        merged = _merge_rag_results(qdrant_results, keyword_results,
                                    mode=config.rag_fusion_mode if config.rag_optimized_retrieval_enabled else "legacy")
    with rag_phase("parent_expansion"):
        expanded = _expand_parent_child_contexts(merged, knowledge_base_ids, candidate_limit)
    logger.info(
        "RAG hybrid searched. knowledge_base_ids=%s vector=%s keyword=%s merged=%s expanded=%s",
        knowledge_base_ids,
        len(qdrant_results),
        len(keyword_results),
        len(merged),
        len(expanded),
    )
    return expanded


def _document_allowed(knowledge_base_id: str, doc_id: str, revision: str = "legacy") -> bool:
    allowed = allowed_document_ids(str(knowledge_base_id))
    if allowed is None:
        return True
    revisions = allowed_document_revisions(str(knowledge_base_id)) or {}
    return str(doc_id) in allowed and revisions.get(str(doc_id)) == revision


def _authorized_vector_filter(knowledge_base_ids: list[str]) -> dict | None:
    if current_retrieval_cache_scope() is None:
        return {"must": [{"key": "knowledge_base_id", "match": {"any": knowledge_base_ids}}]}
    # Pair each document allowlist with its KB; never union IDs across scopes.
    clauses = []
    for kb in knowledge_base_ids:
        documents = allowed_document_ids(kb)
        revisions = allowed_document_revisions(kb) or {}
        groups: dict[str, list[str]] = {}
        for doc in documents or ():
            if doc in revisions:
                groups.setdefault(revisions[doc], []).append(doc)
        for revision, doc_ids in groups.items():
            version_filter = {"key": "retrieval_revision", "match": {"value": revision}}
            if revision == "legacy":
                version_filter = {"should": [version_filter, {"is_empty": {"key": "retrieval_revision"}}]}
            clauses.append({"must": [
                {"key": "knowledge_base_id", "match": {"value": kb}},
                {"key": "doc_id", "match": {"any": sorted(doc_ids)}},
                version_filter,
            ]})
    return {"should": clauses} if clauses else None


def _candidate_limit(top_k: int) -> int:
    config = current_rag_config()
    if not config.rerank_enabled:
        return top_k
    return max(top_k, config.rerank_candidate_limit)


def _ensure_keyword_index_locked() -> KeywordIndex:
    global _chunks
    # Compatibility for restoration/tests replacing the whole corpus with a list.
    # Once wrapped, ordinary mutations invalidate in O(1), not by hashing all rows
    # for every query. Frozen KnowledgeChunk values cannot mutate underneath it.
    if not isinstance(_chunks, VersionedChunkList):
        _chunks = VersionedChunkList(_chunks, lock=_lock)
    _check_corpus_revisions_locked()
    _keyword_index.synchronize(_chunks, _keyword_terms)
    return _keyword_index


def rebuild_keyword_index() -> None:
    """Rebuild derived postings from the current corpus; never change documents."""
    global _keyword_index
    with _lock:
        _keyword_index = KeywordIndex()
        _ensure_keyword_index_locked()


def _keyword_search(question: str, knowledge_base_ids: list[str], top_k: int) -> list[dict[str, str]]:
    query_terms = _keyword_terms(question)
    allowed = {kb: allowed_document_ids(kb) for kb in knowledge_base_ids}
    revisions = {kb: allowed_document_revisions(kb) for kb in knowledge_base_ids}
    with _lock:
        candidates = _ensure_keyword_index_locked().candidates(query_terms, knowledge_base_ids, top_k, allowed, revisions)
    scored = [(score_candidate(query_terms, candidate), candidate.chunk) for candidate in candidates]
    scored = [(score, item) for score, item in scored if score > 0]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "doc_id": item.doc_id,
            "knowledge_base_id": item.knowledge_base_id,
            "title": item.title,
            "chunk": item.chunk,
            "score": str(round(score, 4)),
            "backend": "keyword",
            "hitModes": "keyword",
            "chunkIndex": str(item.chunk_index),
            "parentTitle": item.parent_title,
            "startChar": str(item.start_char),
            "endChar": str(item.end_char),
            "contextVersion": item.context_version,
            "documentTitle": item.document_title,
            "retrieval_text": item.retrieval_text,
            "retrieval_revision": item.retrieval_revision,
        }
        for score, item in scored[:top_k]
    ]


def _merge_rag_results(
    vector_results: list[dict[str, str]],
    keyword_results: list[dict[str, str]],
    *, mode: str | None = None,
) -> list[dict[str, str]]:
    mode = mode or current_rag_config().rag_fusion_mode
    if mode == "rrf":
        return _rrf_results(vector_results, keyword_results)
    merged: dict[str, dict[str, str]] = {}
    for item in [*vector_results, *keyword_results]:
        chunk = item.get("chunk", "")
        doc_id = item.get("doc_id", "")
        if not chunk:
            continue
        key = _chunk_key(item)
        backend = item.get("backend") or "unknown"
        hit_modes = set(filter(None, (item.get("hitModes") or backend).split(",")))
        score = _safe_float(item.get("score"))
        existing = merged.get(key)
        if existing is None:
            merged[key] = {
                **item,
                "score": str(score),
                "backend": backend,
                "hitModes": ",".join(sorted(hit_modes)),
            }
            continue
        existing_backends = set(filter(None, (existing.get("backend") or "").split(",")))
        existing_backends.add(backend)
        existing_modes = set(filter(None, (existing.get("hitModes") or existing.get("backend") or "").split(",")))
        existing_modes.update(hit_modes)
        if score > _safe_float(existing.get("score")):
            existing.update({**item, "score": str(score)})
        existing["backend"] = ",".join(sorted(existing_backends))
        existing["hitModes"] = ",".join(sorted(existing_modes))
    return sorted(merged.values(), key=lambda item: _safe_float(item.get("score")), reverse=True)


def _chunk_key(item):
    return citation_identity(item)


def _rrf_results(vector_results, keyword_results):
    # Original backend scores are retained for diagnostics, never relabelled as
    # relevance probabilities. Fusion uses ranks rather than incompatible scales.
    merged = {_chunk_key(item): item for item in _merge_rag_results(vector_results, keyword_results, mode="legacy")}
    first_seen = {}
    for channel, rows in (("vector", vector_results), ("keyword", keyword_results)):
        seen = set()
        for rank, row in enumerate(rows, start=1):
            key = _chunk_key(row)
            if key not in merged or key in seen:
                continue
            seen.add(key)
            first_seen.setdefault(key, len(first_seen))
            item = merged[key]
            item[f"{channel}_rank"] = rank
            item["fusion_score"] = float(item.get("fusion_score") or 0.0) + 1.0 / (60 + rank)
    return sorted(merged.values(), key=lambda item: (-float(item.get("fusion_score") or 0), first_seen[_chunk_key(item)]))


def _merge_expanded_results(results, additions):
    merged = _merge_rag_results(results, additions, mode="legacy")
    if any("fusion_score" in item for item in results):
        # Do not feed an already fused ranking back into RRF or sort it by raw
        # vector/keyword score during parent expansion.
        order = {_chunk_key(item): index for index, item in enumerate([*results, *additions])}
        merged.sort(key=lambda item: (-float(item.get("fusion_score") or 0), order[_chunk_key(item)]))
    return merged


def _expand_parent_child_contexts(
    results: list[dict[str, str]],
    knowledge_base_ids: list[str],
    limit: int,
) -> list[dict[str, str]]:
    results = [row for row in results if _document_allowed(row.get("knowledge_base_id", ""), row.get("doc_id", ""), row.get("retrieval_revision") or "legacy")]
    config = current_rag_config()
    if not config.rag_parent_child_expand_enabled or not results:
        return results
    target_ids = set(knowledge_base_ids)
    existing_keys = {_chunk_key(item) for item in results}
    additions: list[dict[str, str]] = []
    existing_by_key = {_chunk_key(item): item for item in results}
    sibling_limit = max(0, config.rag_parent_child_sibling_limit)
    if sibling_limit <= 0:
        return results
    with _lock:
        chunks = [
            item for item in _chunks
            if item.knowledge_base_id in target_ids
            and _document_allowed(item.knowledge_base_id, item.doc_id, item.retrieval_revision)
        ]
    by_doc = {(item.knowledge_base_id, item.doc_id, item.chunk_index): item for item in chunks}
    for result in results[:limit]:
        doc_id = result.get("doc_id", "")
        kb_id = result.get("knowledge_base_id", "")
        if not kb_id:
            matches = {item.knowledge_base_id for item in chunks if item.doc_id == doc_id}
            if len(matches) != 1:
                continue
            kb_id = next(iter(matches))
        parent_title = result.get("parentTitle", "")
        try:
            chunk_index = int(result.get("chunkIndex") or 0)
        except ValueError:
            chunk_index = 0
        if chunk_index <= 0:
            continue
        candidates = _parent_child_candidates(chunks, by_doc, doc_id, parent_title, chunk_index, sibling_limit, kb_id)
        for candidate in candidates:
            candidate_result = {
                "doc_id": candidate.doc_id,
                "knowledge_base_id": candidate.knowledge_base_id,
                "title": candidate.title,
                "chunk": candidate.chunk,
                "score": str(round(max(_safe_float(result.get("score")) * 0.82, 0.01), 4)),
                "backend": "parent_child",
                "hitModes": "parent_child",
                "chunkIndex": str(candidate.chunk_index),
                "parentTitle": candidate.parent_title,
                "startChar": str(candidate.start_char),
                "endChar": str(candidate.end_char),
                "contextVersion": candidate.context_version,
                "documentTitle": candidate.document_title,
                "retrieval_text": candidate.retrieval_text,
                "retrieval_revision": candidate.retrieval_revision,
                **({"fusion_score": float(result["fusion_score"]) * 0.82} if "fusion_score" in result else {}),
            }
            key = _chunk_key(candidate_result)
            if key in existing_keys:
                existing = existing_by_key.get(key)
                if existing is not None:
                    existing["backend"] = _join_modes(existing.get("backend", ""), "parent_child")
                    existing["hitModes"] = _join_modes(existing.get("hitModes", ""), "parent_child")
                continue
            existing_keys.add(key)
            additions.append(candidate_result)
            if len(additions) >= limit:
                return _merge_expanded_results(results, additions)
    return _merge_expanded_results(results, additions)


def _join_modes(existing: str, mode: str) -> str:
    modes = set(filter(None, existing.split(",")))
    modes.add(mode)
    return ",".join(sorted(modes))


def _parent_child_candidates(
    chunks: list[KnowledgeChunk],
    by_doc: dict[tuple[str, str, int], KnowledgeChunk],
    doc_id: str,
    parent_title: str,
    chunk_index: int,
    sibling_limit: int,
    knowledge_base_id: str,
) -> list[KnowledgeChunk]:
    candidates: list[KnowledgeChunk] = []
    if parent_title:
        candidates.extend([
            item for item in chunks
            if item.doc_id == doc_id
            and item.knowledge_base_id == knowledge_base_id
            and item.parent_title == parent_title
            and item.chunk_index != chunk_index
        ][:sibling_limit])
    for offset in range(1, sibling_limit + 1):
        for index in (chunk_index - offset, chunk_index + offset):
            item = by_doc.get((knowledge_base_id, doc_id, index))
            if item and item not in candidates:
                candidates.append(item)
    return candidates[:sibling_limit]


def _keyword_terms(text: str) -> list[str]:
    lowered = text.lower()
    latin_terms = re.findall(r"[a-z0-9][a-z0-9._-]{1,}", lowered)
    latin_sub_terms = [
        token
        for term in latin_terms
        for token in re.split(r"[._-]+", term)
        if len(token) >= 2
    ]
    chinese_terms = [
        token
        for token in re.split(r"[\s，。！？、,.!?；;：:（）()\[\]【】\"'“”]+", text)
        if len(token) >= 2 and not re.fullmatch(r"[a-zA-Z0-9._-]+", token)
    ]
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    terms = [*latin_terms, *latin_sub_terms, *chinese_terms, *chinese_chars]
    seen: set[str] = set()
    result: list[str] = []
    for term in terms:
        cleaned = term.strip().lower()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _keyword_score(query_terms: list[str], item: KnowledgeChunk) -> float:
    if not query_terms:
        return 0.0
    target = keyword_text(item).lower()
    doc_terms = _keyword_terms(target)
    doc_len = max(len(doc_terms), 1)
    avg_doc_len = _average_keyword_doc_len(item.knowledge_base_id)
    total_docs = _knowledge_base_doc_count(item.knowledge_base_id)
    k1 = 1.5
    b = 0.75
    score = 0.0
    matched = 0
    for term in query_terms:
        tf = doc_terms.count(term)
        if tf <= 0:
            continue
        matched += 1
        containing_docs = _document_frequency(term, item.knowledge_base_id)
        idf = math.log(1 + (total_docs - containing_docs + 0.5) / (containing_docs + 0.5))
        length_norm = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
        score += idf * (tf * (k1 + 1) / max(length_norm, 0.0001))
    if matched == 0:
        return 0.0
    coverage = matched / len(query_terms)
    normalized = score / (score + 4.0) if score > 0 else 0.0
    return min(1.0, normalized * 0.72 + coverage * 0.28)


def _knowledge_base_doc_count(knowledge_base_id: str) -> int:
    with _lock:
        return max(1, sum(1 for item in _chunks if item.knowledge_base_id == knowledge_base_id))


def _average_keyword_doc_len(knowledge_base_id: str) -> float:
    with _lock:
        lengths = [
            len(_keyword_terms(keyword_text(item)))
            for item in _chunks
            if item.knowledge_base_id == knowledge_base_id
        ]
    return max(sum(lengths) / len(lengths), 1.0) if lengths else 1.0


def _document_frequency(term: str, knowledge_base_id: str) -> int:
    with _lock:
        return max(
            0,
            sum(
                1
                for item in _chunks
                if item.knowledge_base_id == knowledge_base_id
                and term in _keyword_terms(keyword_text(item))
            ),
        )


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _extract_document_candidate_memories(
    user_id: str | None,
    knowledge_base_id: str,
    title: str,
    text: str,
) -> list[dict[str, object]]:
    if not user_id:
        return []
    excerpt = _document_memory_excerpt(title, text)
    if not excerpt:
        return []
    from app.memory.store import extract_structured_memory

    return extract_structured_memory(user_id, f"knowledge:{knowledge_base_id}", excerpt)


def _document_memory_excerpt(title: str, text: str) -> str:
    cleaned_lines = [line.strip() for line in text.splitlines() if line.strip()]
    excerpt = "\n".join(cleaned_lines[:24])
    if len(excerpt) > 1600:
        excerpt = excerpt[:1600]
    return f"文件 {title} 中的信息：\n{excerpt}" if excerpt else ""


def _qdrant_delete_document(doc_id: str, knowledge_base_id: str, retrieval_revision: str = "legacy") -> bool:
    if not settings.qdrant_enabled:
        return False
    if not _qdrant_available():
        raise RuntimeError("Knowledge vector backend is temporarily unavailable for document replacement")
    _ensure_qdrant_collection()
    revision_filter = {"key": "retrieval_revision", "match": {"value": retrieval_revision}}
    if retrieval_revision == "legacy":
        revision_filter = {"should": [revision_filter, {"is_empty": {"key": "retrieval_revision"}}]}
    with httpx.Client(timeout=settings.vector_backend_timeout_seconds, headers=qdrant_headers()) as client:
        client.post(
            f"{settings.qdrant_endpoint.rstrip('/')}/collections/{settings.qdrant_collection}/points/delete",
            params={"wait": "true"},
            json={"filter": {"must": [
                {"key": "knowledge_base_id", "match": {"value": knowledge_base_id}},
                {"key": "doc_id", "match": {"value": doc_id}},
                revision_filter,
            ]}},
        ).raise_for_status()
    return True


def _qdrant_upsert(item: KnowledgeChunk, chunk_index: int) -> bool:
    if not settings.qdrant_enabled or not _qdrant_available():
        return False

    _ensure_qdrant_collection()
    point_id = str(uuid5(NAMESPACE_URL, f"{item.knowledge_base_id}:{item.doc_id}:{item.retrieval_revision}:{chunk_index}"))
    body = {
        "points": [
            {
                "id": point_id,
                "vector": embed_text(embedding_text(item)),
                "payload": {
                    "doc_id": item.doc_id,
                    "knowledge_base_id": item.knowledge_base_id,
                    "title": item.title,
                    "chunk": item.chunk,
                    "chunk_index": item.chunk_index,
                    "parent_title": item.parent_title,
                    "start_char": item.start_char,
                    "end_char": item.end_char,
                    "retrieval_text": item.retrieval_text,
                    "context_version": item.context_version,
                    "document_title": item.document_title,
                    "context_sources": list(item.context_sources),
                    "embedding_context_version": item.context_version,
                    "embedding_provider": settings.embedding_provider,
                    "embedding_model": settings.embedding_model,
                    "embedding_dimension": EMBEDDING_DIMENSION,
                    "retrieval_revision": item.retrieval_revision,
                },
            }
        ]
    }
    with httpx.Client(timeout=settings.vector_backend_timeout_seconds, headers=qdrant_headers()) as client:
        client.put(
            f"{settings.qdrant_endpoint.rstrip('/')}/collections/{settings.qdrant_collection}/points",
            params={"wait": "true"},
            json=body,
        ).raise_for_status()
    logger.info("Qdrant knowledge chunk upserted. doc_id=%s knowledge_base_id=%s", item.doc_id, item.knowledge_base_id)
    return True


def _qdrant_search(question: str, knowledge_base_ids: list[str], top_k: int) -> list[dict[str, str]]:
    authorized_filter = _authorized_vector_filter(knowledge_base_ids)
    if authorized_filter is None:
        return []
    if not settings.qdrant_enabled:
        return []
    if not _qdrant_available():
        mark_candidate_uncacheable()
        return []

    try:
        with rag_phase("qdrant_prepare"):
            _ensure_qdrant_collection()
    except Exception as exception:
        mark_candidate_uncacheable()
        if _authorization_failure(exception):
            raise
        _mark_qdrant_failure(exception)
        _log_qdrant_failure("search_prepare", exception, ",".join(knowledge_base_ids))
        return []
    try:
        with rag_phase("embedding"):
            query_vector = embed_query(question)
    except Exception as exception:
        mark_candidate_uncacheable()
        if _authorization_failure(exception):
            raise
        _mark_qdrant_failure(exception)
        _log_qdrant_failure("query_embedding", exception, ",".join(knowledge_base_ids))
        return []
    body = {
        "vector": query_vector,
        "limit": top_k,
        "with_payload": True,
        "filter": authorized_filter,
    }
    try:
        with rag_phase("vector_search"):
            with httpx.Client(timeout=settings.vector_backend_timeout_seconds, headers=qdrant_headers()) as client:
                response = client.post(
                    f"{settings.qdrant_endpoint.rstrip('/')}/collections/{settings.qdrant_collection}/points/search",
                    json=body,
                )
                response.raise_for_status()
                points = response.json().get("result") or []
    except Exception as exception:
        mark_candidate_uncacheable()
        if _authorization_failure(exception):
            raise
        _mark_qdrant_failure(exception)
        _log_qdrant_failure("search", exception, ",".join(knowledge_base_ids))
        return []

    results = []
    for point in points:
        payload = point.get("payload") or {}
        if not _document_allowed(payload.get("knowledge_base_id", ""), payload.get("doc_id", ""), payload.get("retrieval_revision") or "legacy"):
            continue
        results.append({
            "doc_id": payload.get("doc_id", ""),
            "knowledge_base_id": payload.get("knowledge_base_id", ""),
            "title": payload.get("title", ""),
            "chunk": payload.get("chunk", ""),
            "score": str(point.get("score", 0)),
            "backend": "qdrant",
            "hitModes": "vector",
            "chunkIndex": str(payload.get("chunk_index", "")),
            "parentTitle": payload.get("parent_title", ""),
            "startChar": str(payload.get("start_char", "")),
            "endChar": str(payload.get("end_char", "")),
            "contextVersion": payload.get("context_version") or LEGACY_CONTEXT_VERSION,
            "embeddingContextVersion": payload.get("embedding_context_version") or LEGACY_CONTEXT_VERSION,
            "documentTitle": payload.get("document_title", ""),
            "retrieval_text": payload.get("retrieval_text", ""),
            "retrieval_revision": payload.get("retrieval_revision") or "legacy",
        })
    return results


def _ensure_qdrant_collection() -> None:
    with httpx.Client(timeout=settings.vector_backend_timeout_seconds, headers=qdrant_headers()) as client:
        response = client.get(
            f"{settings.qdrant_endpoint.rstrip('/')}/collections/{settings.qdrant_collection}"
        )
        if response.status_code == 200:
            data = response.json().get("result") or {}
            existing_size = _qdrant_vector_size(data)
            if existing_size == EMBEDDING_DIMENSION:
                return
            if not settings.recreate_vector_collections_on_dimension_mismatch:
                raise ValueError(
                    f"Qdrant collection {settings.qdrant_collection} has vector size "
                    f"{existing_size}, expected {EMBEDDING_DIMENSION}."
                )
            client.delete(
                f"{settings.qdrant_endpoint.rstrip('/')}/collections/{settings.qdrant_collection}"
            ).raise_for_status()
        if response.status_code != 404:
            response.raise_for_status()
        client.put(
            f"{settings.qdrant_endpoint.rstrip('/')}/collections/{settings.qdrant_collection}",
            json={"vectors": {"size": EMBEDDING_DIMENSION, "distance": "Cosine"}},
        ).raise_for_status()


def _qdrant_vector_size(collection_info: dict) -> int | None:
    vectors = (collection_info.get("config") or {}).get("params", {}).get("vectors")
    if isinstance(vectors, dict) and "size" in vectors:
        return int(vectors["size"])
    if isinstance(vectors, dict):
        for value in vectors.values():
            if isinstance(value, dict) and "size" in value:
                return int(value["size"])
    return None


def _qdrant_available() -> bool:
    return _qdrant_disabled_until <= time()


def _authorization_failure(error: BaseException) -> bool:
    current = error
    for _ in range(4):
        if current is None:
            break
        status = getattr(current, "status_code", None) or getattr(getattr(current, "response", None), "status_code", None)
        if isinstance(current, PermissionError) or status in (401, 403):
            return True
        current = current.__cause__ or current.__context__
    return False


def _mark_qdrant_failure(exception: Exception) -> None:
    global _qdrant_disabled_until
    cooldown = max(settings.vector_backend_failure_cooldown_seconds, 0)
    if cooldown <= 0:
        return
    was_available = _qdrant_available()
    _qdrant_disabled_until = time() + cooldown
    if was_available:
        logger.warning(
            "Qdrant knowledge backend temporarily disabled. cooldown_seconds=%s error_type=%s",
            cooldown,
            type(exception).__name__,
        )


def _log_qdrant_failure(action: str, exception: Exception, knowledge_base_id: str = "", doc_id: str = "") -> None:
    logger.warning(
        "Qdrant knowledge %s failed. knowledge_base_id=%s doc_id=%s error_type=%s",
        action,
        knowledge_base_id,
        doc_id,
        type(exception).__name__,
    )
