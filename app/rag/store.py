from dataclasses import dataclass
import logging
import math
import re
from threading import Lock
from time import time
from urllib.parse import urlparse
from uuid import NAMESPACE_URL, uuid5

import httpx

from app.core.config import settings
from app.core.qdrant import qdrant_headers
from app.embedding.client import EMBEDDING_DIMENSION, embed_text
from app.rag.document_loader import StorageObjectRef, load_document_text, split_text_with_metadata
from app.rag.reranker import rerank_items


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


_lock = Lock()
_chunks: list[KnowledgeChunk] = []
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
) -> dict[str, object]:
    task_id = f"ingest-{file_id}"
    title, text = load_document_text(storage_ref or storage_path)
    chunks = split_text_with_metadata(text)
    logger.info("Document ingest started. file_id=%s knowledge_base_id=%s chunks=%s", file_id, knowledge_base_id, len(chunks))

    with _lock:
        _chunks[:] = [
            item for item in _chunks
            if not (item.doc_id == file_id and item.knowledge_base_id == knowledge_base_id)
        ]
        for index, chunk in enumerate(chunks, start=1):
            item = KnowledgeChunk(
                doc_id=file_id,
                knowledge_base_id=knowledge_base_id,
                title=f"{title} #{index}",
                chunk=chunk.text,
                chunk_index=chunk.index,
                parent_title=chunk.parent_title,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
            )
            _chunks.append(item)
            try:
                _qdrant_upsert(item, index)
            except Exception as exception:
                _mark_qdrant_failure(exception)
                _log_qdrant_failure("upsert", exception, knowledge_base_id, file_id)
        _tasks[task_id] = {
            "task_id": task_id,
            "file_id": file_id,
            "knowledge_base_id": knowledge_base_id,
            "status": "completed",
            "message": f"Document ingest completed. chunks={len(chunks)}",
        }
        if user_id and storage_ref:
            _remember_storage_owner(user_id, storage_ref)

    candidate_memories = _extract_document_candidate_memories(user_id, knowledge_base_id, title, text)
    return {
        "task_id": task_id,
        "candidate_memories": candidate_memories,
    }


def get_task(task_id: str) -> dict[str, str] | None:
    with _lock:
        task = _tasks.get(task_id)
        return dict(task) if task else None


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
            if knowledge_base_id is None or item.knowledge_base_id == knowledge_base_id
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

    candidate_limit = _candidate_limit(top_k)
    qdrant_results = _qdrant_search(question, knowledge_base_ids, candidate_limit)
    keyword_results = _keyword_search(question, knowledge_base_ids, candidate_limit)
    merged = _merge_rag_results(qdrant_results, keyword_results)
    expanded = _expand_parent_child_contexts(merged, knowledge_base_ids, candidate_limit)
    logger.info(
        "RAG hybrid searched. knowledge_base_ids=%s vector=%s keyword=%s merged=%s expanded=%s",
        knowledge_base_ids,
        len(qdrant_results),
        len(keyword_results),
        len(merged),
        len(expanded),
    )
    return rerank_items(question, expanded, "chunk", top_k)


def _candidate_limit(top_k: int) -> int:
    if not settings.rerank_enabled:
        return top_k
    return max(top_k, settings.rerank_candidate_limit)


def _keyword_search(question: str, knowledge_base_ids: list[str], top_k: int) -> list[dict[str, str]]:
    query_terms = _keyword_terms(question)
    target_ids = set(knowledge_base_ids)
    with _lock:
        matched = [
            item for item in _chunks
            if item.knowledge_base_id in target_ids
        ]

    scored: list[tuple[float, KnowledgeChunk]] = []
    for item in matched:
        score = _keyword_score(query_terms, item)
        if score <= 0:
            continue
        scored.append((score, item))
    if not scored and not query_terms:
        scored = [(0.01, item) for item in matched[:top_k]]

    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "doc_id": item.doc_id,
            "title": item.title,
            "chunk": item.chunk,
            "score": str(round(score, 4)),
            "backend": "keyword",
            "hitModes": "keyword",
            "chunkIndex": str(item.chunk_index),
            "parentTitle": item.parent_title,
            "startChar": str(item.start_char),
            "endChar": str(item.end_char),
        }
        for score, item in scored[:top_k]
    ]


def _merge_rag_results(
    vector_results: list[dict[str, str]],
    keyword_results: list[dict[str, str]],
) -> list[dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    for item in [*vector_results, *keyword_results]:
        chunk = item.get("chunk", "")
        doc_id = item.get("doc_id", "")
        if not chunk:
            continue
        key = f"{doc_id}:{chunk}"
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


def _expand_parent_child_contexts(
    results: list[dict[str, str]],
    knowledge_base_ids: list[str],
    limit: int,
) -> list[dict[str, str]]:
    if not settings.rag_parent_child_expand_enabled or not results:
        return results
    target_ids = set(knowledge_base_ids)
    existing_keys = {f"{item.get('doc_id', '')}:{item.get('chunk', '')}" for item in results}
    additions: list[dict[str, str]] = []
    existing_by_key = {f"{item.get('doc_id', '')}:{item.get('chunk', '')}": item for item in results}
    sibling_limit = max(0, settings.rag_parent_child_sibling_limit)
    if sibling_limit <= 0:
        return results
    with _lock:
        chunks = [
            item for item in _chunks
            if item.knowledge_base_id in target_ids
        ]
    by_doc = {(item.doc_id, item.chunk_index): item for item in chunks}
    for result in results[:limit]:
        doc_id = result.get("doc_id", "")
        parent_title = result.get("parentTitle", "")
        try:
            chunk_index = int(result.get("chunkIndex") or 0)
        except ValueError:
            chunk_index = 0
        if chunk_index <= 0:
            continue
        candidates = _parent_child_candidates(chunks, by_doc, doc_id, parent_title, chunk_index, sibling_limit)
        for candidate in candidates:
            key = f"{candidate.doc_id}:{candidate.chunk}"
            if key in existing_keys:
                existing = existing_by_key.get(key)
                if existing is not None:
                    existing["backend"] = _join_modes(existing.get("backend", ""), "parent_child")
                    existing["hitModes"] = _join_modes(existing.get("hitModes", ""), "parent_child")
                continue
            existing_keys.add(key)
            additions.append({
                "doc_id": candidate.doc_id,
                "title": candidate.title,
                "chunk": candidate.chunk,
                "score": str(round(max(_safe_float(result.get("score")) * 0.82, 0.01), 4)),
                "backend": "parent_child",
                "hitModes": "parent_child",
                "chunkIndex": str(candidate.chunk_index),
                "parentTitle": candidate.parent_title,
                "startChar": str(candidate.start_char),
                "endChar": str(candidate.end_char),
            })
            if len(additions) >= limit:
                return _merge_rag_results(results, additions)
    return _merge_rag_results(results, additions)


def _join_modes(existing: str, mode: str) -> str:
    modes = set(filter(None, existing.split(",")))
    modes.add(mode)
    return ",".join(sorted(modes))


def _parent_child_candidates(
    chunks: list[KnowledgeChunk],
    by_doc: dict[tuple[str, int], KnowledgeChunk],
    doc_id: str,
    parent_title: str,
    chunk_index: int,
    sibling_limit: int,
) -> list[KnowledgeChunk]:
    candidates: list[KnowledgeChunk] = []
    if parent_title:
        candidates.extend([
            item for item in chunks
            if item.doc_id == doc_id
            and item.parent_title == parent_title
            and item.chunk_index != chunk_index
        ][:sibling_limit])
    for offset in range(1, sibling_limit + 1):
        for index in (chunk_index - offset, chunk_index + offset):
            item = by_doc.get((doc_id, index))
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
    target = f"{item.title}\n{item.parent_title}\n{item.chunk}".lower()
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
            len(_keyword_terms(f"{item.title}\n{item.parent_title}\n{item.chunk}"))
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
                and term in _keyword_terms(f"{item.title}\n{item.parent_title}\n{item.chunk}")
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


def _qdrant_upsert(item: KnowledgeChunk, chunk_index: int) -> None:
    if not settings.qdrant_enabled or not _qdrant_available():
        return

    _ensure_qdrant_collection()
    point_id = str(uuid5(NAMESPACE_URL, f"{item.doc_id}:{chunk_index}"))
    body = {
        "points": [
            {
                "id": point_id,
                "vector": embed_text(item.chunk),
                "payload": {
                    "doc_id": item.doc_id,
                    "knowledge_base_id": item.knowledge_base_id,
                    "title": item.title,
                    "chunk": item.chunk,
                    "chunk_index": item.chunk_index,
                    "parent_title": item.parent_title,
                    "start_char": item.start_char,
                    "end_char": item.end_char,
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


def _qdrant_search(question: str, knowledge_base_ids: list[str], top_k: int) -> list[dict[str, str]]:
    if not settings.qdrant_enabled or not _qdrant_available():
        return []

    try:
        _ensure_qdrant_collection()
    except Exception as exception:
        _mark_qdrant_failure(exception)
        _log_qdrant_failure("search_prepare", exception, ",".join(knowledge_base_ids))
        return []
    body = {
        "vector": embed_text(question),
        "limit": top_k,
        "with_payload": True,
        "filter": {
            "must": [
                {
                    "key": "knowledge_base_id",
                    "match": {"any": knowledge_base_ids},
                }
            ]
        },
    }
    try:
        with httpx.Client(timeout=settings.vector_backend_timeout_seconds, headers=qdrant_headers()) as client:
            response = client.post(
                f"{settings.qdrant_endpoint.rstrip('/')}/collections/{settings.qdrant_collection}/points/search",
                json=body,
            )
            response.raise_for_status()
            points = response.json().get("result") or []
    except Exception as exception:
        _mark_qdrant_failure(exception)
        _log_qdrant_failure("search", exception, ",".join(knowledge_base_ids))
        return []

    results = []
    for point in points:
        payload = point.get("payload") or {}
        results.append({
            "doc_id": payload.get("doc_id", ""),
            "title": payload.get("title", ""),
            "chunk": payload.get("chunk", ""),
            "score": str(point.get("score", 0)),
            "backend": "qdrant",
            "hitModes": "vector",
            "chunkIndex": str(payload.get("chunk_index", "")),
            "parentTitle": payload.get("parent_title", ""),
            "startChar": str(payload.get("start_char", "")),
            "endChar": str(payload.get("end_char", "")),
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


def _mark_qdrant_failure(exception: Exception) -> None:
    global _qdrant_disabled_until
    cooldown = max(settings.vector_backend_failure_cooldown_seconds, 0)
    if cooldown <= 0:
        return
    was_available = _qdrant_available()
    _qdrant_disabled_until = time() + cooldown
    if was_available:
        logger.warning(
            "Qdrant knowledge backend temporarily disabled. cooldown_seconds=%s error=%s",
            cooldown,
            exception,
        )


def _log_qdrant_failure(action: str, exception: Exception, knowledge_base_id: str = "", doc_id: str = "") -> None:
    logger.warning(
        "Qdrant knowledge %s failed. knowledge_base_id=%s doc_id=%s error=%s",
        action,
        knowledge_base_id,
        doc_id,
        exception,
    )
