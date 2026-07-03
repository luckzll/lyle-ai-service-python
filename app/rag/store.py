from dataclasses import dataclass
import logging
from threading import Lock
from time import time
from urllib.parse import urlparse
from uuid import NAMESPACE_URL, uuid5

import httpx

from app.core.config import settings
from app.embedding.client import EMBEDDING_DIMENSION, embed_text
from app.rag.document_loader import StorageObjectRef, load_document_text, split_text
from app.rag.reranker import rerank_items


@dataclass(frozen=True)
class KnowledgeChunk:
    doc_id: str
    knowledge_base_id: str
    title: str
    chunk: str


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
    chunks = split_text(text)
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
                chunk=chunk,
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
    if qdrant_results:
        logger.info("RAG searched by Qdrant. knowledge_base_ids=%s results=%s", knowledge_base_ids, len(qdrant_results))
        return rerank_items(question, qdrant_results, "chunk", top_k)

    target_ids = set(knowledge_base_ids)
    with _lock:
        matched = [
            item for item in _chunks
            if item.knowledge_base_id in target_ids
        ]

    results = [
        {
            "doc_id": item.doc_id,
            "title": item.title,
            "chunk": item.chunk,
        }
        for item in matched[:candidate_limit]
    ]
    return rerank_items(question, results, "chunk", top_k)


def _candidate_limit(top_k: int) -> int:
    if not settings.rerank_enabled:
        return top_k
    return max(top_k, settings.rerank_candidate_limit)


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
                },
            }
        ]
    }
    with httpx.Client(timeout=settings.vector_backend_timeout_seconds) as client:
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
        with httpx.Client(timeout=settings.vector_backend_timeout_seconds) as client:
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
        })
    return results


def _ensure_qdrant_collection() -> None:
    with httpx.Client(timeout=settings.vector_backend_timeout_seconds) as client:
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
