from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import Lock
from time import time

from app.core.config import settings
from app.core.database import postgres_connection


@dataclass
class MemoryFragment:
    id: int
    user_id: str
    conversation_id: str | None
    source_type: str
    source_id: str
    source_ref: str
    source_message_id: str | None
    source_chunk_id: str | None
    source_created_at: float | None
    source_text: str
    normalized_text: str
    category: str
    confidence: float
    status: str
    created_at: float
    updated_at: float


_lock = Lock()
_fragments: dict[str, dict[int, MemoryFragment]] = {}
_next_id = 1
_postgres_ready = False


def create_memory_fragment(
    user_id: str,
    source_text: str,
    normalized_text: str,
    category: str,
    *,
    conversation_id: str | None = None,
    source_type: str = "conversation",
    source_id: str = "",
    source_ref: str = "",
    source_message_id: str | None = None,
    source_chunk_id: str | None = None,
    source_created_at: float | None = None,
    confidence: float = 0.8,
    status: str = "active",
) -> dict[str, object]:
    now = time()
    status = _normalise_fragment_status(status)
    if settings.postgres_enabled:
        _ensure_postgres_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memory_fragment (
                      user_id, conversation_id, source_type, source_id, source_ref,
                      source_message_id, source_chunk_id, source_created_at, source_text,
                      normalized_text, category, confidence, status, created_at, updated_at
                    )
                    VALUES (
                      %s, %s, %s, %s, %s, %s, %s,
                      CASE WHEN %s::double precision IS NULL THEN NULL ELSE to_timestamp(%s::double precision) END,
                      %s, %s, %s, %s, %s, to_timestamp(%s), to_timestamp(%s)
                    )
                    RETURNING id, extract(epoch from created_at), extract(epoch from updated_at)
                    """,
                    (
                        user_id,
                        conversation_id,
                        source_type,
                        source_id,
                        source_ref,
                        source_message_id,
                        source_chunk_id,
                        source_created_at,
                        source_created_at,
                        source_text,
                        normalized_text,
                        category,
                        confidence,
                        status,
                        now,
                        now,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        fragment = MemoryFragment(
            id=int(row[0]),
            user_id=user_id,
            conversation_id=conversation_id,
            source_type=source_type,
            source_id=source_id,
            source_ref=source_ref,
            source_message_id=source_message_id,
            source_chunk_id=source_chunk_id,
            source_created_at=source_created_at,
            source_text=source_text,
            normalized_text=normalized_text,
            category=category,
            confidence=confidence,
            status=status,
            created_at=float(row[1]),
            updated_at=float(row[2]),
        )
        return _fragment_to_api(fragment)

    global _next_id
    with _lock:
        fragment_id = _next_id
        _next_id += 1
        fragment = MemoryFragment(
            id=fragment_id,
            user_id=user_id,
            conversation_id=conversation_id,
            source_type=source_type,
            source_id=source_id,
            source_ref=source_ref,
            source_message_id=source_message_id,
            source_chunk_id=source_chunk_id,
            source_created_at=source_created_at,
            source_text=source_text,
            normalized_text=normalized_text,
            category=category,
            confidence=confidence,
            status=status,
            created_at=now,
            updated_at=now,
        )
        _fragments.setdefault(user_id, {})[fragment.id] = fragment
    return _fragment_to_api(fragment)


def list_memory_fragments(
    user_id: str,
    *,
    status: str | None = None,
    category: str | None = None,
    limit: int = 100,
) -> list[dict[str, object]]:
    safe_limit = max(1, min(limit, 500))
    if settings.postgres_enabled:
        _ensure_postgres_schema()
        clauses = ["user_id = %s"]
        params: list[object] = [user_id]
        if status:
            clauses.append("status = %s")
            params.append(status)
        if category:
            clauses.append("category = %s")
            params.append(category)
        params.append(safe_limit)
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, user_id, conversation_id, source_type, source_id, source_ref,
                           source_message_id, source_chunk_id, extract(epoch from source_created_at), source_text,
                           normalized_text, category, confidence, status,
                           extract(epoch from created_at), extract(epoch from updated_at)
                    FROM memory_fragment
                    WHERE {" AND ".join(clauses)}
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                    params,
                )
                rows = cur.fetchall()
        return [_fragment_to_api(_fragment_from_row(row)) for row in rows]

    with _lock:
        items = list(_fragments.get(user_id, {}).values())
    if status:
        items = [item for item in items if item.status == status]
    if category:
        items = [item for item in items if item.category == category]
    items.sort(key=lambda item: item.updated_at, reverse=True)
    return [_fragment_to_api(item) for item in items[:safe_limit]]


def get_memory_fragment(user_id: str, fragment_id: int) -> dict[str, object] | None:
    if settings.postgres_enabled:
        _ensure_postgres_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, user_id, conversation_id, source_type, source_id, source_ref,
                           source_message_id, source_chunk_id, extract(epoch from source_created_at), source_text,
                           normalized_text, category, confidence, status,
                           extract(epoch from created_at), extract(epoch from updated_at)
                    FROM memory_fragment
                    WHERE user_id = %s AND id = %s
                    """,
                    (user_id, fragment_id),
                )
                row = cur.fetchone()
        return _fragment_to_api(_fragment_from_row(row)) if row else None

    with _lock:
        item = _fragments.get(user_id, {}).get(fragment_id)
    return _fragment_to_api(item) if item else None


def set_memory_fragment_status(user_id: str, fragment_id: int, status: str) -> dict[str, object]:
    status = _normalise_fragment_status(status)
    now = time()
    if settings.postgres_enabled:
        _ensure_postgres_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE memory_fragment
                    SET status = %s, updated_at = to_timestamp(%s)
                    WHERE user_id = %s AND id = %s
                    RETURNING id, user_id, conversation_id, source_type, source_id, source_ref,
                              source_message_id, source_chunk_id, extract(epoch from source_created_at), source_text,
                              normalized_text, category, confidence, status,
                              extract(epoch from created_at), extract(epoch from updated_at)
                    """,
                    (status, now, user_id, fragment_id),
                )
                row = cur.fetchone()
            conn.commit()
        return _fragment_to_api(_fragment_from_row(row)) if row else {}

    with _lock:
        item = _fragments.get(user_id, {}).get(fragment_id)
        if not item:
            return {}
        item.status = status
        item.updated_at = now
        return _fragment_to_api(item)


def delete_memory_fragments(user_id: str, *, category: str) -> list[int]:
    """Hard delete fragments for one governed profile category."""
    if settings.postgres_enabled:
        _ensure_postgres_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM memory_fragment WHERE user_id = %s AND category = %s RETURNING id",
                    (user_id, category),
                )
                deleted = [int(row[0]) for row in cur.fetchall()]
            conn.commit()
        return deleted
    with _lock:
        user_fragments = _fragments.get(user_id, {})
        deleted = [item_id for item_id, item in user_fragments.items() if item.category == category]
        for item_id in deleted:
            user_fragments.pop(item_id, None)
        if not user_fragments:
            _fragments.pop(user_id, None)
    return deleted


def _ensure_postgres_schema() -> None:
    global _postgres_ready
    if _postgres_ready:
        return
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_fragment (
                  id BIGSERIAL PRIMARY KEY,
                  user_id VARCHAR(128) NOT NULL,
                  conversation_id VARCHAR(128),
                  source_type VARCHAR(32) NOT NULL,
                  source_id VARCHAR(256) NOT NULL DEFAULT '',
                  source_ref TEXT NOT NULL DEFAULT '',
                  source_message_id VARCHAR(128),
                  source_chunk_id VARCHAR(256),
                  source_created_at TIMESTAMPTZ,
                  source_text TEXT NOT NULL,
                  normalized_text TEXT NOT NULL,
                  category VARCHAR(128) NOT NULL,
                  confidence DOUBLE PRECISION NOT NULL DEFAULT 0.8,
                  status VARCHAR(32) NOT NULL DEFAULT 'active',
                  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_fragment_user_status "
                "ON memory_fragment(user_id, status, updated_at DESC)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_fragment_user_category "
                "ON memory_fragment(user_id, category)"
            )
        conn.commit()
    _postgres_ready = True


def _normalise_fragment_status(status: str) -> str:
    value = (status or "active").strip().lower()
    return value if value in {"pending", "active", "disabled", "forgotten", "expired"} else "active"


def _fragment_from_row(row: tuple) -> MemoryFragment:
    return MemoryFragment(
        id=int(row[0]),
        user_id=row[1],
        conversation_id=row[2],
        source_type=row[3],
        source_id=row[4] or "",
        source_ref=row[5] or "",
        source_message_id=row[6],
        source_chunk_id=row[7],
        source_created_at=float(row[8]) if row[8] is not None else None,
        source_text=row[9],
        normalized_text=row[10],
        category=row[11],
        confidence=float(row[12] or 0.8),
        status=row[13] or "active",
        created_at=float(row[14]),
        updated_at=float(row[15]),
    )


def _fragment_to_api(fragment: MemoryFragment) -> dict[str, object]:
    item = asdict(fragment)
    item["userId"] = item.pop("user_id")
    item["conversationId"] = item.pop("conversation_id")
    item["sourceType"] = item.pop("source_type")
    item["sourceId"] = item.pop("source_id")
    item["sourceRef"] = item.pop("source_ref")
    item["sourceMessageId"] = item.pop("source_message_id")
    item["sourceChunkId"] = item.pop("source_chunk_id")
    item["sourceCreatedAt"] = item.pop("source_created_at")
    item["sourceText"] = item.pop("source_text")
    item["normalizedText"] = item.pop("normalized_text")
    item["createdAt"] = item.pop("created_at")
    item["updatedAt"] = item.pop("updated_at")
    return item
