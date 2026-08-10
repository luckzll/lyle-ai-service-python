from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import Lock
from time import time

from app.core.config import settings
from app.core.database import postgres_connection
from app.memory.fragments import get_memory_fragment


@dataclass
class ProfileEvidence:
    id: int
    user_id: str
    profile_id: int
    fragment_id: int
    relation: str
    confidence: float
    created_at: float


_lock = Lock()
_evidence: dict[str, dict[int, ProfileEvidence]] = {}
_next_id = 1
_postgres_ready = False


def create_profile_evidence(
    user_id: str,
    profile_id: int,
    fragment_id: int,
    *,
    relation: str = "support",
    confidence: float = 0.8,
) -> dict[str, object]:
    now = time()
    relation = _normalise_relation(relation)
    if settings.postgres_enabled:
        _ensure_postgres_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO profile_evidence (user_id, profile_id, fragment_id, relation, confidence, created_at)
                    VALUES (%s, %s, %s, %s, %s, to_timestamp(%s))
                    ON CONFLICT (profile_id, fragment_id, relation)
                    DO UPDATE SET confidence = GREATEST(profile_evidence.confidence, EXCLUDED.confidence)
                    RETURNING id, user_id, profile_id, fragment_id, relation, confidence, extract(epoch from created_at)
                    """,
                    (user_id, profile_id, fragment_id, relation, confidence, now),
                )
                row = cur.fetchone()
            conn.commit()
        return _evidence_to_api(_evidence_from_row(row), user_id)

    global _next_id
    with _lock:
        existing = next(
            (
                item
                for item in _evidence.get(user_id, {}).values()
                if item.profile_id == profile_id and item.fragment_id == fragment_id and item.relation == relation
            ),
            None,
        )
        if existing:
            existing.confidence = max(existing.confidence, confidence)
            return _evidence_to_api(existing, user_id)
        item = ProfileEvidence(
            id=_next_id,
            user_id=user_id,
            profile_id=profile_id,
            fragment_id=fragment_id,
            relation=relation,
            confidence=confidence,
            created_at=now,
        )
        _next_id += 1
        _evidence.setdefault(user_id, {})[item.id] = item
    return _evidence_to_api(item, user_id)


def list_profile_evidence(user_id: str, profile_id: int, *, limit: int = 100) -> list[dict[str, object]]:
    safe_limit = max(1, min(limit, 500))
    if settings.postgres_enabled:
        _ensure_postgres_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, user_id, profile_id, fragment_id, relation, confidence, extract(epoch from created_at)
                    FROM profile_evidence
                    WHERE user_id = %s AND profile_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (user_id, profile_id, safe_limit),
                )
                rows = cur.fetchall()
        return [_evidence_to_api(_evidence_from_row(row), user_id) for row in rows]

    with _lock:
        items = [item for item in _evidence.get(user_id, {}).values() if item.profile_id == profile_id]
    items.sort(key=lambda item: item.created_at, reverse=True)
    return [_evidence_to_api(item, user_id) for item in items[:safe_limit]]


def evidence_count_by_profile(user_id: str) -> dict[int, int]:
    if settings.postgres_enabled:
        _ensure_postgres_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT profile_id, COUNT(*)
                    FROM profile_evidence
                    WHERE user_id = %s
                    GROUP BY profile_id
                    """,
                    (user_id,),
                )
                rows = cur.fetchall()
        return {int(row[0]): int(row[1]) for row in rows}
    with _lock:
        counts: dict[int, int] = {}
        for item in _evidence.get(user_id, {}).values():
            counts[item.profile_id] = counts.get(item.profile_id, 0) + 1
        return counts


def delete_profile_evidence(user_id: str, profile_id: int) -> int:
    """Hard delete source links before a profile privacy purge."""
    if settings.postgres_enabled:
        _ensure_postgres_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM profile_evidence WHERE user_id = %s AND profile_id = %s",
                    (user_id, profile_id),
                )
                deleted = max(0, cur.rowcount)
            conn.commit()
        return deleted
    with _lock:
        user_evidence = _evidence.get(user_id, {})
        deleted_ids = [item_id for item_id, item in user_evidence.items() if item.profile_id == profile_id]
        for item_id in deleted_ids:
            user_evidence.pop(item_id, None)
        if not user_evidence:
            _evidence.pop(user_id, None)
    return len(deleted_ids)


def _ensure_postgres_schema() -> None:
    global _postgres_ready
    if _postgres_ready:
        return
    from app.memory.fragments import _ensure_postgres_schema as ensure_fragment_schema
    from app.memory.profile_store import _ensure_postgres_schema as ensure_profile_schema

    ensure_fragment_schema()
    ensure_profile_schema()
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS profile_evidence (
                  id BIGSERIAL PRIMARY KEY,
                  user_id VARCHAR(128) NOT NULL,
                  profile_id BIGINT NOT NULL REFERENCES user_profile(id) ON DELETE CASCADE,
                  fragment_id BIGINT NOT NULL REFERENCES memory_fragment(id) ON DELETE CASCADE,
                  relation VARCHAR(32) NOT NULL DEFAULT 'support',
                  confidence DOUBLE PRECISION NOT NULL DEFAULT 0.8,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  UNIQUE(profile_id, fragment_id, relation)
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_profile_evidence_profile "
                "ON profile_evidence(profile_id, created_at DESC)"
            )
        conn.commit()
    _postgres_ready = True


def _normalise_relation(relation: str) -> str:
    value = (relation or "support").strip().lower()
    return value if value in {
        "support", "supplement", "contradict", "correction", "duplicate", "manual"
    } else "support"


def _evidence_from_row(row: tuple) -> ProfileEvidence:
    return ProfileEvidence(
        id=int(row[0]),
        user_id=row[1],
        profile_id=int(row[2]),
        fragment_id=int(row[3]),
        relation=row[4] or "support",
        confidence=float(row[5] or 0.8),
        created_at=float(row[6]),
    )


def _evidence_to_api(item: ProfileEvidence, user_id: str) -> dict[str, object]:
    result = asdict(item)
    result["userId"] = result.pop("user_id")
    result["profileId"] = result.pop("profile_id")
    result["fragmentId"] = result.pop("fragment_id")
    result["createdAt"] = result.pop("created_at")
    fragment = get_memory_fragment(user_id, item.fragment_id)
    if fragment:
        result["fragment"] = fragment
    return result
