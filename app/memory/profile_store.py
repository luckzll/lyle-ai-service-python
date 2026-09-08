from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from threading import Lock
from time import time

from app.core.config import settings
from app.core.database import postgres_connection
from app.memory.profile_schema import is_multi_value_path, schema_item
from app.memory.schema_store import schema_dirty


@dataclass
class UserProfile:
    id: int
    user_id: str
    schema_path: str
    value: str
    summary: str
    confidence: float
    status: str
    last_used_at: float | None
    hit_count: int
    correction_count: int
    created_at: float
    updated_at: float


_lock = Lock()
_postgres_schema_lock = Lock()
_profiles: dict[str, dict[str, UserProfile]] = {}
_next_id = 1
_postgres_ready = False
PROFILE_MAX_CONFIDENCE = 0.95
PROFILE_MONTHLY_DECAY = 0.01
PROFILE_DECAY_INTERVAL_SECONDS = 30 * 24 * 60 * 60
PROFILE_ACTIVE_RECALL_MIN_CONFIDENCE = 0.5


@schema_dirty
def upsert_user_profile(
    user_id: str,
    schema_path: str,
    value: str,
    *,
    summary: str = "",
    confidence: float = 0.8,
    status: str = "active",
) -> dict[str, object]:
    now = time()
    status = _normalise_profile_status(status)
    existing = get_user_profile(user_id, schema_path)
    merged_value = _merge_profile_value(schema_path, str(existing.get("value") or "") if existing else "", value)
    if settings.postgres_enabled:
        _ensure_postgres_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO user_profile (
                      user_id, schema_path, value, summary, confidence, evidence_confidence, status, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, to_timestamp(%s), to_timestamp(%s))
                    ON CONFLICT (user_id, schema_path)
                    DO UPDATE SET
                      value = EXCLUDED.value,
                      summary = CASE WHEN EXCLUDED.summary = '' THEN user_profile.summary ELSE EXCLUDED.summary END,
                      confidence = GREATEST(user_profile.confidence, EXCLUDED.confidence),
                      evidence_confidence = GREATEST(
                        COALESCE(user_profile.evidence_confidence, user_profile.confidence),
                        EXCLUDED.evidence_confidence
                      ),
                      status = EXCLUDED.status,
                      updated_at = EXCLUDED.updated_at
                    RETURNING id, user_id, schema_path, value, summary, confidence, status,
                              extract(epoch from last_used_at), hit_count, correction_count,
                              extract(epoch from created_at), extract(epoch from updated_at)
                    """,
                    (user_id, schema_path, merged_value, summary, confidence, confidence, status, now, now),
                )
                row = cur.fetchone()
            conn.commit()
        return _profile_to_api(_profile_from_row(row))

    global _next_id
    with _lock:
        user_profiles = _profiles.setdefault(user_id, {})
        current = user_profiles.get(schema_path)
        if current is None:
            current = UserProfile(
                id=_next_id,
                user_id=user_id,
                schema_path=schema_path,
                value=merged_value,
                summary=summary,
                confidence=_clamp_confidence(confidence),
                status=status,
                last_used_at=None,
                hit_count=0,
                correction_count=0,
                created_at=now,
                updated_at=now,
            )
            _next_id += 1
        else:
            current.value = merged_value
            current.summary = summary or current.summary
            current.confidence = _clamp_confidence(max(current.confidence, confidence))
            current.status = status
            current.updated_at = now
        user_profiles[schema_path] = current
    return _profile_to_api(current)


def get_user_profile(user_id: str, schema_path: str) -> dict[str, object] | None:
    if settings.postgres_enabled:
        _ensure_postgres_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, user_id, schema_path, value, summary, confidence, status,
                           extract(epoch from last_used_at), hit_count, correction_count,
                           extract(epoch from created_at), extract(epoch from updated_at)
                    FROM user_profile
                    WHERE user_id = %s AND schema_path = %s
                    """,
                    (user_id, schema_path),
                )
                row = cur.fetchone()
        return _profile_to_api(_profile_from_row(row)) if row else None

    with _lock:
        item = _profiles.get(user_id, {}).get(schema_path)
    return _profile_to_api(item) if item else None


def list_user_profiles(user_id: str, *, status: str | None = None, limit: int = 200) -> list[dict[str, object]]:
    safe_limit = max(1, min(limit, 500))
    if settings.postgres_enabled:
        _ensure_postgres_schema()
        clauses = ["user_id = %s"]
        params: list[object] = [user_id]
        if status:
            clauses.append("status = %s")
            params.append(status)
        params.append(safe_limit)
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, user_id, schema_path, value, summary, confidence, status,
                           extract(epoch from last_used_at), hit_count, correction_count,
                           extract(epoch from created_at), extract(epoch from updated_at)
                    FROM user_profile
                    WHERE {" AND ".join(clauses)}
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                    params,
                )
                rows = cur.fetchall()
        return [_profile_to_api(_profile_from_row(row)) for row in rows]

    with _lock:
        items = list(_profiles.get(user_id, {}).values())
    if status:
        items = [item for item in items if item.status == status]
    items.sort(key=lambda item: item.updated_at, reverse=True)
    return [_profile_to_api(item) for item in items[:safe_limit]]


@schema_dirty
def set_user_profile_status(user_id: str, schema_path: str, status: str) -> dict[str, object]:
    status = _normalise_profile_status(status)
    now = time()
    if settings.postgres_enabled:
        _ensure_postgres_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE user_profile
                    SET status = %s,
                        correction_count = correction_count + CASE WHEN %s = 'conflicted' THEN 1 ELSE 0 END,
                        updated_at = to_timestamp(%s)
                    WHERE user_id = %s AND schema_path = %s
                    RETURNING id, user_id, schema_path, value, summary, confidence, status,
                              extract(epoch from last_used_at), hit_count, correction_count,
                              extract(epoch from created_at), extract(epoch from updated_at)
                    """,
                    (status, status, now, user_id, schema_path),
                )
                row = cur.fetchone()
            conn.commit()
        return _profile_to_api(_profile_from_row(row)) if row else {}

    with _lock:
        item = _profiles.get(user_id, {}).get(schema_path)
        if not item:
            return {}
        item.status = status
        item.updated_at = now
        if status == "conflicted":
            item.correction_count += 1
    return _profile_to_api(item)


def delete_user_profile(user_id: str, schema_path: str) -> bool:
    """Hard delete reserved for the asynchronous privacy purge worker."""
    if settings.postgres_enabled:
        _ensure_postgres_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM user_profile WHERE user_id = %s AND schema_path = %s",
                    (user_id, schema_path),
                )
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted
    with _lock:
        user_profiles = _profiles.get(user_id, {})
        deleted = user_profiles.pop(schema_path, None) is not None
        if not user_profiles:
            _profiles.pop(user_id, None)
    return deleted


@schema_dirty
def replace_user_profile_value(
    user_id: str,
    schema_path: str,
    value: str,
    *,
    summary: str | None = None,
    confidence: float | None = None,
    status: str = "active",
) -> dict[str, object]:
    """Replace a profile value after an explicit user governance decision.

    This deliberately bypasses the multi-value merge used during extraction: a
    correction is the user's chosen current truth, not another supporting item.
    """
    current = get_user_profile(user_id, schema_path)
    if not current:
        return upsert_user_profile(
            user_id,
            schema_path,
            value,
            summary=summary if summary is not None else value,
            confidence=confidence if confidence is not None else 0.9,
            status=status,
        )
    now = time()
    next_confidence = _clamp_confidence(
        confidence if confidence is not None else max(float(current.get("confidence") or 0.8), 0.9),
        PROFILE_MAX_CONFIDENCE,
    )
    normalized_status = _normalise_profile_status(status)
    next_summary = summary if summary is not None else value
    if settings.postgres_enabled:
        _ensure_postgres_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE user_profile
                    SET value = %s,
                        summary = %s,
                        confidence = %s,
                        evidence_confidence = %s,
                        status = %s,
                        correction_count = correction_count + 1,
                        updated_at = to_timestamp(%s)
                    WHERE user_id = %s AND schema_path = %s
                    RETURNING id, user_id, schema_path, value, summary, confidence, status,
                              extract(epoch from last_used_at), hit_count, correction_count,
                              extract(epoch from created_at), extract(epoch from updated_at)
                    """,
                    (
                        value,
                        next_summary,
                        next_confidence,
                        next_confidence,
                        normalized_status,
                        now,
                        user_id,
                        schema_path,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        return _profile_to_api(_profile_from_row(row)) if row else {}

    with _lock:
        item = _profiles.get(user_id, {}).get(schema_path)
        if not item:
            return {}
        item.value = value
        item.summary = next_summary
        item.confidence = next_confidence
        item.status = normalized_status
        item.correction_count += 1
        item.updated_at = now
    return _profile_to_api(item)


@schema_dirty
def update_user_profile_summary(
    user_id: str,
    schema_path: str,
    summary: str,
    *,
    confidence_boost: float = 0.0,
) -> dict[str, object]:
    summary = summary.strip()
    if not summary:
        return get_user_profile(user_id, schema_path) or {}
    now = time()
    if settings.postgres_enabled:
        _ensure_postgres_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE user_profile
                    SET summary = %s,
                        confidence = LEAST(%s, confidence + %s),
                        evidence_confidence = LEAST(
                          %s,
                          COALESCE(evidence_confidence, confidence) + %s
                        ),
                        updated_at = to_timestamp(%s)
                    WHERE user_id = %s AND schema_path = %s
                    RETURNING id, user_id, schema_path, value, summary, confidence, status,
                              extract(epoch from last_used_at), hit_count, correction_count,
                              extract(epoch from created_at), extract(epoch from updated_at)
                    """,
                    (
                        summary,
                        PROFILE_MAX_CONFIDENCE,
                        confidence_boost,
                        PROFILE_MAX_CONFIDENCE,
                        confidence_boost,
                        now,
                        user_id,
                        schema_path,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        return _profile_to_api(_profile_from_row(row)) if row else {}

    with _lock:
        item = _profiles.get(user_id, {}).get(schema_path)
        if not item:
            return {}
        item.summary = summary
        item.confidence = _clamp_confidence(item.confidence + confidence_boost, PROFILE_MAX_CONFIDENCE)
        item.updated_at = now
    return _profile_to_api(item)


def mark_profiles_used(user_id: str, schema_paths: list[str]) -> None:
    if not schema_paths:
        return
    now = time()
    if settings.postgres_enabled:
        _ensure_postgres_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE user_profile
                    SET hit_count = hit_count + 1,
                        last_used_at = to_timestamp(%s)
                    WHERE user_id = %s AND schema_path = ANY(%s)
                    """,
                    (now, user_id, schema_paths),
                )
            conn.commit()
        return
    with _lock:
        for path in schema_paths:
            item = _profiles.get(user_id, {}).get(path)
            if item:
                item.hit_count += 1
                item.last_used_at = now


def apply_profile_decay(*, user_id: str | None = None, now: float | None = None) -> dict[str, int]:
    now = now if now is not None else time()
    updated = 0
    expired = 0
    if settings.postgres_enabled:
        _ensure_postgres_schema()
        clauses = ["status = 'active'"]
        params: list[object] = []
        if user_id:
            clauses.append("user_id = %s")
            params.append(user_id)
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, user_id, schema_path, value, summary, confidence, status,
                           extract(epoch from last_used_at), hit_count, correction_count,
                           extract(epoch from created_at), extract(epoch from updated_at)
                    FROM user_profile
                    WHERE {" AND ".join(clauses)}
                    """,
                    params,
                )
                rows = cur.fetchall()
                for row in rows:
                    profile = _profile_from_row(row)
                    decayed = _effective_profile_confidence(profile, now)
                    if decayed >= profile.confidence:
                        continue
                    status = "expired" if decayed < PROFILE_ACTIVE_RECALL_MIN_CONFIDENCE else "active"
                    cur.execute(
                        """
                        UPDATE user_profile
                        SET confidence = %s, status = %s, updated_at = to_timestamp(%s)
                        WHERE id = %s
                        """,
                        (decayed, status, now, profile.id),
                    )
                    updated += 1
                    if status == "expired":
                        expired += 1
            conn.commit()
        return {"updated": updated, "expired": expired}

    with _lock:
        users = [user_id] if user_id else list(_profiles.keys())
        for current_user_id in users:
            for item in _profiles.get(current_user_id or "", {}).values():
                if item.status != "active":
                    continue
                decayed = _effective_profile_confidence(item, now)
                if decayed >= item.confidence:
                    continue
                item.confidence = decayed
                item.updated_at = now
                updated += 1
                if decayed < PROFILE_ACTIVE_RECALL_MIN_CONFIDENCE:
                    item.status = "expired"
                    expired += 1
    return {"updated": updated, "expired": expired}


def _ensure_postgres_schema() -> None:
    global _postgres_ready
    if _postgres_ready:
        return
    with _postgres_schema_lock:
        if _postgres_ready:
            return
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_profile (
                      id BIGSERIAL PRIMARY KEY,
                      user_id VARCHAR(128) NOT NULL,
                      schema_path VARCHAR(256) NOT NULL,
                      value TEXT NOT NULL,
                      summary TEXT NOT NULL DEFAULT '',
                      confidence DOUBLE PRECISION NOT NULL DEFAULT 0.8,
                      evidence_confidence DOUBLE PRECISION,
                      status VARCHAR(32) NOT NULL DEFAULT 'active',
                      last_used_at TIMESTAMPTZ,
                      hit_count BIGINT NOT NULL DEFAULT 0,
                      correction_count BIGINT NOT NULL DEFAULT 0,
                      created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                      updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                      UNIQUE(user_id, schema_path)
                    )
                    """
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_user_profile_user_status "
                    "ON user_profile(user_id, status, updated_at DESC)"
                )
                cur.execute("ALTER TABLE user_profile ADD COLUMN IF NOT EXISTS evidence_confidence DOUBLE PRECISION")
                cur.execute("UPDATE user_profile SET evidence_confidence = confidence WHERE evidence_confidence IS NULL")
            conn.commit()
        _postgres_ready = True


def _merge_profile_value(schema_path: str, existing_value: str, new_value: str) -> str:
    if not existing_value or not is_multi_value_path(schema_path):
        return new_value
    existing_items = _split_items(existing_value)
    new_items = _split_items(new_value)
    merged = list(existing_items)
    for item in new_items:
        if item and item not in merged:
            merged.append(item)
    return "、".join(merged) if merged else new_value


def _split_items(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[、,，;；]+", value or "") if item.strip()]


def _normalise_profile_status(status: str) -> str:
    value = (status or "active").strip().lower()
    return value if value in {"active", "disabled", "forgotten", "expired", "conflicted"} else "active"


def _profile_from_row(row: tuple) -> UserProfile:
    return UserProfile(
        id=int(row[0]),
        user_id=row[1],
        schema_path=row[2],
        value=row[3],
        summary=row[4] or "",
        confidence=_clamp_confidence(row[5] or 0.8),
        status=row[6] or "active",
        last_used_at=float(row[7]) if row[7] is not None else None,
        hit_count=int(row[8] or 0),
        correction_count=int(row[9] or 0),
        created_at=float(row[10]),
        updated_at=float(row[11]),
    )


def _profile_to_api(profile: UserProfile) -> dict[str, object]:
    item = asdict(profile)
    item["confidence"] = _effective_profile_confidence(profile)
    item["evidenceConfidence"] = item["confidence"]
    item["userId"] = item.pop("user_id")
    item["schemaPath"] = item.pop("schema_path")
    item["lastUsedAt"] = item.pop("last_used_at")
    item["hitCount"] = item.pop("hit_count")
    item["correctionCount"] = item.pop("correction_count")
    item["createdAt"] = item.pop("created_at")
    item["updatedAt"] = item.pop("updated_at")
    schema = schema_item(profile.schema_path)
    item["label"] = schema.label
    item["group"] = schema.group
    item["multiValue"] = schema.multi_value
    item["sensitive"] = schema.sensitive
    return item


def _effective_profile_confidence(profile: UserProfile, now: float | None = None) -> float:
    reference = profile.last_used_at or profile.updated_at or profile.created_at
    if not reference:
        return _clamp_confidence(profile.confidence)
    elapsed = max(0.0, (now if now is not None else time()) - reference)
    months_unused = int(elapsed // PROFILE_DECAY_INTERVAL_SECONDS)
    if months_unused <= 0:
        return _clamp_confidence(profile.confidence)
    return _clamp_confidence(profile.confidence - months_unused * PROFILE_MONTHLY_DECAY)


def _clamp_confidence(value: float, upper: float = 1.0) -> float:
    return round(min(upper, max(0.0, float(value))), 4)


# Compatibility API: public commands write facts; legacy rows are ID anchors
# and migration inputs only. Keep profile:<id> URLs and evidence FKs stable.
_legacy_list_user_profiles = list_user_profiles
_legacy_get_user_profile = get_user_profile
_legacy_upsert_user_profile = upsert_user_profile
_legacy_set_user_profile_status = set_user_profile_status
_legacy_replace_user_profile_value = replace_user_profile_value
_legacy_update_user_profile_summary = update_user_profile_summary
_legacy_delete_user_profile = delete_user_profile
_legacy_apply_profile_decay = apply_profile_decay


def list_user_profiles(user_id, *, status=None, limit=200):
    from app.memory import fact_store
    if fact_store.enabled():
        return fact_store.project_profiles(user_id, status=status, limit=limit)
    return _legacy_list_user_profiles(user_id, status=status, limit=limit)


def get_user_profile(user_id, schema_path):
    from app.memory import fact_store
    if fact_store.enabled():
        return next((p for p in list_user_profiles(user_id, limit=500) if p['schemaPath'] == schema_path), None)
    return _legacy_get_user_profile(user_id, schema_path)


def upsert_user_profile(user_id, schema_path, value, *, summary='', confidence=.8, status='active'):
    from app.memory import fact_store
    if fact_store.enabled():
        return fact_store.write_slot(user_id, schema_path, value, confidence=confidence, status=_normalise_profile_status(status))
    return _legacy_upsert_user_profile(user_id, schema_path, value, summary=summary, confidence=confidence, status=status)


def set_user_profile_status(user_id, schema_path, status):
    from app.memory import fact_store
    if fact_store.enabled():
        return fact_store.write_slot(user_id, schema_path, status=_normalise_profile_status(status))
    return _legacy_set_user_profile_status(user_id, schema_path, status)


def replace_user_profile_value(user_id, schema_path, value, *, summary=None, confidence=None, status='active'):
    from app.memory import fact_store
    if fact_store.enabled():
        return fact_store.write_slot(user_id, schema_path, value, replace=True, confidence=confidence or .95,
                                     status=_normalise_profile_status(status))
    return _legacy_replace_user_profile_value(user_id, schema_path, value, summary=summary, confidence=confidence, status=status)


def update_user_profile_summary(user_id, schema_path, summary, *, confidence_boost=0.0):
    from app.memory import fact_store
    if fact_store.enabled():
        return get_user_profile(user_id, schema_path) or {}  # Derived text cannot change facts or confidence.
    return _legacy_update_user_profile_summary(user_id, schema_path, summary, confidence_boost=confidence_boost)


def delete_user_profile(user_id, schema_path):
    from app.memory import fact_store
    if fact_store.enabled():
        fact_store.purge_slot(user_id, schema_path)
    return _legacy_delete_user_profile(user_id, schema_path)


def apply_profile_decay(*, user_id=None, now=None):
    from app.memory import fact_store
    if fact_store.enabled():
        # Confidence is derived at read time from fact confirmation and usage.
        # Never mutate retired profile values or double-apply decay.
        return {'updated': 0, 'expired': 0}
    return _legacy_apply_profile_decay(user_id=user_id, now=now)
