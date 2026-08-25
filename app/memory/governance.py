from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from time import time
from typing import Any

import httpx

from app.core.config import settings
from app.core.database import postgres_connection
from app.core.qdrant import qdrant_headers
from app.memory.evidence_store import (
    create_profile_evidence,
    delete_profile_evidence,
    evidence_count_by_profile,
    list_profile_evidence,
)
from app.memory.fragments import (
    create_memory_fragment,
    delete_memory_fragments,
    get_memory_fragment,
    list_memory_fragments,
    set_memory_fragment_status,
)
from app.memory.profile_schema import schema_item, schema_path_for_memory
from app.memory.profile_store import (
    delete_user_profile,
    get_user_profile,
    list_user_profiles,
    replace_user_profile_value,
    set_user_profile_status,
)


class MemoryGovernanceNotFound(LookupError):
    pass


class MemoryGovernanceConflict(RuntimeError):
    def __init__(self, message: str, current: dict[str, object] | None = None):
        super().__init__(message)
        self.current = current or {}


class MemoryGovernanceValidation(ValueError):
    pass


@dataclass
class _Conflict:
    id: int
    user_id: str
    schema_path: str
    status: str
    current_profile_id: int | None
    candidate_fragment_id: int | None
    conflict_fingerprint: str
    resolution: str | None
    resolved_value: str | None
    version: int
    created_at: float
    updated_at: float
    resolved_at: float | None


_lock = Lock()
_conflicts: dict[str, dict[int, _Conflict]] = {}
_events: dict[str, list[dict[str, object]]] = {}
_jobs: dict[str, dict[int, dict[str, object]]] = {}
_next_conflict_id = 1
_next_event_id = 1
_next_job_id = 1
_schema_ready = False


def reliability_for_profile(profile: dict[str, object]) -> str:
    status = str(profile.get("status") or "active").lower()
    confidence = float(profile.get("evidenceConfidence") or profile.get("confidence") or 0)
    if status in {"conflicted", "pending"} or confidence < 0.6:
        return "REVIEW_REQUIRED"
    if confidence >= 0.82:
        return "RELIABLE"
    return "BASIC"


def list_memory_records(
    user_id: str,
    *,
    status: str | None = None,
    query: str | None = None,
    limit: int = 100,
) -> dict[str, object]:
    safe_limit = max(1, min(limit, 300))
    wanted = (status or "").strip().upper()
    needle = (query or "").strip().lower()
    counts = evidence_count_by_profile(user_id)
    items: list[dict[str, object]] = []
    for profile in list_user_profiles(user_id, limit=500):
        record = memory_record_from_profile(user_id, profile, counts.get(int(profile.get("id") or 0), 0))
        if wanted and str(record["status"]) != wanted:
            continue
        if needle and needle not in " ".join(
            str(record.get(key) or "").lower() for key in ("title", "value", "schemaPath")
        ):
            continue
        items.append(record)
    items.sort(key=lambda item: float(item.get("updatedAtEpoch") or 0), reverse=True)
    return {"items": items[:safe_limit], "nextCursor": None}


def get_memory_record(user_id: str, memory_id: str) -> dict[str, object]:
    profile = _profile_for_memory_id(user_id, memory_id)
    if not profile:
        raise MemoryGovernanceNotFound("记忆不存在")
    record = memory_record_from_profile(user_id, profile)
    record["sources"] = list_memory_sources(user_id, memory_id)["items"]
    record["history"] = [
        event
        for event in list_governance_events(user_id, limit=100)["items"]
        if str(event.get("subjectId")) == memory_id
    ][:20]
    return record


def memory_record_from_profile(
    user_id: str,
    profile: dict[str, object],
    source_count: int | None = None,
) -> dict[str, object]:
    profile_id = int(profile.get("id") or 0)
    schema_path = str(profile.get("schemaPath") or "")
    status = _public_status(str(profile.get("status") or "active"))
    if source_count is None:
        source_count = len(list_profile_evidence(user_id, profile_id, limit=500))
    latest_source = None
    if source_count:
        sources = list_memory_sources(user_id, f"profile:{profile_id}", limit=1)["items"]
        latest_source = sources[0] if sources else None
    return {
        "memoryId": f"profile:{profile_id}",
        "profileId": profile_id,
        "schemaPath": schema_path,
        "title": str(profile.get("label") or schema_item(schema_path).label or "长期记忆"),
        "value": str(profile.get("summary") or profile.get("value") or ""),
        "rawValue": str(profile.get("value") or ""),
        "status": status,
        "reliability": reliability_for_profile(profile),
        "confidence": float(profile.get("evidenceConfidence") or profile.get("confidence") or 0),
        "confirmedByUser": _profile_confirmed_by_user(user_id, profile_id),
        "sourceCount": source_count,
        "latestSource": latest_source,
        "usageDescription": _usage_description(schema_path),
        "lastUsedAt": _iso(profile.get("lastUsedAt")),
        "updatedAt": _iso(profile.get("updatedAt")),
        "updatedAtEpoch": float(profile.get("updatedAt") or 0),
        "allowedActions": _allowed_actions(status),
    }


def list_memory_sources(user_id: str, memory_id: str, *, limit: int = 100) -> dict[str, object]:
    profile = _profile_for_memory_id(user_id, memory_id)
    if not profile:
        raise MemoryGovernanceNotFound("记忆不存在")
    evidence = list_profile_evidence(user_id, int(profile["id"]), limit=max(1, min(limit, 300)))
    access_by_evidence = _validate_source_access(user_id, evidence)
    return {
        "items": [
            _source_from_evidence(item, access_by_evidence.get(str(item.get("id"))))
            for item in evidence
        ]
    }


def list_memory_inbox(user_id: str) -> dict[str, object]:
    items: list[dict[str, object]] = []
    for conflict in _list_conflicts(user_id, status="OPEN"):
        items.append({
            "inboxId": f"conflict:{conflict['id']}",
            "type": "CONFLICT",
            "title": f"{conflict.get('title') or '记忆'}需要核对",
            "description": "两次表达可能不一致，处理前不会主动使用这条信息。",
            "status": "OPEN",
            "createdAt": conflict.get("createdAt"),
            "conflict": conflict,
        })
    # Pending structured memories stay compatible with the pre-V6.6 confirmation flow.
    from app.memory.store import _load_all_structured, _structured_target_key

    for candidate in _load_all_structured(user_id):
        if candidate.status != "pending" or not candidate.key.startswith("pending:"):
            continue
        target_key = _structured_target_key(candidate.key)
        schema_path = schema_path_for_memory(target_key, candidate.value)
        items.append({
            "inboxId": f"candidate:{candidate.key}",
            "type": "CANDIDATE",
            "title": f"{schema_item(schema_path).label}待确认",
            "description": candidate.value,
            "status": "PENDING",
            "createdAt": _iso(candidate.updated_at),
            "candidate": {
                "key": candidate.key,
                "targetKey": target_key,
                "schemaPath": schema_path,
                "value": candidate.value,
                "confidence": candidate.confidence,
            },
        })
    for record in list_memory_records(user_id, limit=300)["items"]:
        if record.get("reliability") == "REVIEW_REQUIRED" and record.get("status") == "ACTIVE":
            items.append({
                "inboxId": f"review:{record['memoryId']}",
                "type": "LOW_RELIABILITY",
                "title": f"{record['title']}可靠性较低",
                "description": "建议核对来源或纠正当前内容。",
                "status": "OPEN",
                "createdAt": record.get("updatedAt"),
                "memory": record,
            })
    for job in list_purge_jobs(user_id, statuses={"FAILED"})["items"]:
        items.append({
            "inboxId": f"purge:{job['jobId']}",
            "type": "PURGE_FAILED",
            "title": "记忆删除尚未完成",
            "description": "Lyle 不会继续使用它，你可以重试清理。",
            "status": "OPEN",
            "createdAt": job.get("updatedAt"),
            "purgeJob": job,
        })
    items.sort(key=lambda item: str(item.get("createdAt") or ""), reverse=True)
    return {"items": items, "count": len(items)}


def create_memory_conflict(
    user_id: str,
    schema_path: str,
    candidate_fragment_id: int,
) -> dict[str, object]:
    current = get_user_profile(user_id, schema_path)
    candidate = get_memory_fragment(user_id, candidate_fragment_id)
    if not current or not candidate:
        raise MemoryGovernanceNotFound("无法创建冲突：记忆或候选来源不存在")
    if str(candidate.get("category") or "") != schema_path:
        raise MemoryGovernanceValidation("候选来源与记忆类型不一致")
    fingerprint = _semantic_conflict_fingerprint(user_id, schema_path, current, candidate)
    existing = _find_open_conflict(user_id, fingerprint)
    if existing:
        return _hydrate_conflict(user_id, existing)
    now = time()
    if settings.postgres_enabled:
        _ensure_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memory_conflict (
                      user_id, schema_path, current_profile_id, candidate_fragment_id,
                      conflict_fingerprint, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, to_timestamp(%s), to_timestamp(%s))
                    ON CONFLICT (user_id, conflict_fingerprint) WHERE status = 'OPEN'
                    DO UPDATE SET updated_at = memory_conflict.updated_at
                    RETURNING id, user_id, schema_path, status, current_profile_id, candidate_fragment_id,
                              conflict_fingerprint, resolution, resolved_value, version,
                              extract(epoch from created_at), extract(epoch from updated_at),
                              extract(epoch from resolved_at)
                    """,
                    (user_id, schema_path, int(current["id"]), candidate_fragment_id, fingerprint, now, now),
                )
                row = cur.fetchone()
            conn.commit()
        conflict = _conflict_to_api(_conflict_from_row(row))
    else:
        global _next_conflict_id
        with _lock:
            item = next(
                (
                    conflict
                    for conflict in _conflicts.get(user_id, {}).values()
                    if conflict.status == "OPEN" and conflict.conflict_fingerprint == fingerprint
                ),
                None,
            )
            if item is None:
                conflict_id = _next_conflict_id
                _next_conflict_id += 1
                item = _Conflict(
                    conflict_id, user_id, schema_path, "OPEN", int(current["id"]), candidate_fragment_id,
                    fingerprint, None, None, 1, now, now, None,
                )
                _conflicts.setdefault(user_id, {})[conflict_id] = item
        conflict = _conflict_to_api(item)
    if not settings.memory_conflict_shadow_enabled:
        set_user_profile_status(user_id, schema_path, "conflicted")
    return _hydrate_conflict(user_id, conflict)


def _semantic_conflict_fingerprint(
    user_id: str,
    schema_path: str,
    current: dict[str, object],
    candidate: dict[str, object],
) -> str:
    """Identify the conflicting claim, not the evidence row that carried it."""
    payload = json.dumps(
        [
            _canonical_conflict_text(user_id),
            _canonical_conflict_text(schema_path),
            _canonical_conflict_text(current.get("value")),
            _canonical_conflict_text(candidate.get("normalizedText") or candidate.get("sourceText")),
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_conflict_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    return re.sub(r"\s+", " ", text)


def resolve_memory_conflict(
    user_id: str,
    conflict_id: int,
    *,
    resolution: str,
    merged_value: str | None,
    expected_version: int,
    client_request_id: str,
) -> dict[str, object]:
    existing_event = _event_by_request(user_id, client_request_id)
    if existing_event:
        return _idempotent_event_result(user_id, existing_event)
    resolution = resolution.strip().upper()
    if resolution not in {"KEEP_CURRENT", "USE_CANDIDATE", "MERGE", "USE_NEITHER", "FORGET"}:
        raise MemoryGovernanceValidation("不支持的冲突处理方式")
    conflict = _get_conflict(user_id, conflict_id)
    if not conflict:
        raise MemoryGovernanceNotFound("待处理记忆不存在")
    if str(conflict.get("status")) != "OPEN":
        raise MemoryGovernanceConflict("这条记忆已经更新", conflict)
    if int(conflict.get("version") or 0) != expected_version:
        raise MemoryGovernanceConflict("这条记忆已在其他页面更新", conflict)
    current = _profile_for_memory_id(user_id, f"profile:{conflict['currentProfileId']}")
    candidate = get_memory_fragment(user_id, int(conflict.get("candidateFragmentId") or 0))
    if not current or not candidate:
        raise MemoryGovernanceNotFound("冲突关联的记忆来源已失效")
    candidate_value = str(candidate.get("normalizedText") or candidate.get("sourceText") or "").strip()
    resolved_value: str | None = None
    if resolution == "KEEP_CURRENT":
        set_user_profile_status(user_id, str(current["schemaPath"]), "active")
        set_memory_fragment_status(user_id, int(candidate["id"]), "disabled")
        resolved_value = str(current.get("value") or "")
    elif resolution == "USE_CANDIDATE":
        resolved_value = candidate_value
        replace_user_profile_value(user_id, str(current["schemaPath"]), resolved_value, summary=resolved_value)
        set_memory_fragment_status(user_id, int(candidate["id"]), "active")
    elif resolution == "MERGE":
        resolved_value = (merged_value or "").strip()
        if not resolved_value:
            raise MemoryGovernanceValidation("合并内容不能为空")
        replace_user_profile_value(user_id, str(current["schemaPath"]), resolved_value, summary=resolved_value)
        set_memory_fragment_status(user_id, int(candidate["id"]), "active")
    elif resolution == "USE_NEITHER":
        set_user_profile_status(user_id, str(current["schemaPath"]), "disabled")
        set_memory_fragment_status(user_id, int(candidate["id"]), "disabled")
    else:
        from app.memory.store import set_profile_memory_status

        set_profile_memory_status(user_id, str(current["schemaPath"]), "forgotten")
        set_memory_fragment_status(user_id, int(candidate["id"]), "forgotten")

    updated_conflict = _mark_conflict_resolved(
        user_id, conflict_id, resolution, resolved_value, expected_version
    )
    latest_record = None
    try:
        latest_record = get_memory_record(user_id, f"profile:{current['id']}")
    except MemoryGovernanceNotFound:
        pass
    result = {
        "conflict": updated_conflict,
        "memory": latest_record,
        "resolution": resolution,
    }
    event = _record_event(
        user_id,
        subject_type="CONFLICT",
        subject_id=str(conflict_id),
        action="CONFLICT_RESOLVED",
        before=conflict,
        after={"status": "RESOLVED", "resolution": resolution, "memoryId": f"profile:{current['id']}"},
        reason_code=resolution,
        source_ref="",
        client_request_id=client_request_id,
        result=result,
    )
    result["governanceEventId"] = event["eventId"]
    return result


def govern_memory_record(
    user_id: str,
    memory_id: str,
    *,
    action: str,
    client_request_id: str,
    value: str | None = None,
    reason: str | None = None,
) -> dict[str, object]:
    existing_event = _event_by_request(user_id, client_request_id)
    if existing_event:
        return _idempotent_event_result(user_id, existing_event)
    profile = _profile_for_memory_id(user_id, memory_id)
    if not profile:
        raise MemoryGovernanceNotFound("记忆不存在")
    action = action.strip().upper()
    schema_path = str(profile.get("schemaPath") or "")
    before = memory_record_from_profile(user_id, profile)
    if action == "CORRECT":
        next_value = (value or "").strip()
        if not next_value:
            raise MemoryGovernanceValidation("纠正后的内容不能为空")
        source_text = f"用户纠正：{next_value}"
        fragment = create_memory_fragment(
            user_id,
            source_text=source_text,
            normalized_text=next_value,
            category=schema_path,
            source_type="manual",
            source_ref="manual:correction",
            source_created_at=time(),
            confidence=0.95,
            status="active",
        )
        updated = replace_user_profile_value(user_id, schema_path, next_value, summary=next_value, confidence=0.95)
        create_profile_evidence(
            user_id, int(updated["id"]), int(fragment["id"]), relation="correction", confidence=0.95
        )
        _disable_stale_structured_memory(user_id, schema_path)
    elif action in {"DISABLE", "RESTORE", "FORGET"}:
        from app.memory.store import set_profile_memory_status

        target_status = {"DISABLE": "disabled", "RESTORE": "active", "FORGET": "forgotten"}[action]
        set_profile_memory_status(user_id, schema_path, target_status)
    else:
        raise MemoryGovernanceValidation("不支持的记忆治理动作")
    after = get_memory_record(user_id, memory_id)
    result = {"memory": after, "action": action}
    event = _record_event(
        user_id,
        subject_type="RECORD",
        subject_id=memory_id,
        action=action,
        before=before,
        after=after,
        reason_code=(reason or "USER_REQUEST")[:64],
        source_ref="manual",
        client_request_id=client_request_id,
        result=result,
    )
    result["governanceEventId"] = event["eventId"]
    return result


def request_memory_purge(user_id: str, memory_id: str, *, client_request_id: str) -> dict[str, object]:
    existing = _purge_job_by_request(user_id, client_request_id)
    if existing:
        return existing
    profile = _profile_for_memory_id(user_id, memory_id)
    if not profile:
        raise MemoryGovernanceNotFound("记忆不存在")
    from app.memory.store import set_profile_memory_status

    set_profile_memory_status(user_id, str(profile["schemaPath"]), "forgotten")
    now = time()
    if settings.postgres_enabled:
        _ensure_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memory_purge_job (
                      user_id, memory_id, status, next_attempt_at, client_request_id, created_at, updated_at
                    ) VALUES (%s, %s, 'PENDING', to_timestamp(%s), %s, to_timestamp(%s), to_timestamp(%s))
                    ON CONFLICT (user_id, client_request_id)
                    DO UPDATE SET updated_at = memory_purge_job.updated_at
                    RETURNING id, user_id, memory_id, status, attempt_count,
                              extract(epoch from next_attempt_at), step_state_json, client_request_id,
                              extract(epoch from created_at), extract(epoch from updated_at),
                              extract(epoch from completed_at)
                    """,
                    (user_id, memory_id, now, client_request_id, now, now),
                )
                row = cur.fetchone()
            conn.commit()
        job = _purge_job_from_row(row)
    else:
        global _next_job_id
        with _lock:
            job_id = _next_job_id
            _next_job_id += 1
            job = {
                "jobId": job_id,
                "userId": user_id,
                "memoryId": memory_id,
                "status": "PENDING",
                "attemptCount": 0,
                "nextAttemptAt": _iso(now),
                "steps": {},
                "clientRequestId": client_request_id,
                "createdAt": _iso(now),
                "updatedAt": _iso(now),
                "completedAt": None,
            }
            _jobs.setdefault(user_id, {})[job_id] = job
    _record_event(
        user_id,
        subject_type="PURGE_JOB",
        subject_id=str(job["jobId"]),
        action="PURGE_REQUESTED",
        before={"memoryId": memory_id, "status": _public_status(str(profile.get("status") or ""))},
        after={"memoryId": memory_id, "status": "PURGING", "jobId": job["jobId"]},
        reason_code="USER_REQUEST",
        source_ref="",
        client_request_id=f"purge-event:{client_request_id}",
        result={"job": job},
    )
    return deepcopy(job)


def get_purge_job(user_id: str, job_id: int) -> dict[str, object]:
    job = _get_purge_job(user_id, job_id)
    if not job:
        raise MemoryGovernanceNotFound("删除任务不存在")
    return job


def retry_purge_job(user_id: str, job_id: int) -> dict[str, object]:
    job = _get_purge_job(user_id, job_id)
    if not job:
        raise MemoryGovernanceNotFound("删除任务不存在")
    if str(job.get("status")) not in {"FAILED", "RETRY"}:
        return job
    _update_purge_job(user_id, job_id, status="PENDING", next_attempt_at=time())
    return get_purge_job(user_id, job_id)


def list_purge_jobs(user_id: str, *, statuses: set[str] | None = None) -> dict[str, object]:
    if settings.postgres_enabled:
        _ensure_schema()
        params: list[object] = [user_id]
        clause = ""
        if statuses:
            clause = " AND status = ANY(%s)"
            params.append(sorted(statuses))
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, user_id, memory_id, status, attempt_count,
                           extract(epoch from next_attempt_at), step_state_json, client_request_id,
                           extract(epoch from created_at), extract(epoch from updated_at),
                           extract(epoch from completed_at)
                    FROM memory_purge_job
                    WHERE user_id = %s {clause}
                    ORDER BY created_at DESC
                    """,
                    params,
                )
                rows = cur.fetchall()
        return {"items": [_purge_job_from_row(row) for row in rows]}
    with _lock:
        items = [deepcopy(item) for item in _jobs.get(user_id, {}).values()]
    if statuses:
        items = [item for item in items if str(item.get("status")) in statuses]
    items.sort(key=lambda item: str(item.get("createdAt") or ""), reverse=True)
    return {"items": items}


def process_pending_purge_jobs(*, limit: int = 5) -> dict[str, int]:
    jobs = _claim_purge_jobs(limit=max(1, min(limit, 20)))
    succeeded = 0
    failed = 0
    for job in jobs:
        try:
            _execute_purge_job(job)
            succeeded += 1
        except Exception:
            failed += 1
            attempts = int(job.get("attemptCount") or 0) + 1
            status = "FAILED" if attempts >= 5 else "RETRY"
            _update_purge_job(
                str(job["userId"]),
                int(job["jobId"]),
                status=status,
                attempt_count=attempts,
                next_attempt_at=time() + min(300, 2 ** attempts * 5),
            )
    return {"claimed": len(jobs), "succeeded": succeeded, "failed": failed}


def list_governance_events(user_id: str, *, limit: int = 100) -> dict[str, object]:
    safe_limit = max(1, min(limit, 300))
    if settings.postgres_enabled:
        _ensure_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, subject_type, subject_id, action, before_snapshot_json,
                           after_snapshot_json, reason_code, source_ref, client_request_id,
                           extract(epoch from created_at)
                    FROM memory_governance_event
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (user_id, safe_limit),
                )
                rows = cur.fetchall()
        return {"items": [_event_from_row(row) for row in rows], "nextCursor": None}
    with _lock:
        items = [deepcopy(item) for item in _events.get(user_id, [])]
    items.sort(key=lambda item: str(item.get("createdAt") or ""), reverse=True)
    return {"items": items[:safe_limit], "nextCursor": None}


def _source_from_evidence(
    item: dict[str, object],
    access: tuple[bool, str | None] | None = None,
) -> dict[str, object]:
    fragment = item.get("fragment") if isinstance(item.get("fragment"), dict) else {}
    status = str(fragment.get("status") or "")
    available = bool(fragment) and status not in {"forgotten", "purged"}
    unavailable_reason = "原始内容已删除或无法访问" if not available else None
    if available and access is not None:
        available, unavailable_reason = access
    source_type = str(fragment.get("sourceType") or "manual").upper()
    if source_type == "KNOWLEDGE_FILE":
        source_type = "KNOWLEDGE"
    occurred_at = fragment.get("sourceCreatedAt") or fragment.get("createdAt") or item.get("createdAt")
    conversation_id = fragment.get("conversationId")
    message_id = fragment.get("sourceMessageId")
    chunk_id = fragment.get("sourceChunkId")
    link = None
    if available and source_type == "CONVERSATION" and conversation_id:
        link = f"/?conversationId={conversation_id}"
        if message_id:
            link += f"&messageId={message_id}"
    elif available and source_type == "KNOWLEDGE" and fragment.get("sourceId"):
        link = f"/knowledge?fileId={fragment['sourceId']}"
        if chunk_id:
            link += f"&chunkId={chunk_id}"
    return {
        "sourceId": f"evidence:{item.get('id')}",
        "type": source_type,
        "label": _source_label(source_type, fragment),
        "relation": str(item.get("relation") or "support").upper(),
        "excerpt": str(fragment.get("sourceText") or "") if available else "",
        "occurredAt": _iso(occurred_at),
        "availability": "AVAILABLE" if available else "UNAVAILABLE",
        "unavailableReason": unavailable_reason if not available else None,
        "conversationId": conversation_id,
        "messageId": message_id,
        "chunkId": chunk_id,
        "link": link,
    }


def _validate_source_access(
    user_id: str,
    evidence: list[dict[str, object]],
) -> dict[str, tuple[bool, str | None]]:
    if not settings.memory_source_validation_enabled:
        return {}

    result: dict[str, tuple[bool, str | None]] = {}
    requests: list[dict[str, object]] = []
    request_keys: set[str] = set()
    for item in evidence:
        key = str(item.get("id") or "")
        fragment = item.get("fragment") if isinstance(item.get("fragment"), dict) else {}
        if not key or not fragment or str(fragment.get("status") or "") in {"forgotten", "purged"}:
            continue
        source_type = str(fragment.get("sourceType") or "manual").upper()
        if source_type == "KNOWLEDGE_FILE":
            source_type = "KNOWLEDGE"
        if source_type == "CONVERSATION":
            conversation_id = str(fragment.get("conversationId") or "").strip()
            message_id = str(fragment.get("sourceMessageId") or "").strip()
            if not conversation_id or not message_id:
                result[key] = (False, "原会话或消息定位信息不完整")
                continue
            request = {
                "id": key,
                "type": source_type,
                "conversationId": conversation_id,
                "messageId": message_id,
            }
        elif source_type == "KNOWLEDGE":
            source_id = str(fragment.get("sourceId") or "").strip()
            if not source_id:
                result[key] = (False, "原知识文件定位信息不完整")
                continue
            request = {
                "id": key,
                "type": source_type,
                "sourceId": source_id,
                "chunkId": str(fragment.get("sourceChunkId") or "").strip() or None,
            }
        else:
            continue
        result[key] = (False, "原始内容已删除、无权访问或校验暂不可用")
        request_keys.add(key)
        requests.append(request)

    if not requests:
        return result
    try:
        url = settings.memory_source_validator_base_url.rstrip("/") + "/internal/memory-sources:validate"
        with httpx.Client(timeout=max(0.25, settings.memory_source_validation_timeout_seconds)) as client:
            response = client.post(
                url,
                headers={"X-Internal-Token": settings.internal_api_token},
                json={"userId": user_id, "sources": requests},
            )
            response.raise_for_status()
            payload = response.json()
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            raise ValueError("source validation response has no items")
        for item in items:
            if not isinstance(item, dict):
                continue
            key = str(item.get("id") or "")
            if key not in request_keys:
                continue
            available = item.get("available") is True
            reason = str(item.get("reason") or "").strip() or None
            result[key] = (available, None if available else reason or "原始内容已删除或无法访问")
    except (httpx.HTTPError, ValueError, TypeError):
        if not settings.memory_source_validation_fail_closed:
            for key in request_keys:
                result.pop(key, None)
    return result


def _profile_for_memory_id(user_id: str, memory_id: str) -> dict[str, object] | None:
    if not memory_id.startswith("profile:"):
        return None
    try:
        profile_id = int(memory_id.split(":", 1)[1])
    except ValueError:
        return None
    return next(
        (item for item in list_user_profiles(user_id, limit=500) if int(item.get("id") or 0) == profile_id),
        None,
    )


def _profile_confirmed_by_user(user_id: str, profile_id: int) -> bool:
    for evidence in list_profile_evidence(user_id, profile_id, limit=100):
        relation = str(evidence.get("relation") or "")
        fragment = evidence.get("fragment") if isinstance(evidence.get("fragment"), dict) else {}
        if relation in {"manual", "correction"} or str(fragment.get("sourceType")) == "manual":
            return True
    return False


def _usage_description(schema_path: str) -> str:
    if schema_path.startswith("preferences.visual"):
        return "Lyle 会在图片、视觉风格和人物形象相关的问题中使用它。"
    if schema_path.startswith("preferences.communication") or schema_path == "preferences.output_style":
        return "Lyle 会在组织回复和输出内容时使用它。"
    if schema_path.startswith("projects."):
        return "Lyle 会在当前项目、规划和推进相关的问题中使用它。"
    if schema_path.startswith("goals."):
        return "Lyle 会在目标、计划和复盘相关的问题中使用它。"
    return "Lyle 只会在与你的问题直接相关时使用它。"


def _public_status(status: str) -> str:
    return {
        "active": "ACTIVE",
        "pending": "PENDING",
        "conflicted": "CONFLICTED",
        "disabled": "DISABLED",
        "forgotten": "FORGOTTEN",
        "expired": "REVIEW_REQUIRED",
        "purging": "PURGING",
    }.get(status.lower(), status.upper() or "ACTIVE")


def _allowed_actions(status: str) -> list[str]:
    if status == "ACTIVE":
        return ["CORRECT", "DISABLE", "FORGET", "PURGE"]
    if status == "DISABLED":
        return ["RESTORE", "FORGET", "PURGE"]
    if status in {"FORGOTTEN", "REVIEW_REQUIRED"}:
        return ["CORRECT", "PURGE"]
    if status == "CONFLICTED":
        return ["RESOLVE", "FORGET", "PURGE"]
    return ["PURGE"]


def _source_label(source_type: str, fragment: dict[str, object]) -> str:
    if source_type == "CONVERSATION":
        return "与 Lyle 的对话"
    if source_type == "KNOWLEDGE":
        return str(fragment.get("sourceId") or "知识库文件")
    if source_type == "MANUAL":
        return "用户手动确认"
    return "历史数据"


def _iso(value: object) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return value
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def _disable_stale_structured_memory(user_id: str, schema_path: str) -> None:
    from app.memory.store import _load_all_structured, set_structured_memory_status

    for item in _load_all_structured(user_id):
        if item.key.startswith("pending:"):
            continue
        if schema_path_for_memory(item.key, item.value) == schema_path and item.status == "active":
            set_structured_memory_status(user_id, item.key, "disabled")
    set_user_profile_status(user_id, schema_path, "active")


def _hydrate_conflict(user_id: str, conflict: dict[str, object]) -> dict[str, object]:
    current = _profile_for_memory_id(user_id, f"profile:{conflict.get('currentProfileId')}")
    candidate = get_memory_fragment(user_id, int(conflict.get("candidateFragmentId") or 0))
    result = deepcopy(conflict)
    result["title"] = str(current.get("label") or "记忆") if current else "记忆"
    result["currentMemory"] = memory_record_from_profile(user_id, current) if current else None
    result["candidate"] = {
        "fragmentId": candidate.get("id"),
        "value": candidate.get("normalizedText") or candidate.get("sourceText"),
        "source": _source_from_fragment(candidate),
    } if candidate else None
    return result


def _source_from_fragment(fragment: dict[str, object]) -> dict[str, object]:
    source_type = str(fragment.get("sourceType") or "manual").upper()
    return {
        "type": source_type,
        "label": _source_label(source_type, fragment),
        "excerpt": str(fragment.get("sourceText") or ""),
        "occurredAt": _iso(fragment.get("sourceCreatedAt") or fragment.get("createdAt")),
        "conversationId": fragment.get("conversationId"),
        "messageId": fragment.get("sourceMessageId"),
        "chunkId": fragment.get("sourceChunkId"),
    }


def _find_open_conflict(user_id: str, fingerprint: str) -> dict[str, object] | None:
    for item in _list_conflicts(user_id, status="OPEN"):
        if str(item.get("conflictFingerprint")) == fingerprint:
            return item
    return None


def _list_conflicts(user_id: str, *, status: str | None = None) -> list[dict[str, object]]:
    if settings.postgres_enabled:
        _ensure_schema()
        params: list[object] = [user_id]
        clause = ""
        if status:
            clause = " AND status = %s"
            params.append(status)
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, user_id, schema_path, status, current_profile_id, candidate_fragment_id,
                           conflict_fingerprint, resolution, resolved_value, version,
                           extract(epoch from created_at), extract(epoch from updated_at),
                           extract(epoch from resolved_at)
                    FROM memory_conflict
                    WHERE user_id = %s {clause}
                    ORDER BY updated_at DESC
                    """,
                    params,
                )
                rows = cur.fetchall()
        return [_hydrate_conflict(user_id, _conflict_to_api(_conflict_from_row(row))) for row in rows]
    with _lock:
        values = list(_conflicts.get(user_id, {}).values())
    if status:
        values = [item for item in values if item.status == status]
    values.sort(key=lambda item: item.updated_at, reverse=True)
    return [_hydrate_conflict(user_id, _conflict_to_api(item)) for item in values]


def _get_conflict(user_id: str, conflict_id: int) -> dict[str, object] | None:
    return next((item for item in _list_conflicts(user_id) if int(item.get("id") or 0) == conflict_id), None)


def _mark_conflict_resolved(
    user_id: str, conflict_id: int, resolution: str, value: str | None, expected_version: int
) -> dict[str, object]:
    now = time()
    if settings.postgres_enabled:
        _ensure_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE memory_conflict
                    SET status = 'RESOLVED', resolution = %s, resolved_value = %s,
                        version = version + 1, updated_at = to_timestamp(%s), resolved_at = to_timestamp(%s)
                    WHERE user_id = %s AND id = %s AND status = 'OPEN' AND version = %s
                    RETURNING id, user_id, schema_path, status, current_profile_id, candidate_fragment_id,
                              conflict_fingerprint, resolution, resolved_value, version,
                              extract(epoch from created_at), extract(epoch from updated_at),
                              extract(epoch from resolved_at)
                    """,
                    (resolution, value, now, now, user_id, conflict_id, expected_version),
                )
                row = cur.fetchone()
            conn.commit()
        if not row:
            raise MemoryGovernanceConflict("这条记忆已在其他页面更新", _get_conflict(user_id, conflict_id))
        return _hydrate_conflict(user_id, _conflict_to_api(_conflict_from_row(row)))
    with _lock:
        item = _conflicts.get(user_id, {}).get(conflict_id)
        if not item or item.status != "OPEN" or item.version != expected_version:
            raise MemoryGovernanceConflict("这条记忆已在其他页面更新")
        item.status = "RESOLVED"
        item.resolution = resolution
        item.resolved_value = value
        item.version += 1
        item.updated_at = now
        item.resolved_at = now
        result = _conflict_to_api(item)
    return _hydrate_conflict(user_id, result)


def _conflict_from_row(row: tuple) -> _Conflict:
    return _Conflict(
        id=int(row[0]), user_id=row[1], schema_path=row[2], status=row[3],
        current_profile_id=int(row[4]) if row[4] is not None else None,
        candidate_fragment_id=int(row[5]) if row[5] is not None else None,
        conflict_fingerprint=row[6], resolution=row[7], resolved_value=row[8], version=int(row[9]),
        created_at=float(row[10]), updated_at=float(row[11]),
        resolved_at=float(row[12]) if row[12] is not None else None,
    )


def _conflict_to_api(item: _Conflict) -> dict[str, object]:
    return {
        "id": item.id,
        "schemaPath": item.schema_path,
        "status": item.status,
        "currentProfileId": item.current_profile_id,
        "candidateFragmentId": item.candidate_fragment_id,
        "conflictFingerprint": item.conflict_fingerprint,
        "resolution": item.resolution,
        "resolvedValue": item.resolved_value,
        "version": item.version,
        "createdAt": _iso(item.created_at),
        "updatedAt": _iso(item.updated_at),
        "resolvedAt": _iso(item.resolved_at),
    }


def _record_event(
    user_id: str,
    *,
    subject_type: str,
    subject_id: str,
    action: str,
    before: dict[str, object],
    after: dict[str, object],
    reason_code: str,
    source_ref: str,
    client_request_id: str,
    result: dict[str, object],
) -> dict[str, object]:
    existing = _event_by_request(user_id, client_request_id)
    if existing:
        return existing
    now = time()
    if settings.postgres_enabled:
        _ensure_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memory_governance_event (
                      user_id, subject_type, subject_id, action, before_snapshot_json,
                      after_snapshot_json, reason_code, source_ref, client_request_id, created_at
                    ) VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, to_timestamp(%s))
                    ON CONFLICT (user_id, client_request_id)
                    DO UPDATE SET client_request_id = memory_governance_event.client_request_id
                    RETURNING id, subject_type, subject_id, action, before_snapshot_json,
                              after_snapshot_json, reason_code, source_ref, client_request_id,
                              extract(epoch from created_at)
                    """,
                    (
                        user_id, subject_type, subject_id, action,
                        json.dumps(before, ensure_ascii=False), json.dumps(after, ensure_ascii=False),
                        reason_code, source_ref, client_request_id, now,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        event = _event_from_row(row)
    else:
        global _next_event_id
        with _lock:
            event = {
                "eventId": _next_event_id,
                "subjectType": subject_type,
                "subjectId": subject_id,
                "action": action,
                "beforeSnapshot": deepcopy(before),
                "afterSnapshot": deepcopy(after),
                "reasonCode": reason_code,
                "sourceRef": source_ref,
                "clientRequestId": client_request_id,
                "createdAt": _iso(now),
            }
            _next_event_id += 1
            _events.setdefault(user_id, []).append(event)
    event["result"] = deepcopy(result)
    if not settings.postgres_enabled:
        with _lock:
            for stored in _events.get(user_id, []):
                if stored.get("eventId") == event.get("eventId"):
                    stored["result"] = deepcopy(result)
    return event


def _event_by_request(user_id: str, client_request_id: str) -> dict[str, object] | None:
    if settings.postgres_enabled:
        _ensure_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, subject_type, subject_id, action, before_snapshot_json,
                           after_snapshot_json, reason_code, source_ref, client_request_id,
                           extract(epoch from created_at)
                    FROM memory_governance_event
                    WHERE user_id = %s AND client_request_id = %s
                    """,
                    (user_id, client_request_id),
                )
                row = cur.fetchone()
        return _event_from_row(row) if row else None
    with _lock:
        item = next(
            (event for event in _events.get(user_id, []) if event.get("clientRequestId") == client_request_id),
            None,
        )
        return deepcopy(item) if item else None


def _idempotent_event_result(user_id: str, event: dict[str, object]) -> dict[str, object]:
    stored = event.get("result")
    if isinstance(stored, dict) and stored:
        return deepcopy(stored)
    subject_type = str(event.get("subjectType") or "")
    subject_id = str(event.get("subjectId") or "")
    if subject_type == "RECORD":
        return {
            "memory": get_memory_record(user_id, subject_id),
            "action": str(event.get("action") or ""),
            "governanceEventId": int(event.get("eventId") or 0),
        }
    if subject_type == "CONFLICT":
        conflict = _get_conflict(user_id, int(subject_id))
        memory = None
        if conflict and conflict.get("currentProfileId"):
            try:
                memory = get_memory_record(user_id, f"profile:{conflict['currentProfileId']}")
            except MemoryGovernanceNotFound:
                pass
        after = event.get("afterSnapshot") if isinstance(event.get("afterSnapshot"), dict) else {}
        return {
            "conflict": conflict,
            "memory": memory,
            "resolution": after.get("resolution"),
            "governanceEventId": int(event.get("eventId") or 0),
        }
    return deepcopy(event.get("afterSnapshot") or {})


def _event_from_row(row: tuple) -> dict[str, object]:
    return {
        "eventId": int(row[0]),
        "subjectType": row[1],
        "subjectId": row[2],
        "action": row[3],
        "beforeSnapshot": row[4] or {},
        "afterSnapshot": row[5] or {},
        "reasonCode": row[6] or "",
        "sourceRef": row[7] or "",
        "clientRequestId": row[8],
        "createdAt": _iso(row[9]),
    }


def _purge_job_by_request(user_id: str, client_request_id: str) -> dict[str, object] | None:
    return next(
        (
            item for item in list_purge_jobs(user_id)["items"]
            if str(item.get("clientRequestId")) == client_request_id
        ),
        None,
    )


def _get_purge_job(user_id: str, job_id: int) -> dict[str, object] | None:
    return next(
        (item for item in list_purge_jobs(user_id)["items"] if int(item.get("jobId") or 0) == job_id),
        None,
    )


def _claim_purge_jobs(limit: int) -> list[dict[str, object]]:
    if settings.postgres_enabled:
        _ensure_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id
                    FROM memory_purge_job
                    WHERE status IN ('PENDING', 'RETRY')
                      AND (next_attempt_at IS NULL OR next_attempt_at <= now())
                    ORDER BY created_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT %s
                    """,
                    (limit,),
                )
                ids = [int(row[0]) for row in cur.fetchall()]
                if ids:
                    cur.execute(
                        """
                        UPDATE memory_purge_job
                        SET status = 'RUNNING', updated_at = now()
                        WHERE id = ANY(%s)
                        """,
                        (ids,),
                    )
            conn.commit()
        jobs: list[dict[str, object]] = []
        for user_job in _all_purge_jobs_by_ids(ids):
            jobs.append(user_job)
        return jobs
    now = time()
    claimed: list[dict[str, object]] = []
    with _lock:
        for user_jobs in _jobs.values():
            for job in user_jobs.values():
                if len(claimed) >= limit:
                    break
                next_at = job.get("nextAttemptAt")
                next_epoch = datetime.fromisoformat(str(next_at)).timestamp() if next_at else 0
                if job.get("status") in {"PENDING", "RETRY"} and next_epoch <= now:
                    job["status"] = "RUNNING"
                    job["updatedAt"] = _iso(now)
                    claimed.append(deepcopy(job))
    return claimed


def _all_purge_jobs_by_ids(ids: list[int]) -> list[dict[str, object]]:
    if not ids:
        return []
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, user_id, memory_id, status, attempt_count,
                       extract(epoch from next_attempt_at), step_state_json, client_request_id,
                       extract(epoch from created_at), extract(epoch from updated_at),
                       extract(epoch from completed_at)
                FROM memory_purge_job WHERE id = ANY(%s)
                """,
                (ids,),
            )
            return [_purge_job_from_row(row) for row in cur.fetchall()]


def _execute_purge_job(job: dict[str, object]) -> None:
    user_id = str(job["userId"])
    job_id = int(job["jobId"])
    memory_id = str(job["memoryId"])
    profile = _profile_for_memory_id(user_id, memory_id)
    schema_path = str(profile.get("schemaPath") or "") if profile else ""
    steps = dict(job.get("steps") or {})
    from app.memory.store import _load_all_structured, purge_structured_memory

    if not steps.get("structured"):
        for item in _load_all_structured(user_id):
            if item.key.startswith("pending:"):
                continue
            if schema_path and schema_path_for_memory(item.key, item.value) == schema_path:
                purge_structured_memory(user_id, item.key)
        steps["structured"] = "SUCCEEDED"
        _update_purge_job(user_id, job_id, steps=steps)
    if not steps.get("profile"):
        if profile:
            delete_profile_evidence(user_id, int(profile["id"]))
            delete_user_profile(user_id, schema_path)
        steps["profile"] = "SUCCEEDED"
        _update_purge_job(user_id, job_id, steps=steps)
    if not steps.get("fragments"):
        if schema_path:
            delete_memory_fragments(user_id, category=schema_path)
        steps["fragments"] = "SUCCEEDED"
        _update_purge_job(user_id, job_id, steps=steps)
    if not steps.get("vectors"):
        _purge_user_vectors(user_id)
        steps["vectors"] = "SUCCEEDED"
        _update_purge_job(user_id, job_id, steps=steps)
    if not steps.get("cache"):
        _purge_user_memory_cache(user_id)
        steps["cache"] = "SUCCEEDED"
        _update_purge_job(user_id, job_id, steps=steps)
    steps["verified"] = "SUCCEEDED"
    _update_purge_job(
        user_id, job_id, status="SUCCEEDED", steps=steps, completed_at=time(), next_attempt_at=None
    )
    _redact_purge_events(user_id, memory_id)


def _purge_user_vectors(user_id: str) -> None:
    # Semantic turns are not keyed by profile. Privacy correctness therefore
    # takes precedence over retention and clears this user's derived vector cache.
    if settings.qdrant_enabled:
        with httpx.Client(
            timeout=max(1.0, settings.vector_backend_timeout_seconds),
            headers=qdrant_headers(),
        ) as client:
            client.post(
                f"{settings.qdrant_endpoint.rstrip('/')}/collections/{settings.qdrant_memory_collection}/points/delete",
                params={"wait": "true"},
                json={"filter": {"must": [{"key": "user_id", "match": {"value": user_id}}]}},
            ).raise_for_status()
    if settings.milvus_enabled:
        from app.memory.store import _milvus_client

        _milvus_client().delete(
            collection_name=settings.milvus_collection,
            filter=f'user_id == "{user_id}"',
        )


def _purge_user_memory_cache(user_id: str) -> None:
    if not settings.redis_enabled:
        return
    from app.memory.store import _redis_client

    client = _redis_client()
    keys = list(client.scan_iter(match=f"memory:short:{user_id}:*", count=100))
    if keys:
        client.delete(*keys)


def _update_purge_job(
    user_id: str,
    job_id: int,
    *,
    status: str | None = None,
    attempt_count: int | None = None,
    next_attempt_at: float | None | object = ...,
    steps: dict[str, object] | None = None,
    completed_at: float | None = None,
) -> None:
    now = time()
    if settings.postgres_enabled:
        _ensure_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE memory_purge_job
                    SET status = COALESCE(%s, status),
                        attempt_count = COALESCE(%s, attempt_count),
                        next_attempt_at = CASE
                          WHEN %s THEN CASE WHEN %s::double precision IS NULL THEN NULL ELSE to_timestamp(%s::double precision) END
                          ELSE next_attempt_at
                        END,
                        step_state_json = COALESCE(%s::jsonb, step_state_json),
                        completed_at = CASE WHEN %s::double precision IS NULL THEN completed_at ELSE to_timestamp(%s::double precision) END,
                        updated_at = to_timestamp(%s)
                    WHERE user_id = %s AND id = %s
                    """,
                    (
                        status, attempt_count,
                        next_attempt_at is not ..., None if next_attempt_at is ... else next_attempt_at,
                        None if next_attempt_at is ... else next_attempt_at,
                        json.dumps(steps, ensure_ascii=False) if steps is not None else None,
                        completed_at, completed_at, now, user_id, job_id,
                    ),
                )
            conn.commit()
        return
    with _lock:
        job = _jobs.get(user_id, {}).get(job_id)
        if not job:
            return
        if status is not None:
            job["status"] = status
        if attempt_count is not None:
            job["attemptCount"] = attempt_count
        if next_attempt_at is not ...:
            job["nextAttemptAt"] = _iso(next_attempt_at)
        if steps is not None:
            job["steps"] = deepcopy(steps)
        if completed_at is not None:
            job["completedAt"] = _iso(completed_at)
        job["updatedAt"] = _iso(now)


def _purge_job_from_row(row: tuple) -> dict[str, object]:
    return {
        "jobId": int(row[0]),
        "userId": row[1],
        "memoryId": row[2],
        "status": row[3],
        "attemptCount": int(row[4] or 0),
        "nextAttemptAt": _iso(row[5]),
        "steps": row[6] or {},
        "clientRequestId": row[7],
        "createdAt": _iso(row[8]),
        "updatedAt": _iso(row[9]),
        "completedAt": _iso(row[10]),
    }


def _redact_purge_events(user_id: str, memory_id: str) -> None:
    redacted = {"memoryIdHash": hashlib.sha256(memory_id.encode()).hexdigest(), "status": "SUCCEEDED"}
    if settings.postgres_enabled:
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE memory_governance_event
                    SET before_snapshot_json = %s::jsonb,
                        after_snapshot_json = %s::jsonb,
                        source_ref = ''
                    WHERE user_id = %s
                      AND (subject_id = %s OR before_snapshot_json ->> 'memoryId' = %s)
                    """,
                    (json.dumps(redacted), json.dumps(redacted), user_id, memory_id, memory_id),
                )
            conn.commit()
        return
    with _lock:
        for event in _events.get(user_id, []):
            if event.get("subjectId") == memory_id or (event.get("beforeSnapshot") or {}).get("memoryId") == memory_id:
                event["beforeSnapshot"] = deepcopy(redacted)
                event["afterSnapshot"] = deepcopy(redacted)
                event["sourceRef"] = ""


def _ensure_schema() -> None:
    global _schema_ready
    if _schema_ready or not settings.postgres_enabled:
        return
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE memory_fragment ADD COLUMN IF NOT EXISTS source_message_id VARCHAR(128)")
            cur.execute("ALTER TABLE memory_fragment ADD COLUMN IF NOT EXISTS source_chunk_id VARCHAR(256)")
            cur.execute("ALTER TABLE memory_fragment ADD COLUMN IF NOT EXISTS source_created_at TIMESTAMPTZ")
            cur.execute("ALTER TABLE user_profile ADD COLUMN IF NOT EXISTS evidence_confidence DOUBLE PRECISION")
            cur.execute("UPDATE user_profile SET evidence_confidence = confidence WHERE evidence_confidence IS NULL")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_conflict (
                  id BIGSERIAL PRIMARY KEY, user_id VARCHAR(128) NOT NULL, schema_path VARCHAR(256) NOT NULL,
                  status VARCHAR(32) NOT NULL DEFAULT 'OPEN', current_profile_id BIGINT,
                  candidate_fragment_id BIGINT, conflict_fingerprint VARCHAR(128) NOT NULL,
                  resolution VARCHAR(32), resolved_value TEXT, version BIGINT NOT NULL DEFAULT 1,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  resolved_at TIMESTAMPTZ
                )
                """
            )
            cur.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS uk_memory_conflict_open "
                "ON memory_conflict(user_id, conflict_fingerprint) WHERE status = 'OPEN'"
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_governance_event (
                  id BIGSERIAL PRIMARY KEY, user_id VARCHAR(128) NOT NULL, subject_type VARCHAR(32) NOT NULL,
                  subject_id VARCHAR(256) NOT NULL, action VARCHAR(64) NOT NULL,
                  before_snapshot_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                  after_snapshot_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                  reason_code VARCHAR(64) NOT NULL DEFAULT '', source_ref TEXT NOT NULL DEFAULT '',
                  client_request_id VARCHAR(64) NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  UNIQUE(user_id, client_request_id)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_purge_job (
                  id BIGSERIAL PRIMARY KEY, user_id VARCHAR(128) NOT NULL, memory_id VARCHAR(256) NOT NULL,
                  status VARCHAR(32) NOT NULL DEFAULT 'PENDING', attempt_count INTEGER NOT NULL DEFAULT 0,
                  next_attempt_at TIMESTAMPTZ, step_state_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                  client_request_id VARCHAR(64) NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(), completed_at TIMESTAMPTZ,
                  UNIQUE(user_id, client_request_id)
                )
                """
            )
        conn.commit()
    _schema_ready = True
