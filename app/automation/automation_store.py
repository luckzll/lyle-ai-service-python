from __future__ import annotations

import json
from typing import Any

from fastapi import HTTPException

from app.automation.schedule import next_run_at
from app.core.config import settings
from app.core.database import postgres_connection
from app.projects.project_store import get_project
from app.workflows.workflow_store import get_workflow


TRIGGER_TYPES = {"schedule", "project_stale", "task_due"}


def create_automation(user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_postgres()
    workflow_id = int(payload.get("workflowId") or payload.get("workflow_id") or 0)
    get_workflow(user_id, workflow_id)
    name = str(payload.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="自动化名称不能为空。")
    trigger_type = _trigger_type(payload.get("triggerType") or "schedule")
    schedule_expr = str(payload.get("scheduleExpr") or payload.get("schedule_expr") or "daily") if trigger_type == "schedule" else None
    timezone = str(payload.get("timezone") or "Asia/Shanghai")
    input_payload = payload.get("input") if isinstance(payload.get("input"), dict) else {}
    _validate_trigger_input(user_id, trigger_type, input_payload)
    status = str(payload.get("status") or "paused")
    if status not in {"enabled", "paused"}:
        raise HTTPException(status_code=400, detail="自动化状态不合法。")
    next_at = _next_run_at(str(schedule_expr), timezone) if trigger_type == "schedule" else None
    with postgres_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO automation_rule
              (user_id, workflow_id, name, trigger_type, schedule_expr, timezone, status,
               next_run_at, input, max_retries, retry_delay_seconds, notify_on_failure)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
            RETURNING id
            """,
            (
                user_id,
                workflow_id,
                name[:256],
                trigger_type,
                schedule_expr,
                timezone,
                status,
                next_at,
                json.dumps(input_payload, ensure_ascii=False),
                _bounded_int(payload.get("maxRetries"), 0, 5, 0),
                _bounded_int(payload.get("retryDelaySeconds"), 10, 86400, 60),
                bool(payload.get("notifyOnFailure", True)),
            ),
        ).fetchone()
        conn.commit()
    return get_automation(user_id, int(row[0]))


def list_automations(user_id: str) -> list[dict[str, Any]]:
    _ensure_postgres()
    with postgres_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, workflow_id, name, trigger_type, schedule_expr, timezone, status,
                   next_run_at, last_run_at, input, max_retries, retry_delay_seconds,
                   consecutive_failures, notify_on_failure, lease_until, last_event_key,
                   last_event_at, pending_event_key, pending_event_input, created_at, updated_at
            FROM automation_rule
            WHERE user_id = %s
            ORDER BY updated_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [_row(row) for row in rows]


def get_automation(user_id: str, automation_id: int) -> dict[str, Any]:
    _ensure_postgres()
    with postgres_connection() as conn:
        row = conn.execute(
            """
            SELECT id, workflow_id, name, trigger_type, schedule_expr, timezone, status,
                   next_run_at, last_run_at, input, max_retries, retry_delay_seconds,
                   consecutive_failures, notify_on_failure, lease_until, last_event_key,
                   last_event_at, pending_event_key, pending_event_input, created_at, updated_at
            FROM automation_rule
            WHERE user_id = %s AND id = %s
            """,
            (user_id, automation_id),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="自动化不存在。")
    return _row(row)


def set_automation_status(user_id: str, automation_id: int, status: str) -> dict[str, Any]:
    _ensure_postgres()
    if status not in {"enabled", "paused"}:
        raise HTTPException(status_code=400, detail="自动化状态不合法。")
    current = get_automation(user_id, automation_id)
    next_at = (
        _next_run_at(str(current["scheduleExpr"]), str(current["timezone"]))
        if status == "enabled" and current["triggerType"] == "schedule"
        else None if status == "enabled" else current["nextRunAt"]
    )
    with postgres_connection() as conn:
        conn.execute(
            """
            UPDATE automation_rule
            SET status = %s, next_run_at = %s, lease_until = NULL,
                consecutive_failures = CASE WHEN %s = 'paused' THEN 0 ELSE consecutive_failures END,
                updated_at = now()
            WHERE user_id = %s AND id = %s
            """,
            (status, next_at, status, user_id, automation_id),
        )
        conn.commit()
    return get_automation(user_id, automation_id)


def update_automation(user_id: str, automation_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_postgres()
    current = get_automation(user_id, automation_id)
    workflow_id = int(payload.get("workflowId") or payload.get("workflow_id") or current["workflowId"])
    get_workflow(user_id, workflow_id)
    name = str(payload.get("name") or current["name"]).strip()
    if not name:
        raise HTTPException(status_code=400, detail="自动化名称不能为空。")
    trigger_type = _trigger_type(payload.get("triggerType") or current.get("triggerType") or "schedule")
    schedule_expr = (
        str(
            payload.get("scheduleExpr")
            or payload.get("schedule_expr")
            or (current.get("scheduleExpr") if current.get("triggerType") == "schedule" else None)
            or "daily"
        )
        if trigger_type == "schedule"
        else None
    )
    timezone = str(payload.get("timezone") or current["timezone"] or "Asia/Shanghai")
    status = str(payload.get("status") or current["status"])
    if status not in {"enabled", "paused"}:
        raise HTTPException(status_code=400, detail="自动化状态不合法。")
    input_payload = payload.get("input") if isinstance(payload.get("input"), dict) else current.get("input") or {}
    _validate_trigger_input(user_id, trigger_type, input_payload)
    max_retries = _bounded_int(payload.get("maxRetries"), 0, 5, int(current.get("maxRetries") or 0))
    retry_delay_seconds = _bounded_int(
        payload.get("retryDelaySeconds"), 10, 86400, int(current.get("retryDelaySeconds") or 60)
    )
    notify_on_failure = bool(payload.get("notifyOnFailure", current.get("notifyOnFailure", True)))
    candidate_next_at = _next_run_at(str(schedule_expr), timezone) if trigger_type == "schedule" else None
    next_at = candidate_next_at if status == "enabled" else current["nextRunAt"] if trigger_type == "schedule" else None
    event_config_changed = trigger_type != current.get("triggerType") or input_payload != (current.get("input") or {})
    with postgres_connection() as conn:
        conn.execute(
            """
            UPDATE automation_rule
            SET workflow_id = %s, name = %s, trigger_type = %s, schedule_expr = %s, timezone = %s,
                status = %s, next_run_at = %s, input = %s::jsonb, max_retries = %s,
                retry_delay_seconds = %s, notify_on_failure = %s,
                last_event_key = CASE WHEN %s THEN NULL ELSE last_event_key END,
                last_event_at = CASE WHEN %s THEN NULL ELSE last_event_at END,
                pending_event_key = CASE WHEN %s THEN NULL ELSE pending_event_key END,
                pending_event_input = CASE WHEN %s THEN NULL ELSE pending_event_input END,
                lease_until = CASE WHEN %s THEN NULL ELSE lease_until END,
                updated_at = now()
            WHERE user_id = %s AND id = %s
            """,
            (
                workflow_id,
                name[:256],
                trigger_type,
                schedule_expr,
                timezone,
                status,
                next_at,
                json.dumps(input_payload, ensure_ascii=False),
                max_retries,
                retry_delay_seconds,
                notify_on_failure,
                event_config_changed,
                event_config_changed,
                event_config_changed,
                event_config_changed,
                event_config_changed,
                user_id,
                automation_id,
            ),
        )
        conn.commit()
    return get_automation(user_id, automation_id)


def list_automation_runs(user_id: str, automation_id: int, limit: int = 50) -> list[dict[str, Any]]:
    _ensure_postgres()
    get_automation(user_id, automation_id)
    limit = max(1, min(limit, 200))
    with postgres_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, workflow_run_id, status, error_code, error_message, attempt,
                   trigger_source, created_at, completed_at
            FROM automation_run
            WHERE user_id = %s AND automation_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (user_id, automation_id, limit),
        ).fetchall()
    return [
        {
            "id": row[0],
            "workflowRunId": row[1],
            "status": row[2],
            "errorCode": row[3],
            "errorMessage": row[4],
            "attempt": row[5],
            "triggerSource": row[6],
            "createdAt": row[7].isoformat(),
            "completedAt": row[8].isoformat() if row[8] else None,
        }
        for row in rows
    ]


def _next_run_at(schedule_expr: str, timezone: str) -> object:
    try:
        return next_run_at(schedule_expr, timezone)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _row(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "id": row[0],
        "workflowId": row[1],
        "name": row[2],
        "triggerType": row[3],
        "scheduleExpr": row[4],
        "timezone": row[5],
        "status": row[6],
        "nextRunAt": row[7].isoformat() if row[7] else None,
        "lastRunAt": row[8].isoformat() if row[8] else None,
        "input": row[9] or {},
        "maxRetries": row[10],
        "retryDelaySeconds": row[11],
        "consecutiveFailures": row[12],
        "notifyOnFailure": row[13],
        "leaseUntil": row[14].isoformat() if row[14] else None,
        "lastEventKey": row[15],
        "lastEventAt": row[16].isoformat() if row[16] else None,
        "pendingEventKey": row[17],
        "pendingEventInput": row[18] or {},
        "createdAt": row[19].isoformat(),
        "updatedAt": row[20].isoformat(),
    }


def delete_automation(user_id: str, automation_id: int) -> dict[str, Any]:
    _ensure_postgres()
    with postgres_connection() as conn:
        conn.execute(
            """
            DELETE FROM agent_notification
            WHERE user_id = %s AND source_type IN ('automation', 'automation_event')
              AND source_id IN (
                SELECT id FROM automation_run WHERE user_id = %s AND automation_id = %s
              )
            """,
            (user_id, user_id, automation_id),
        )
        row = conn.execute(
            "DELETE FROM automation_rule WHERE user_id = %s AND id = %s RETURNING id",
            (user_id, automation_id),
        ).fetchone()
        conn.commit()
    if row is None:
        raise HTTPException(status_code=404, detail="自动化不存在。")
    return {"ok": True, "id": int(row[0])}


def _bounded_int(value: object, minimum: int, maximum: int, default: int) -> int:
    if value in (None, ""):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="自动化重试配置必须是整数。") from exc
    if parsed < minimum or parsed > maximum:
        raise HTTPException(status_code=400, detail=f"自动化重试配置必须在 {minimum} 到 {maximum} 之间。")
    return parsed


def _trigger_type(value: object) -> str:
    trigger_type = str(value or "schedule").strip().lower()
    if trigger_type not in TRIGGER_TYPES:
        raise HTTPException(status_code=400, detail="自动化触发类型无效。")
    return trigger_type


def _validate_trigger_input(user_id: str, trigger_type: str, payload: dict[str, Any]) -> None:
    if trigger_type == "project_stale":
        project_id = _bounded_int(payload.get("projectId"), 1, 2_147_483_647, 0)
        if not project_id:
            raise HTTPException(status_code=400, detail="项目停滞触发器必须选择项目。")
        get_project(user_id, project_id)
        _bounded_int(payload.get("staleHours"), 1, 8760, 72)
    elif trigger_type == "task_due":
        task_id = _bounded_int(payload.get("taskId"), 1, 2_147_483_647, 0)
        if not task_id:
            raise HTTPException(status_code=400, detail="任务到期触发器必须选择任务。")
        with postgres_connection() as conn:
            row = conn.execute("SELECT id FROM agent_todo WHERE user_id = %s AND id = %s", (user_id, task_id)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="任务不存在。")


def _ensure_postgres() -> None:
    if not settings.postgres_enabled:
        raise HTTPException(status_code=503, detail="PostgreSQL 未启用。")
