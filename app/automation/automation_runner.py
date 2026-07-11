from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import HTTPException

from app.automation.automation_store import get_automation
from app.automation.retry_policy import should_retry
from app.automation.schedule import next_run_at
from app.core.config import settings
from app.core.database import postgres_connection
from app.security.execution_audit import write_execution_audit
from app.workflows.workflow_runner import run_workflow


def run_automation(
    user_id: str,
    automation_id: int,
    override_input: dict[str, Any] | None = None,
    *,
    scheduled: bool = False,
) -> dict[str, Any]:
    if not settings.postgres_enabled:
        raise HTTPException(status_code=503, detail="PostgreSQL 未启用。")
    automation = get_automation(user_id, automation_id)
    payload_input = dict(automation.get("input") or {})
    if scheduled and automation.get("pendingEventInput"):
        payload_input.update(dict(automation["pendingEventInput"]))
    if override_input:
        payload_input.update(override_input)
    attempt = int(automation.get("consecutiveFailures") or 0) + 1 if scheduled else 1
    trigger_source = str(automation.get("triggerType") or "schedule") if scheduled else "manual"
    with postgres_connection() as conn:
        run_row = conn.execute(
            """
            INSERT INTO automation_run (user_id, automation_id, status, attempt, trigger_source)
            VALUES (%s, %s, 'running', %s, %s)
            RETURNING id
            """,
            (user_id, automation_id, attempt, trigger_source),
        ).fetchone()
        conn.commit()
    automation_run_id = int(run_row[0])
    try:
        workflow_run = run_workflow(
            user_id,
            int(automation["workflowId"]),
            {"triggerType": "automation", **payload_input},
        )
        workflow_status = str(workflow_run.get("status") or "failed")
        status = "completed" if workflow_status == "completed" else "failed"
        error_code = workflow_run.get("errorCode") or ("AUTOMATION_CONFIRMATION_REQUIRED" if workflow_status == "paused" else None)
        error_message = workflow_run.get("errorMessage") or ("自动化工作流需要人工确认。" if workflow_status == "paused" else None)
        retryable = workflow_status == "failed"
        will_retry, next_at = _finish_automation_run(
            user_id=user_id,
            automation=automation,
            automation_run_id=automation_run_id,
            workflow_run_id=workflow_run.get("id"),
            status=status,
            error_code=error_code,
            error_message=error_message,
            attempt=attempt,
            scheduled=scheduled,
            retryable=retryable,
        )
        write_execution_audit(
            user_id=user_id,
            source_type="automation",
            source_id=str(automation_run_id),
            executor_type="automation",
            executor_name=str(automation.get("name") or automation_id),
            operation="run",
            risk_level="read_only",
            status=status,
            input_summary=json.dumps(payload_input, ensure_ascii=False),
            output_summary=f"Automation run {status}. workflow_run_id={workflow_run.get('id')}",
            error_code=error_code,
            error_message=error_message,
        )
        return {
            "id": automation_run_id,
            "automationId": automation_id,
            "workflowRunId": workflow_run.get("id"),
            "status": status,
            "attempt": attempt,
            "willRetry": will_retry,
            "nextRunAt": next_at.isoformat() if next_at else automation.get("nextRunAt"),
            "workflowRun": workflow_run,
        }
    except Exception as exc:
        will_retry, next_at = _finish_automation_run(
            user_id=user_id,
            automation=automation,
            automation_run_id=automation_run_id,
            workflow_run_id=None,
            status="failed",
            error_code="AUTOMATION_RUN_FAILED",
            error_message=str(exc),
            attempt=attempt,
            scheduled=scheduled,
            retryable=True,
        )
        write_execution_audit(
            user_id=user_id,
            source_type="automation",
            source_id=str(automation_run_id),
            executor_type="automation",
            executor_name=str(automation.get("name") or automation_id),
            operation="run",
            risk_level="read_only",
            status="failed",
            input_summary=json.dumps(payload_input, ensure_ascii=False),
            error_code="AUTOMATION_RUN_FAILED",
            error_message=str(exc),
        )
        return {
            "id": automation_run_id,
            "automationId": automation_id,
            "workflowRunId": None,
            "status": "failed",
            "attempt": attempt,
            "willRetry": will_retry,
            "nextRunAt": next_at.isoformat() if next_at else automation.get("nextRunAt"),
            "errorCode": "AUTOMATION_RUN_FAILED",
            "errorMessage": str(exc),
        }


def due_automations(limit: int = 20) -> list[dict[str, Any]]:
    if not settings.postgres_enabled:
        return []
    with postgres_connection() as conn:
        rows = conn.execute(
            """
            WITH due AS (
              SELECT id
              FROM automation_rule
              WHERE status = 'enabled' AND next_run_at IS NOT NULL AND next_run_at <= now()
                AND (lease_until IS NULL OR lease_until <= now())
              ORDER BY next_run_at ASC
              LIMIT %s
              FOR UPDATE SKIP LOCKED
            )
            UPDATE automation_rule rule
            SET lease_until = now() + interval '5 minutes', updated_at = now()
            FROM due
            WHERE rule.id = due.id
            RETURNING rule.id, rule.user_id
            """,
            (max(1, min(limit, 100)),),
        ).fetchall()
        conn.commit()
    return [{"id": int(row[0]), "userId": row[1]} for row in rows]


def prime_event_automations(limit: int = 50) -> int:
    if not settings.postgres_enabled:
        return 0
    primed = 0
    with postgres_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, user_id, trigger_type, input, last_event_key
            FROM automation_rule
            WHERE status = 'enabled' AND trigger_type IN ('project_stale', 'task_due')
              AND pending_event_key IS NULL AND next_run_at IS NULL
              AND (lease_until IS NULL OR lease_until <= now())
            ORDER BY updated_at ASC
            LIMIT %s
            FOR UPDATE SKIP LOCKED
            """,
            (max(1, min(limit, 200)),),
        ).fetchall()
        for automation_id, user_id, trigger_type, input_payload, last_event_key in rows:
            event = _event_candidate(conn, str(user_id), str(trigger_type), input_payload or {})
            if not event or event["key"] == last_event_key:
                continue
            conn.execute(
                """
                UPDATE automation_rule
                SET pending_event_key = %s, pending_event_input = %s::jsonb,
                    next_run_at = now(), updated_at = now()
                WHERE id = %s AND user_id = %s
                """,
                (event["key"], json.dumps(event["input"], ensure_ascii=False), automation_id, user_id),
            )
            primed += 1
        conn.commit()
    return primed


def check_event_automation(user_id: str, automation_id: int) -> dict[str, Any]:
    automation = get_automation(user_id, automation_id)
    if automation.get("triggerType") not in {"project_stale", "task_due"}:
        raise HTTPException(status_code=400, detail="只有事件触发自动化可以检查触发条件。")
    if automation.get("status") != "enabled":
        raise HTTPException(status_code=409, detail="请先启用自动化，再检查触发条件。")
    with postgres_connection() as conn:
        conn.execute(
            "SELECT id FROM automation_rule WHERE user_id = %s AND id = %s FOR UPDATE",
            (user_id, automation_id),
        ).fetchone()
        current = get_automation(user_id, automation_id)
        if current.get("pendingEventKey"):
            conn.commit()
            return {"triggered": False, "reason": "事件已在等待执行或重试。"}
        event = _event_candidate(conn, user_id, str(current["triggerType"]), current.get("input") or {})
        if not event or event["key"] == current.get("lastEventKey"):
            conn.commit()
            return {"triggered": False, "reason": "当前没有新的匹配事件。"}
        conn.execute(
            """
            UPDATE automation_rule
            SET pending_event_key = %s, pending_event_input = %s::jsonb,
                next_run_at = now(), lease_until = now() + interval '5 minutes', updated_at = now()
            WHERE user_id = %s AND id = %s
            """,
            (event["key"], json.dumps(event["input"], ensure_ascii=False), user_id, automation_id),
        )
        conn.commit()
    result = run_automation(user_id, automation_id, scheduled=True)
    return {"triggered": True, "event": event["input"]["event"], "run": result}


def _finish_automation_run(
    *,
    user_id: str,
    automation: dict[str, Any],
    automation_run_id: int,
    workflow_run_id: object,
    status: str,
    error_code: object,
    error_message: object,
    attempt: int,
    scheduled: bool,
    retryable: bool,
) -> tuple[bool, object | None]:
    max_retries = int(automation.get("maxRetries") or 0)
    is_event = automation.get("triggerType") in {"project_stale", "task_due"}
    will_retry = scheduled and status == "failed" and should_retry(attempt, max_retries, retryable=retryable)
    if scheduled and will_retry:
        next_at = datetime.now(UTC) + timedelta(seconds=int(automation.get("retryDelaySeconds") or 60))
        failures = attempt
    elif scheduled and not is_event:
        next_at = _next_run_at(
            str(automation.get("scheduleExpr") or "daily"),
            str(automation.get("timezone") or "Asia/Shanghai"),
        )
        failures = 0
    elif scheduled:
        next_at = None
        failures = 0
    else:
        next_at = None
        failures = int(automation.get("consecutiveFailures") or 0)
    with postgres_connection() as conn:
        conn.execute(
            """
            UPDATE automation_run
            SET workflow_run_id = %s, status = %s, error_code = %s,
                error_message = %s, completed_at = now()
            WHERE user_id = %s AND id = %s
            """,
            (workflow_run_id, status, error_code, error_message, user_id, automation_run_id),
        )
        if scheduled and is_event and not will_retry:
            conn.execute(
                """
                UPDATE automation_rule
                SET last_run_at = now(), next_run_at = NULL, consecutive_failures = 0,
                    lease_until = NULL, last_event_key = pending_event_key, last_event_at = now(),
                    pending_event_key = NULL, pending_event_input = NULL, updated_at = now()
                WHERE user_id = %s AND id = %s
                """,
                (user_id, automation["id"]),
            )
        elif scheduled:
            conn.execute(
                """
                UPDATE automation_rule
                SET last_run_at = now(), next_run_at = %s, consecutive_failures = %s,
                    lease_until = NULL, updated_at = now()
                WHERE user_id = %s AND id = %s
                """,
                (next_at, failures, user_id, automation["id"]),
            )
        else:
            conn.execute(
                "UPDATE automation_rule SET last_run_at = now(), updated_at = now() WHERE user_id = %s AND id = %s",
                (user_id, automation["id"]),
            )
        conn.commit()
    if status == "completed" and is_event and scheduled:
        _create_event_notification(user_id, automation_run_id, automation)
    if status == "failed" and not will_retry and bool(automation.get("notifyOnFailure", True)):
        _create_failure_notification(
            user_id,
            automation_run_id,
            str(automation.get("name") or automation["id"]),
            attempt,
            str(error_message or error_code or "未知错误"),
        )
    return will_retry, next_at


def _create_failure_notification(user_id: str, run_id: int, name: str, attempt: int, message: str) -> None:
    with postgres_connection() as conn:
        conn.execute(
            """
            INSERT INTO agent_notification (user_id, source_type, source_id, title, message)
            VALUES (%s, 'automation', %s, %s, %s)
            ON CONFLICT (user_id, source_type, source_id) DO NOTHING
            """,
            (user_id, run_id, f"自动化执行失败：{name}"[:256], f"第 {attempt} 次执行失败：{message}"),
        )
        conn.commit()


def _create_event_notification(user_id: str, run_id: int, automation: dict[str, Any]) -> None:
    event = dict(automation.get("pendingEventInput") or {}).get("event") or {}
    title = str(event.get("title") or automation.get("name") or "自动化事件")[:256]
    message = str(event.get("message") or "自动化事件已触发并执行完成。")
    with postgres_connection() as conn:
        conn.execute(
            """
            INSERT INTO agent_notification (user_id, source_type, source_id, title, message)
            VALUES (%s, 'automation_event', %s, %s, %s)
            ON CONFLICT (user_id, source_type, source_id) DO NOTHING
            """,
            (user_id, run_id, title, message),
        )
        conn.commit()


def _event_candidate(conn: object, user_id: str, trigger_type: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    if trigger_type == "project_stale":
        project_id = int(payload.get("projectId") or 0)
        stale_hours = max(1, min(int(payload.get("staleHours") or 72), 8760))
        row = conn.execute(
            """
            SELECT id, name, COALESCE(last_activity_at, updated_at)
            FROM project
            WHERE user_id = %s AND id = %s AND status = 'active'
            """,
            (user_id, project_id),
        ).fetchone()
        if row is None or row[2] > datetime.now(UTC) - timedelta(hours=stale_hours):
            return None
        activity_at = row[2].isoformat()
        return {
            "key": f"project_stale:{row[0]}:{activity_at}:{stale_hours}",
            "input": {
                "event": {
                    "type": "project_stale",
                    "projectId": row[0],
                    "title": f"项目停滞提醒：{row[1]}",
                    "message": f"项目已超过 {stale_hours} 小时没有活动。",
                    "activityAt": activity_at,
                }
            },
        }
    if trigger_type == "task_due":
        task_id = int(payload.get("taskId") or 0)
        row = conn.execute(
            """
            SELECT id, title, due_at
            FROM agent_todo
            WHERE user_id = %s AND id = %s AND status = 'open' AND due_at IS NOT NULL AND due_at <= now()
            """,
            (user_id, task_id),
        ).fetchone()
        if row is None:
            return None
        due_at = row[2].isoformat()
        return {
            "key": f"task_due:{row[0]}:{due_at}",
            "input": {
                "event": {
                    "type": "task_due",
                    "taskId": row[0],
                    "title": f"任务到期提醒：{row[1]}",
                    "message": f"任务已到截止时间：{row[1]}",
                    "dueAt": due_at,
                }
            },
        }
    return None


def _next_run_at(schedule_expr: str, timezone: str) -> object:
    return next_run_at(schedule_expr, timezone)
