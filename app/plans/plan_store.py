from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from app.core.config import settings
from app.core.database import postgres_connection
from app.projects.project_store import get_project


PLAN_STATUSES = {"draft", "awaiting_confirmation", "running", "paused", "completed", "failed", "cancelled"}
STEP_STATUSES = {"pending", "running", "awaiting_user", "completed", "failed", "skipped"}


def create_plan(user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_postgres()
    project_id = _optional_int(payload.get("projectId") or payload.get("project_id"))
    if project_id is not None:
        get_project(user_id, project_id)
    title = _required_text(payload.get("title"), "计划标题不能为空。")[:256]
    objective = _required_text(payload.get("objective"), "计划目标不能为空。")
    conversation_id = _optional_text(payload.get("conversationId") or payload.get("conversation_id"))
    status = _status(payload.get("status"), PLAN_STATUSES, "awaiting_confirmation")
    requires_confirmation = bool(payload.get("requiresConfirmation", True))
    steps = payload.get("steps") if isinstance(payload.get("steps"), list) else []
    with postgres_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO agent_plan
              (user_id, project_id, conversation_id, title, objective, status, requires_confirmation)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id, project_id, conversation_id, title, objective, status,
                      requires_confirmation, created_at, updated_at, completed_at
            """,
            (user_id, project_id, conversation_id, title, objective, status, requires_confirmation),
        ).fetchone()
        plan_id = row[0]
        for index, raw_step in enumerate(steps):
            if not isinstance(raw_step, dict):
                continue
            step_title = _optional_text(raw_step.get("title"))
            if not step_title:
                continue
            conn.execute(
                """
                INSERT INTO agent_plan_step
                  (user_id, plan_id, sort_order, title, description, status, tool_name)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    user_id,
                    plan_id,
                    _int(raw_step.get("sortOrder") or raw_step.get("sort_order"), index),
                    step_title[:256],
                    _optional_text(raw_step.get("description")),
                    _status(raw_step.get("status"), STEP_STATUSES, "pending"),
                    _optional_text(raw_step.get("toolName") or raw_step.get("tool_name")),
                ),
            )
        conn.commit()
    return get_plan(user_id, int(plan_id))


def get_plan(user_id: str, plan_id: int) -> dict[str, Any]:
    _ensure_postgres()
    with postgres_connection() as conn:
        row = conn.execute(
            """
            SELECT id, project_id, conversation_id, title, objective, status,
                   requires_confirmation, created_at, updated_at, completed_at
            FROM agent_plan
            WHERE user_id = %s AND id = %s
            """,
            (user_id, plan_id),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="计划不存在。")
        steps = conn.execute(
            """
            SELECT id, plan_id, sort_order, title, description, status, tool_name,
                   output_ref, error_code, error_message, started_at, completed_at, created_at, updated_at
            FROM agent_plan_step
            WHERE user_id = %s AND plan_id = %s
            ORDER BY sort_order ASC, id ASC
            """,
            (user_id, plan_id),
        ).fetchall()
    return {**_plan_row(row), "steps": [_step_row(step) for step in steps]}


def list_plans(user_id: str, project_id: int | None = None, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    _ensure_postgres()
    limit = max(1, min(limit, 200))
    clauses = ["user_id = %s"]
    params: list[Any] = [user_id]
    if project_id is not None:
        get_project(user_id, project_id)
        clauses.append("project_id = %s")
        params.append(project_id)
    if status:
        clauses.append("status = %s")
        params.append(status)
    params.append(limit)
    with postgres_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT id, project_id, conversation_id, title, objective, status,
                   requires_confirmation, created_at, updated_at, completed_at
            FROM agent_plan
            WHERE {' AND '.join(clauses)}
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            tuple(params),
        ).fetchall()
        plans = [_plan_row(row) for row in rows]
        if not plans:
            return []
        plan_ids = [plan["id"] for plan in plans]
        step_rows = conn.execute(
            """
            SELECT id, plan_id, sort_order, title, description, status, tool_name,
                   output_ref, error_code, error_message, started_at, completed_at, created_at, updated_at
            FROM agent_plan_step
            WHERE user_id = %s AND plan_id = ANY(%s::bigint[])
            ORDER BY sort_order ASC, id ASC
            """,
            (user_id, plan_ids),
        ).fetchall()
    steps_by_plan: dict[int, list[dict[str, Any]]] = {}
    for step in step_rows:
        steps_by_plan.setdefault(int(step[1]), []).append(_step_row(step))
    return [{**plan, "steps": steps_by_plan.get(int(plan["id"]), [])} for plan in plans]


def transition_plan(user_id: str, plan_id: int, action: str) -> dict[str, Any]:
    current = get_plan(user_id, plan_id)
    next_status = _next_status(current["status"], action)
    completed_expr = "now()" if next_status == "completed" else "completed_at"
    with postgres_connection() as conn:
        conn.execute(
            f"UPDATE agent_plan SET status = %s, updated_at = now(), completed_at = {completed_expr} WHERE user_id = %s AND id = %s",
            (next_status, user_id, plan_id),
        )
        conn.commit()
    if action in {"confirm", "resume"} and next_status == "running":
        return run_plan(user_id, plan_id, resume_waiting=action == "resume")
    return get_plan(user_id, plan_id)


def run_plan(user_id: str, plan_id: int, resume_waiting: bool = False) -> dict[str, Any]:
    """Advance low-risk plan steps synchronously until completion or user input is needed."""
    plan = get_plan(user_id, plan_id)
    if plan["status"] != "running":
        return plan
    with postgres_connection() as conn:
        if resume_waiting:
            conn.execute(
                """
                UPDATE agent_plan_step
                SET status = 'completed',
                    output_ref = COALESCE(output_ref, '用户已确认继续执行。'),
                    error_code = NULL,
                    error_message = NULL,
                    completed_at = COALESCE(completed_at, now()),
                    updated_at = now()
                WHERE user_id = %s AND plan_id = %s AND status = 'awaiting_user'
                """,
                (user_id, plan_id),
            )
        while True:
            step = conn.execute(
                """
                SELECT id, title, description, status, tool_name
                FROM agent_plan_step
                WHERE user_id = %s AND plan_id = %s AND status IN ('pending', 'awaiting_user')
                ORDER BY sort_order ASC, id ASC
                LIMIT 1
                """,
                (user_id, plan_id),
            ).fetchone()
            if step is None:
                conn.execute(
                    """
                    UPDATE agent_plan
                    SET status = 'completed', completed_at = now(), updated_at = now()
                    WHERE user_id = %s AND id = %s
                    """,
                    (user_id, plan_id),
                )
                conn.commit()
                return get_plan(user_id, plan_id)
            step_id, title, description, status, tool_name = step
            if status == "awaiting_user" or _step_requires_user(str(title), description, tool_name):
                conn.execute(
                    """
                    UPDATE agent_plan_step
                    SET status = 'awaiting_user',
                        error_code = 'USER_CONFIRMATION_REQUIRED',
                        error_message = '该步骤涉及写入、外部变更或需要用户确认，计划已暂停。',
                        started_at = COALESCE(started_at, now()),
                        updated_at = now()
                    WHERE user_id = %s AND id = %s
                    """,
                    (user_id, step_id),
                )
                conn.execute(
                    """
                    UPDATE agent_plan
                    SET status = 'paused', updated_at = now()
                    WHERE user_id = %s AND id = %s
                    """,
                    (user_id, plan_id),
                )
                conn.commit()
                return get_plan(user_id, plan_id)
            conn.execute(
                """
                UPDATE agent_plan_step
                SET status = 'running',
                    started_at = COALESCE(started_at, now()),
                    updated_at = now()
                WHERE user_id = %s AND id = %s
                """,
                (user_id, step_id),
            )
            conn.execute(
                """
                UPDATE agent_plan_step
                SET status = 'completed',
                    output_ref = %s,
                    completed_at = now(),
                    updated_at = now()
                WHERE user_id = %s AND id = %s
                """,
                (_step_output(str(title), description, tool_name), user_id, step_id),
            )
            conn.execute(
                "UPDATE agent_plan SET updated_at = now() WHERE user_id = %s AND id = %s",
                (user_id, plan_id),
            )


def _step_requires_user(title: str, description: str | None, tool_name: str | None) -> bool:
    text = f"{title} {description or ''} {tool_name or ''}".lower()
    return any(keyword in text for keyword in ["确认", "写入", "create", "update", "delete", "todo.", "external"])


def _step_output(title: str, description: str | None, tool_name: str | None) -> str:
    if tool_name:
        return f"已完成步骤：{title}。工具：{tool_name}。"
    if description:
        return f"已完成步骤：{title}。摘要：{description}"
    return f"已完成步骤：{title}。"


def _next_status(status: str, action: str) -> str:
    transitions = {
        "confirm": {"draft": "running", "awaiting_confirmation": "running"},
        "pause": {"running": "paused"},
        "resume": {"paused": "running"},
        "cancel": {"draft": "cancelled", "awaiting_confirmation": "cancelled", "running": "cancelled", "paused": "cancelled"},
        "complete": {"running": "completed", "paused": "completed"},
    }
    next_status = transitions.get(action, {}).get(status)
    if not next_status:
        raise HTTPException(status_code=409, detail=f"计划状态 {status} 不支持操作 {action}。")
    return next_status


def _plan_row(row: Any) -> dict[str, Any]:
    return {
        "id": row[0],
        "projectId": row[1],
        "conversationId": row[2],
        "title": row[3],
        "objective": row[4],
        "status": row[5],
        "requiresConfirmation": row[6],
        "createdAt": _iso(row[7]),
        "updatedAt": _iso(row[8]),
        "completedAt": _iso(row[9]),
    }


def _step_row(row: Any) -> dict[str, Any]:
    return {
        "id": row[0],
        "planId": row[1],
        "sortOrder": row[2],
        "title": row[3],
        "description": row[4],
        "status": row[5],
        "toolName": row[6],
        "outputRef": row[7],
        "errorCode": row[8],
        "errorMessage": row[9],
        "startedAt": _iso(row[10]),
        "completedAt": _iso(row[11]),
        "createdAt": _iso(row[12]),
        "updatedAt": _iso(row[13]),
    }


def _ensure_postgres() -> None:
    if not settings.postgres_enabled:
        raise HTTPException(status_code=503, detail="PostgreSQL 未启用，计划系统不可用。")


def _required_text(value: Any, message: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail=message)
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="ID 格式不正确。") from exc


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _status(value: Any, allowed: set[str], default: str) -> str:
    status = str(value or default).strip()
    if status not in allowed:
        raise HTTPException(status_code=400, detail=f"不支持的状态：{status}")
    return status


def _iso(value: Any) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else value
