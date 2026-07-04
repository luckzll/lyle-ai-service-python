from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import HTTPException

from app.core.config import settings
from app.core.database import postgres_connection
from app.projects.project_store import get_project


def list_goals(user_id: str, project_id: int | None = None, status: str | None = "active", limit: int = 100) -> list[dict[str, Any]]:
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
            SELECT id, project_id, title, description, horizon, status, progress,
                   due_at, created_at, updated_at
            FROM goal
            WHERE {' AND '.join(clauses)}
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            tuple(params),
        ).fetchall()
    return [_goal_row(row) for row in rows]


def create_goal(user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_postgres()
    project_id = _optional_int(payload.get("projectId") or payload.get("project_id"))
    if project_id is not None:
        get_project(user_id, project_id)
    title = _required_text(payload.get("title"), "目标标题不能为空。")[:256]
    description = _optional_text(payload.get("description"))
    horizon = _status(payload.get("horizon"), {"day", "week", "month", "quarter", "long_term"}, "week")
    status = _status(payload.get("status"), {"active", "completed", "paused", "cancelled"}, "active")
    progress = min(1.0, max(0.0, _float(payload.get("progress"), 0.0)))
    due_at = _parse_datetime(payload.get("dueAt") or payload.get("due_at"))
    with postgres_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO goal (user_id, project_id, title, description, horizon, status, progress, due_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id, project_id, title, description, horizon, status, progress,
                      due_at, created_at, updated_at
            """,
            (user_id, project_id, title, description, horizon, status, progress, due_at),
        ).fetchone()
        if project_id is not None:
            conn.execute("UPDATE project SET last_activity_at = now(), updated_at = now() WHERE id = %s", (project_id,))
        conn.commit()
    return _goal_with_milestones(user_id, _goal_row(row))


def get_goal(user_id: str, goal_id: int) -> dict[str, Any]:
    _ensure_postgres()
    with postgres_connection() as conn:
        row = conn.execute(
            """
            SELECT id, project_id, title, description, horizon, status, progress,
                   due_at, created_at, updated_at
            FROM goal
            WHERE user_id = %s AND id = %s
            """,
            (user_id, goal_id),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="目标不存在。")
    return _goal_with_milestones(user_id, _goal_row(row))


def update_goal(user_id: str, goal_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    current = get_goal(user_id, goal_id)
    title = _optional_text(payload.get("title")) or current["title"]
    description = _optional_text(payload.get("description")) if "description" in payload else current.get("description")
    horizon = _status(payload.get("horizon"), {"day", "week", "month", "quarter", "long_term"}, current["horizon"])
    status = _status(payload.get("status"), {"active", "completed", "paused", "cancelled"}, current["status"])
    progress = min(1.0, max(0.0, _float(payload.get("progress"), float(current.get("progress") or 0.0))))
    due_at = (
        _parse_datetime(payload.get("dueAt") or payload.get("due_at"))
        if "dueAt" in payload or "due_at" in payload
        else current.get("dueAt")
    )
    with postgres_connection() as conn:
        row = conn.execute(
            """
            UPDATE goal
            SET title = %s,
                description = %s,
                horizon = %s,
                status = %s,
                progress = %s,
                due_at = %s,
                updated_at = now()
            WHERE user_id = %s AND id = %s
            RETURNING id, project_id, title, description, horizon, status, progress,
                      due_at, created_at, updated_at
            """,
            (title, description, horizon, status, progress, due_at, user_id, goal_id),
        ).fetchone()
        conn.commit()
    if row is None:
        raise HTTPException(status_code=404, detail="目标不存在。")
    return _goal_with_milestones(user_id, _goal_row(row))


def create_milestone(user_id: str, goal_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_postgres()
    _ensure_goal_exists(user_id, goal_id)
    title = _required_text(payload.get("title"), "里程碑标题不能为空。")[:256]
    description = _optional_text(payload.get("description"))
    status = _status(payload.get("status"), {"pending", "active", "completed", "skipped"}, "pending")
    sort_order = _int(payload.get("sortOrder") or payload.get("sort_order"), 0)
    due_at = _parse_datetime(payload.get("dueAt") or payload.get("due_at"))
    with postgres_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO milestone (user_id, goal_id, title, description, status, sort_order, due_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id, goal_id, title, description, status, sort_order, due_at, created_at, updated_at
            """,
            (user_id, goal_id, title, description, status, sort_order, due_at),
        ).fetchone()
        conn.commit()
    return _milestone_row(row)


def list_milestones(user_id: str, goal_id: int) -> list[dict[str, Any]]:
    _ensure_postgres()
    _ensure_goal_exists(user_id, goal_id)
    with postgres_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, goal_id, title, description, status, sort_order, due_at, created_at, updated_at
            FROM milestone
            WHERE user_id = %s AND goal_id = %s
            ORDER BY sort_order ASC, created_at ASC
            """,
            (user_id, goal_id),
        ).fetchall()
    return [_milestone_row(row) for row in rows]


def link_task_to_goal(user_id: str, goal_id: int, task_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    _ensure_postgres()
    _ensure_goal_exists(user_id, goal_id)
    payload = payload or {}
    milestone_id = _optional_int(payload.get("milestoneId") or payload.get("milestone_id"))
    relation = (_optional_text(payload.get("relation")) or "contributes_to")[:64]
    with postgres_connection() as conn:
        if milestone_id is not None:
            milestone = conn.execute(
                "SELECT id FROM milestone WHERE user_id = %s AND goal_id = %s AND id = %s",
                (user_id, goal_id, milestone_id),
            ).fetchone()
            if milestone is None:
                raise HTTPException(status_code=404, detail="里程碑不存在。")
        row = conn.execute(
            """
            INSERT INTO goal_task_link (user_id, goal_id, milestone_id, task_id, relation)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT(goal_id, task_id) DO UPDATE SET
              milestone_id = EXCLUDED.milestone_id,
              relation = EXCLUDED.relation
            RETURNING id, goal_id, milestone_id, task_id, relation, created_at
            """,
            (user_id, goal_id, milestone_id, str(task_id), relation),
        ).fetchone()
        conn.commit()
    return {
        "id": row[0],
        "goalId": row[1],
        "milestoneId": row[2],
        "taskId": row[3],
        "relation": row[4],
        "createdAt": _iso(row[5]),
    }


def _goal_with_milestones(user_id: str, goal: dict[str, Any]) -> dict[str, Any]:
    return {**goal, "milestones": list_milestones(user_id, int(goal["id"]))}


def _ensure_goal_exists(user_id: str, goal_id: int) -> None:
    with postgres_connection() as conn:
        row = conn.execute(
            "SELECT id FROM goal WHERE user_id = %s AND id = %s",
            (user_id, goal_id),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="目标不存在。")


def _goal_row(row: Any) -> dict[str, Any]:
    return {
        "id": row[0],
        "projectId": row[1],
        "title": row[2],
        "description": row[3],
        "horizon": row[4],
        "status": row[5],
        "progress": row[6],
        "dueAt": _iso(row[7]),
        "createdAt": _iso(row[8]),
        "updatedAt": _iso(row[9]),
    }


def _milestone_row(row: Any) -> dict[str, Any]:
    return {
        "id": row[0],
        "goalId": row[1],
        "title": row[2],
        "description": row[3],
        "status": row[4],
        "sortOrder": row[5],
        "dueAt": _iso(row[6]),
        "createdAt": _iso(row[7]),
        "updatedAt": _iso(row[8]),
    }


def _ensure_postgres() -> None:
    if not settings.postgres_enabled:
        raise HTTPException(status_code=503, detail="PostgreSQL 未启用，目标系统不可用。")


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


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _status(value: Any, allowed: set[str], default: str) -> str:
    status = str(value or default).strip()
    if status not in allowed:
        raise HTTPException(status_code=400, detail=f"不支持的状态：{status}")
    return status


def _parse_datetime(value: Any) -> datetime | str | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    text = str(value)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text


def _iso(value: Any) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else value
