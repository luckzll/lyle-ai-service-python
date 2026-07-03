from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.database import postgres_connection
from app.tools.notification_service import list_notifications, mark_notification_read
from app.tools.operation_confirmation import confirm_pending_operation, list_pending_operations, reject_pending_operation
from app.tools.task_tool import create_todo
from app.tools.tool_registry import list_available_tools

router = APIRouter()


class TodoStatusRequest(BaseModel):
    status: str = Field(pattern="^(open|done|cancelled)$")


class TodoCreateRequest(BaseModel):
    title: str
    dueAt: datetime | None = None
    conversationId: str | None = None


class TodoDueAtRequest(BaseModel):
    dueAt: datetime | None = None


class CalendarStatusRequest(BaseModel):
    status: str = Field(pattern="^(scheduled|done|cancelled)$")


@router.get("/tools")
def tools() -> dict[str, object]:
    return {"tools": list_available_tools()}


@router.get("/tools/audit")
def tool_audit(limit: int = Query(default=50, ge=1, le=200)) -> dict[str, object]:
    _require_postgres()
    with postgres_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, tool_name, user_id, conversation_id, ok, elapsed_ms, payload, created_at
            FROM tool_audit_log
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,),
        ).fetchall()
    return {
        "items": [
            {
                "id": row[0],
                "toolName": row[1],
                "userId": row[2],
                "conversationId": row[3],
                "ok": row[4],
                "elapsedMs": row[5],
                "payload": row[6],
                "createdAt": row[7].isoformat(),
            }
            for row in rows
        ]
    }


@router.get("/tools/operations/{user_id}")
def pending_operations(user_id: str, limit: int = Query(default=50, ge=1, le=200)) -> dict[str, object]:
    _require_postgres()
    return list_pending_operations(user_id, limit)


@router.post("/tools/operations/{user_id}/{operation_id}/confirm")
def confirm_operation(user_id: str, operation_id: int) -> dict[str, object]:
    _require_postgres()
    try:
        return confirm_pending_operation(user_id, operation_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/tools/operations/{user_id}/{operation_id}/reject")
def reject_operation(user_id: str, operation_id: int) -> dict[str, object]:
    _require_postgres()
    try:
        return reject_pending_operation(user_id, operation_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/tools/todos/{user_id}")
def todos(user_id: str, limit: int = Query(default=50, ge=1, le=200)) -> dict[str, object]:
    _require_postgres()
    with postgres_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, title, status, due_at, conversation_id, created_at, updated_at
            FROM agent_todo
            WHERE user_id = %s
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            (user_id, limit),
        ).fetchall()
    return {
        "items": [
            {
                "id": row[0],
                "title": row[1],
                "status": row[2],
                "dueAt": row[3].isoformat() if row[3] else None,
                "conversationId": row[4],
                "createdAt": row[5].isoformat(),
                "updatedAt": row[6].isoformat(),
            }
            for row in rows
        ]
    }


@router.post("/tools/todos/{user_id}")
def create_todo_api(user_id: str, request: TodoCreateRequest) -> dict[str, object]:
    result = create_todo(user_id, request.conversationId, request.title, request.dueAt)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=str(result.get("content") or "Todo create failed."))
    return result["todo"]  # type: ignore[return-value]


@router.patch("/tools/todos/{user_id}/{todo_id}/status")
def update_todo_status(user_id: str, todo_id: int, request: TodoStatusRequest) -> dict[str, object]:
    _require_postgres()
    with postgres_connection() as conn:
        row = conn.execute(
            """
            UPDATE agent_todo
            SET status = %s, updated_at = %s
            WHERE user_id = %s AND id = %s
            RETURNING id, title, status, updated_at
            """,
            (request.status, datetime.now(UTC), user_id, todo_id),
        ).fetchone()
        conn.commit()
    if row is None:
        raise HTTPException(status_code=404, detail="Todo not found.")
    return {
        "id": row[0],
        "title": row[1],
        "status": row[2],
        "updatedAt": row[3].isoformat(),
    }


@router.patch("/tools/todos/{user_id}/{todo_id}/due-at")
def update_todo_due_at(user_id: str, todo_id: int, request: TodoDueAtRequest) -> dict[str, object]:
    _require_postgres()
    with postgres_connection() as conn:
        row = conn.execute(
            """
            UPDATE agent_todo
            SET due_at = %s, updated_at = %s
            WHERE user_id = %s AND id = %s
            RETURNING id, title, status, due_at, conversation_id, created_at, updated_at
            """,
            (request.dueAt, datetime.now(UTC), user_id, todo_id),
        ).fetchone()
        conn.commit()
    if row is None:
        raise HTTPException(status_code=404, detail="Todo not found.")
    return {
        "id": row[0],
        "title": row[1],
        "status": row[2],
        "dueAt": row[3].isoformat() if row[3] else None,
        "conversationId": row[4],
        "createdAt": row[5].isoformat(),
        "updatedAt": row[6].isoformat(),
    }


@router.delete("/tools/todos/{user_id}/{todo_id}")
def delete_todo(user_id: str, todo_id: int) -> dict[str, object]:
    _require_postgres()
    with postgres_connection() as conn:
        row = conn.execute(
            "DELETE FROM agent_todo WHERE user_id = %s AND id = %s RETURNING id",
            (user_id, todo_id),
        ).fetchone()
        conn.commit()
    if row is None:
        raise HTTPException(status_code=404, detail="Todo not found.")
    return {"deleted": True, "id": todo_id}


@router.get("/tools/calendar/{user_id}")
def calendar_events(user_id: str, limit: int = Query(default=50, ge=1, le=200)) -> dict[str, object]:
    _require_postgres()
    with postgres_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, title, starts_at, status, conversation_id, created_at, updated_at
            FROM agent_calendar_event
            WHERE user_id = %s
            ORDER BY COALESCE(starts_at, updated_at) DESC
            LIMIT %s
            """,
            (user_id, limit),
        ).fetchall()
    return {
        "items": [
            {
                "id": row[0],
                "title": row[1],
                "startsAt": row[2].isoformat() if row[2] else None,
                "status": row[3],
                "conversationId": row[4],
                "createdAt": row[5].isoformat(),
                "updatedAt": row[6].isoformat(),
            }
            for row in rows
        ]
    }


@router.patch("/tools/calendar/{user_id}/{event_id}/status")
def update_calendar_status(user_id: str, event_id: int, request: CalendarStatusRequest) -> dict[str, object]:
    _require_postgres()
    with postgres_connection() as conn:
        row = conn.execute(
            """
            UPDATE agent_calendar_event
            SET status = %s, updated_at = %s
            WHERE user_id = %s AND id = %s
            RETURNING id, title, starts_at, status, updated_at
            """,
            (request.status, datetime.now(UTC), user_id, event_id),
        ).fetchone()
        conn.commit()
    if row is None:
        raise HTTPException(status_code=404, detail="Calendar event not found.")
    return {
        "id": row[0],
        "title": row[1],
        "startsAt": row[2].isoformat() if row[2] else None,
        "status": row[3],
        "updatedAt": row[4].isoformat(),
    }


@router.delete("/tools/calendar/{user_id}/{event_id}")
def delete_calendar_event(user_id: str, event_id: int) -> dict[str, object]:
    _require_postgres()
    with postgres_connection() as conn:
        row = conn.execute(
            "DELETE FROM agent_calendar_event WHERE user_id = %s AND id = %s RETURNING id",
            (user_id, event_id),
        ).fetchone()
        conn.commit()
    if row is None:
        raise HTTPException(status_code=404, detail="Calendar event not found.")
    return {"deleted": True, "id": event_id}


@router.get("/tools/reminders/{user_id}")
def reminders(user_id: str, limit: int = Query(default=50, ge=1, le=200)) -> dict[str, object]:
    _require_postgres()
    return list_notifications(user_id, limit)


@router.patch("/tools/reminders/{user_id}/{notification_id}/read")
def mark_reminder_read(user_id: str, notification_id: int) -> dict[str, object]:
    _require_postgres()
    result = mark_notification_read(user_id, notification_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Reminder not found.")
    return result


def _require_postgres() -> None:
    if not settings.postgres_enabled:
        raise HTTPException(status_code=503, detail="PostgreSQL is not enabled.")
