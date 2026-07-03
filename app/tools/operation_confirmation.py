import json
from datetime import UTC, datetime
from typing import Any

from app.core.config import settings
from app.core.database import postgres_connection
from app.tools.tool_audit import audit_tool_call


def create_pending_operation(
    *,
    user_id: str,
    conversation_id: str | None,
    tool_name: str,
    operation_type: str,
    summary: str,
    payload: dict[str, Any],
    source_text: str,
    risk_level: str = "write",
) -> dict[str, object]:
    if not settings.postgres_enabled:
        return {
            "ok": False,
            "content": "PostgreSQL 未启用，无法生成待确认操作。",
        }

    now = datetime.now(UTC)
    with postgres_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO agent_pending_operation
              (user_id, conversation_id, tool_name, operation_type, risk_level,
               summary, payload, status, source_text, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, 'pending', %s, %s, %s)
            RETURNING id, created_at, updated_at
            """,
            (
                user_id,
                conversation_id or "",
                tool_name,
                operation_type,
                risk_level,
                summary,
                json.dumps(payload, ensure_ascii=False),
                source_text[:1000],
                now,
                now,
            ),
        ).fetchone()
        conn.commit()

    operation = {
        "id": row[0],
        "toolName": tool_name,
        "operationType": operation_type,
        "riskLevel": risk_level,
        "summary": summary,
        "payload": payload,
        "status": "pending",
        "sourceText": source_text[:1000],
        "createdAt": row[1].isoformat(),
        "updatedAt": row[2].isoformat(),
    }
    return {
        "ok": True,
        "content": f"已生成待确认操作：{summary}。用户确认后才会执行。",
        "requiresConfirmation": True,
        "operation": operation,
    }


def list_pending_operations(user_id: str, limit: int = 50) -> dict[str, object]:
    if not settings.postgres_enabled:
        return {"items": []}
    with postgres_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, tool_name, operation_type, risk_level, summary, payload,
                   status, result, source_text, created_at, updated_at,
                   confirmed_at, rejected_at, conversation_id
            FROM agent_pending_operation
            WHERE user_id = %s
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            (user_id, limit),
        ).fetchall()
    return {"items": [_operation_row(row) for row in rows]}


def confirm_pending_operation(user_id: str, operation_id: int) -> dict[str, object]:
    operation = _get_pending_operation(user_id, operation_id)
    result = _execute_operation(user_id, operation)
    now = datetime.now(UTC)
    with postgres_connection() as conn:
        row = conn.execute(
            """
            UPDATE agent_pending_operation
            SET status = 'confirmed', result = %s::jsonb, updated_at = %s, confirmed_at = %s
            WHERE user_id = %s AND id = %s AND status = 'pending'
            RETURNING id, tool_name, operation_type, risk_level, summary, payload,
                      status, result, source_text, created_at, updated_at,
                      confirmed_at, rejected_at, conversation_id
            """,
            (
                json.dumps(result, ensure_ascii=False),
                now,
                now,
                user_id,
                operation_id,
            ),
        ).fetchone()
        conn.commit()
    if row is None:
        raise ValueError("待确认操作不存在或已处理。")

    audit_tool_call(
        tool_name=f"{operation['toolName']}.confirm",
        user_id=user_id,
        conversation_id=str(operation.get("conversationId") or ""),
        question=str(operation.get("sourceText") or ""),
        ok=bool(result.get("ok")),
        content=str(result.get("content") or ""),
        elapsed_ms=0,
    )
    return {"operation": _operation_row(row), "result": result}


def reject_pending_operation(user_id: str, operation_id: int) -> dict[str, object]:
    now = datetime.now(UTC)
    with postgres_connection() as conn:
        row = conn.execute(
            """
            UPDATE agent_pending_operation
            SET status = 'rejected', updated_at = %s, rejected_at = %s
            WHERE user_id = %s AND id = %s AND status = 'pending'
            RETURNING id, tool_name, operation_type, risk_level, summary, payload,
                      status, result, source_text, created_at, updated_at,
                      confirmed_at, rejected_at, conversation_id
            """,
            (now, now, user_id, operation_id),
        ).fetchone()
        conn.commit()
    if row is None:
        raise ValueError("待确认操作不存在或已处理。")
    operation = _operation_row(row)
    audit_tool_call(
        tool_name=f"{operation['toolName']}.reject",
        user_id=user_id,
        conversation_id=str(operation.get("conversationId") or ""),
        question=str(operation.get("sourceText") or ""),
        ok=True,
        content=f"用户拒绝执行：{operation.get('summary')}",
        elapsed_ms=0,
    )
    return {"operation": operation}


def _get_pending_operation(user_id: str, operation_id: int) -> dict[str, object]:
    with postgres_connection() as conn:
        row = conn.execute(
            """
            SELECT id, tool_name, operation_type, risk_level, summary, payload,
                   status, result, source_text, created_at, updated_at,
                   confirmed_at, rejected_at, conversation_id
            FROM agent_pending_operation
            WHERE user_id = %s AND id = %s AND status = 'pending'
            """,
            (user_id, operation_id),
        ).fetchone()
    if row is None:
        raise ValueError("待确认操作不存在或已处理。")
    return _operation_row(row)


def _execute_operation(user_id: str, operation: dict[str, object]) -> dict[str, object]:
    operation_type = str(operation.get("operationType") or "")
    payload = operation.get("payload") if isinstance(operation.get("payload"), dict) else {}
    conversation_id = str(operation.get("conversationId") or "") or None

    if operation_type == "todo.create":
        from app.tools.task_tool import create_todo

        return create_todo(
            user_id,
            conversation_id,
            str(payload.get("title") or ""),
            _parse_datetime(payload.get("dueAt")),
        )
    if operation_type == "calendar.create":
        from app.tools.task_tool import create_calendar_event

        return create_calendar_event(
            user_id,
            conversation_id,
            str(payload.get("title") or ""),
            _parse_datetime(payload.get("startsAt")),
        )
    return {"ok": False, "content": f"不支持的操作类型：{operation_type}"}


def _operation_row(row: tuple[object, ...]) -> dict[str, object]:
    return {
        "id": row[0],
        "toolName": row[1],
        "operationType": row[2],
        "riskLevel": row[3],
        "summary": row[4],
        "payload": row[5] if isinstance(row[5], dict) else json.loads(row[5] or "{}"),
        "status": row[6],
        "result": row[7] if isinstance(row[7], dict) or row[7] is None else json.loads(row[7] or "{}"),
        "sourceText": row[8],
        "createdAt": row[9].isoformat(),
        "updatedAt": row[10].isoformat(),
        "confirmedAt": row[11].isoformat() if row[11] else None,
        "rejectedAt": row[12].isoformat() if row[12] else None,
        "conversationId": row[13],
    }


def _parse_datetime(value: object) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed
