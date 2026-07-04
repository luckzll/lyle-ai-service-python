import json
import hashlib
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
            "error": {
                "code": "DATABASE_ERROR",
                "message": "PostgreSQL 未启用，无法生成待确认操作。",
                "retryable": False,
            },
        }

    _ensure_pending_operation_schema()
    idempotency_key = _idempotency_key(
        user_id=user_id,
        conversation_id=conversation_id,
        tool_name=tool_name,
        operation_type=operation_type,
        payload=payload,
    )
    now = datetime.now(UTC)
    with postgres_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO agent_pending_operation
              (user_id, conversation_id, tool_name, operation_type, risk_level,
               summary, payload, idempotency_key, status, source_text, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, 'pending', %s, %s, %s)
            ON CONFLICT (user_id, idempotency_key)
            DO UPDATE SET updated_at = agent_pending_operation.updated_at
            RETURNING id, tool_name, operation_type, risk_level, summary, payload,
                      status, result, source_text, created_at, updated_at,
                      confirmed_at, rejected_at, conversation_id, idempotency_key
            """,
            (
                user_id,
                conversation_id or "",
                tool_name,
                operation_type,
                risk_level,
                summary,
                json.dumps(payload, ensure_ascii=False),
                idempotency_key,
                source_text[:1000],
                now,
                now,
            ),
        ).fetchone()
        conn.commit()

    operation = _operation_row(row)
    already_confirmed = operation.get("status") == "confirmed"
    return {
        "ok": True,
        "content": (
            f"该操作已确认过：{summary}。"
            if already_confirmed else
            f"已生成待确认操作：{summary}。用户确认后才会执行。"
        ),
        "requiresConfirmation": not already_confirmed,
        "operation": operation,
        "idempotentReplay": operation["createdAt"] != now.isoformat(),
    }


def list_pending_operations(user_id: str, limit: int = 50) -> dict[str, object]:
    if not settings.postgres_enabled:
        return {"items": []}
    _ensure_pending_operation_schema()
    with postgres_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, tool_name, operation_type, risk_level, summary, payload,
                   status, result, source_text, created_at, updated_at,
                   confirmed_at, rejected_at, conversation_id, idempotency_key
            FROM agent_pending_operation
            WHERE user_id = %s
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            (user_id, limit),
        ).fetchall()
    return {"items": [_operation_row(row) for row in rows]}


def confirm_pending_operation(user_id: str, operation_id: int) -> dict[str, object]:
    _ensure_pending_operation_schema()
    now = datetime.now(UTC)
    with postgres_connection() as conn:
        operation = _get_operation_for_update(conn, user_id, operation_id)
        if operation.get("status") == "confirmed":
            conn.commit()
            return {
                "operation": operation,
                "result": operation.get("result") or _tool_result(
                    ok=True,
                    tool=str(operation.get("toolName") or ""),
                    operation_type=str(operation.get("operationType") or ""),
                    user_message="该操作已经确认执行过，本次没有重复执行。",
                    result={},
                ),
                "idempotentReplay": True,
            }
        if operation.get("status") != "pending":
            conn.commit()
            raise ValueError("待确认操作不存在或已处理。")
        result = _execute_operation(user_id, operation)
        row = conn.execute(
            """
            UPDATE agent_pending_operation
            SET status = 'confirmed', result = %s::jsonb, updated_at = %s, confirmed_at = %s
            WHERE user_id = %s AND id = %s AND status = 'pending'
            RETURNING id, tool_name, operation_type, risk_level, summary, payload,
                      status, result, source_text, created_at, updated_at,
                      confirmed_at, rejected_at, conversation_id, idempotency_key
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
    _ensure_pending_operation_schema()
    now = datetime.now(UTC)
    with postgres_connection() as conn:
        row = conn.execute(
            """
            UPDATE agent_pending_operation
            SET status = 'rejected', updated_at = %s, rejected_at = %s
            WHERE user_id = %s AND id = %s AND status = 'pending'
            RETURNING id, tool_name, operation_type, risk_level, summary, payload,
                      status, result, source_text, created_at, updated_at,
                      confirmed_at, rejected_at, conversation_id, idempotency_key
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


def _get_operation(user_id: str, operation_id: int) -> dict[str, object]:
    with postgres_connection() as conn:
        row = conn.execute(
            """
            SELECT id, tool_name, operation_type, risk_level, summary, payload,
                   status, result, source_text, created_at, updated_at,
                   confirmed_at, rejected_at, conversation_id, idempotency_key
            FROM agent_pending_operation
            WHERE user_id = %s AND id = %s
            """,
            (user_id, operation_id),
        ).fetchone()
    if row is None:
            raise ValueError("待确认操作不存在或已处理。")
    return _operation_row(row)


def _get_operation_for_update(conn, user_id: str, operation_id: int) -> dict[str, object]:
    row = conn.execute(
        """
        SELECT id, tool_name, operation_type, risk_level, summary, payload,
               status, result, source_text, created_at, updated_at,
               confirmed_at, rejected_at, conversation_id, idempotency_key
        FROM agent_pending_operation
        WHERE user_id = %s AND id = %s
        FOR UPDATE
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

        return _normalise_tool_result(
            create_todo(
                user_id,
                conversation_id,
                str(payload.get("title") or ""),
                _parse_datetime(payload.get("dueAt")),
            ),
            tool="todo",
            operation_type=operation_type,
            risk_level=str(operation.get("riskLevel") or "write"),
        )
    if operation_type == "calendar.create":
        from app.tools.task_tool import create_calendar_event

        return _normalise_tool_result(
            create_calendar_event(
                user_id,
                conversation_id,
                str(payload.get("title") or ""),
                _parse_datetime(payload.get("startsAt")),
            ),
            tool="calendar",
            operation_type=operation_type,
            risk_level=str(operation.get("riskLevel") or "write"),
        )
    if operation_type == "connector.local_docs.create":
        from app.connectors.local_docs import execute_local_doc_create

        return execute_local_doc_create(payload)
    if operation_type == "connector.github.issue.create":
        from app.connectors.github_connector import execute_github_issue_create

        return execute_github_issue_create(payload)
    return _tool_result(
        ok=False,
        tool=str(operation.get("toolName") or ""),
        operation_type=operation_type,
        user_message=f"不支持的操作类型：{operation_type}",
        error_code="PARAM_MISSING",
        retryable=False,
    )


def _operation_row(row: tuple[object, ...]) -> dict[str, object]:
    item = {
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
    if len(row) > 14:
        item["idempotencyKey"] = row[14]
    return item


def _ensure_pending_operation_schema() -> None:
    with postgres_connection() as conn:
        conn.execute("ALTER TABLE agent_pending_operation ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(128)")
        conn.execute(
            """
            UPDATE agent_pending_operation
            SET idempotency_key = 'legacy:' || id::text
            WHERE idempotency_key IS NULL OR idempotency_key = ''
            """
        )
        conn.execute("ALTER TABLE agent_pending_operation ALTER COLUMN idempotency_key SET NOT NULL")
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_pending_operation_user_idempotency
            ON agent_pending_operation(user_id, idempotency_key)
            """
        )
        conn.commit()


def _idempotency_key(
    *,
    user_id: str,
    conversation_id: str | None,
    tool_name: str,
    operation_type: str,
    payload: dict[str, Any],
) -> str:
    normalized_payload = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    raw = "|".join([user_id, conversation_id or "", tool_name, operation_type, normalized_payload])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _normalise_tool_result(
    result: dict[str, object],
    *,
    tool: str,
    operation_type: str,
    risk_level: str,
) -> dict[str, object]:
    if "operationType" in result and "userMessage" in result:
        return result
    return _tool_result(
        ok=bool(result.get("ok")),
        tool=tool,
        operation_type=operation_type,
        risk_level=risk_level,
        user_message=str(result.get("content") or ""),
        result={key: value for key, value in result.items() if key not in {"ok", "content"}},
        error_code=None if result.get("ok") else "DATABASE_ERROR",
        retryable=False,
    )


def _tool_result(
    *,
    ok: bool,
    tool: str,
    operation_type: str,
    user_message: str,
    risk_level: str = "write",
    result: dict[str, object] | None = None,
    error_code: str | None = None,
    retryable: bool = False,
) -> dict[str, object]:
    return {
        "ok": ok,
        "tool": tool,
        "operationType": operation_type,
        "riskLevel": risk_level,
        "requiresConfirmation": False,
        "retryable": retryable,
        "userMessage": user_message,
        "content": user_message,
        "result": result or {},
        "error": None if ok else {
            "code": error_code or "UNKNOWN_ERROR",
            "message": user_message,
            "retryable": retryable,
        },
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
