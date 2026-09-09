from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from app.core.config import settings
from app.core.database import postgres_connection
from app.tools.operation_confirmation import confirm_pending_operation, reject_pending_operation
from app.experience.events import record_event


def list_confirmations(user_id: str, limit: int = 100) -> dict[str, Any]:
    _ensure()
    with postgres_connection() as conn:
        rows = conn.execute(
            """SELECT id, tool_name, operation_type, risk_level, summary, payload, status,
               source_text, created_at, updated_at, conversation_id
               FROM agent_pending_operation WHERE user_id=%s AND status='pending'
               ORDER BY CASE risk_level WHEN 'high' THEN 3 WHEN 'write' THEN 2 ELSE 1 END DESC,
               created_at ASC LIMIT %s""", (user_id, max(1, min(limit, 200))),
        ).fetchall()
    items = [_display(row) for row in rows]
    if items:
        record_event(user_id, "confirmation_viewed", status="pending")
    return {"items": items}


def get_confirmation(user_id: str, operation_id: int) -> dict[str, Any]:
    items = list_confirmations(user_id, 200)["items"]
    item = next((item for item in items if item["pendingOperationId"] == operation_id), None)
    if item is None:
        raise HTTPException(status_code=404, detail="待确认操作不存在或已处理。")
    return item


def confirm(user_id: str, operation_id: int) -> dict[str, Any]:
    try:
        result = confirm_pending_operation(user_id, operation_id)
        record_event(user_id, "confirmation_confirmed", status="confirmed")
        return result
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def reject(user_id: str, operation_id: int) -> dict[str, Any]:
    try:
        result = reject_pending_operation(user_id, operation_id)
        record_event(user_id, "confirmation_rejected", status="rejected")
        return result
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _display(row: tuple[Any, ...]) -> dict[str, Any]:
    payload = row[5] if isinstance(row[5], dict) else {}
    risk = str(row[3] or "write")
    return {
        "pendingOperationId": row[0], "actionTitle": row[4], "toolName": row[1], "operationType": row[2],
        "riskLevel": risk, "status": row[6], "reason": row[7] or "来自 Lyle 的建议操作。",
        "impactResources": [str(payload.get(key)) for key in ("projectId", "taskId", "path", "repo") if payload.get(key) is not None] or ["已授权范围"],
        "reversible": risk not in {"high", "write_high", "dangerous"} and not str(row[2]).startswith("mcp.invoke"),
        "requiresSecondConfirmation": risk in {"high", "write_high"},
        "beforeAfter": payload.get("diff") or None, "source": f"会话 {row[10]}" if row[10] else "Lyle 执行建议",
        "createdAt": row[8].isoformat(), "updatedAt": row[9].isoformat(),
    }


def _ensure() -> None:
    if not settings.postgres_enabled:
        raise HTTPException(status_code=503, detail="PostgreSQL 未启用。")
