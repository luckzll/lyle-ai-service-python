from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from app.cli.command_executor import run_command
from app.core.config import settings
from app.core.database import postgres_connection
from app.mcp.client_manager import run_mcp_tool_call
from app.mcp.tool_registry import get_mcp_server_config
from app.security.execution_audit import write_execution_audit
from app.security.permission_engine import decide_permission


BUILTIN_TOOL_COMMANDS = {
    "workspace.git_status": "git status",
    "workspace.rg_todo": "rg TODO",
}


def invoke_mcp_tool(user_id: str, tool_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    if not settings.postgres_enabled:
        raise HTTPException(status_code=503, detail="PostgreSQL 未启用。")
    tool = _get_tool(user_id, tool_id)
    if not tool["enabled"]:
        raise HTTPException(status_code=400, detail="MCP Tool 已禁用。")
    decision = decide_permission(
        executor_type="mcp_tool",
        executor_name=str(tool["toolName"]),
        operation="invoke",
        risk_level=str(tool["riskLevel"]),
        trigger_type=str(payload.get("triggerType") or "manual"),
    )
    if not decision.allowed:
        write_execution_audit(
            user_id=user_id,
            source_type="mcp_tool",
            source_id=str(tool_id),
            executor_type="mcp_tool",
            executor_name=str(tool["toolName"]),
            operation="invoke",
            risk_level=str(tool["riskLevel"]),
            status="blocked",
            input_summary=str(payload.get("arguments") or {}),
            error_code="MCP_TOOL_BLOCKED",
            error_message=decision.reason,
        )
        return {"ok": False, "blocked": True, "permission": decision.to_dict(), "tool": tool}
    if decision.requires_confirmation and not payload.get("confirmed"):
        return {
            "ok": False,
            "requiresConfirmation": True,
            "requiresSecondConfirmation": decision.requires_second_confirmation,
            "permission": decision.to_dict(),
            "tool": tool,
        }
    if decision.requires_second_confirmation and not payload.get("secondConfirmed"):
        return {
            "ok": False,
            "requiresConfirmation": True,
            "requiresSecondConfirmation": True,
            "permission": decision.to_dict(),
            "tool": tool,
        }
    tool_name = str(tool["toolName"])
    server = get_mcp_server_config(user_id, int(tool["serverId"]))
    if server["status"] != "enabled":
        raise HTTPException(status_code=400, detail="MCP Server 未启用。")
    if server["serverType"] == "builtin" and tool_name in BUILTIN_TOOL_COMMANDS:
        args = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
        workspace_id = args.get("workspaceId") or payload.get("workspaceId")
        if workspace_id is None:
            raise HTTPException(status_code=400, detail="该工具需要 workspaceId。")
        result = run_command(
            user_id,
            int(workspace_id),
            BUILTIN_TOOL_COMMANDS[tool_name],
            confirmed=True,
            second_confirmed=True,
        )
        write_execution_audit(
            user_id=user_id,
            source_type="mcp_tool",
            source_id=str(tool_id),
            executor_type="mcp_tool",
            executor_name=tool_name,
            operation="invoke",
            risk_level=str(tool["riskLevel"]),
            status="completed" if result.get("ok") else "failed",
            input_summary=str(args),
            output_summary=str(result.get("summary") or result),
            error_code=None if result.get("ok") else str(result.get("errorCode") or "MCP_TOOL_FAILED"),
        )
        return {"ok": bool(result.get("ok")), "tool": tool, "result": result}
    if server["serverType"] == "builtin":
        raise HTTPException(status_code=400, detail=f"暂不支持调用内置 MCP Tool：{tool_name}")
    args = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
    try:
        result = run_mcp_tool_call(server, tool_name, args)
    except HTTPException as exc:
        write_execution_audit(
            user_id=user_id,
            source_type="mcp_tool",
            source_id=str(tool_id),
            executor_type="mcp_tool",
            executor_name=tool_name,
            operation="invoke",
            risk_level=str(tool["riskLevel"]),
            status="failed",
            input_summary=str(args),
            error_code="MCP_TOOL_FAILED",
            error_message=str(exc.detail),
        )
        raise
    write_execution_audit(
        user_id=user_id,
        source_type="mcp_tool",
        source_id=str(tool_id),
        executor_type="mcp_tool",
        executor_name=tool_name,
        operation="invoke",
        risk_level=str(tool["riskLevel"]),
        status="completed" if result.get("ok") else "failed",
        input_summary=str(args),
        output_summary=str(result),
        error_code=None if result.get("ok") else "MCP_TOOL_ERROR_RESULT",
    )
    return {"ok": bool(result.get("ok")), "tool": tool, "result": result}


def invoke_mcp_tool_by_name(user_id: str, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    tool = _get_tool_by_name(user_id, tool_name)
    return invoke_mcp_tool(user_id, int(tool["id"]), payload)


def _get_tool(user_id: str, tool_id: int) -> dict[str, Any]:
    with postgres_connection() as conn:
        row = conn.execute(
            """
            SELECT id, server_id, tool_name, description, input_schema, output_schema,
                   risk_level, enabled, last_seen_at, created_at, updated_at
            FROM mcp_tool
            WHERE user_id = %s AND id = %s
            """,
            (user_id, tool_id),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="MCP Tool 不存在。")
    return _tool_row(row)


def _get_tool_by_name(user_id: str, tool_name: str) -> dict[str, Any]:
    with postgres_connection() as conn:
        row = conn.execute(
            """
            SELECT id, server_id, tool_name, description, input_schema, output_schema,
                   risk_level, enabled, last_seen_at, created_at, updated_at
            FROM mcp_tool
            WHERE user_id = %s AND tool_name = %s
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (user_id, tool_name),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="MCP Tool 不存在。")
    return _tool_row(row)


def _tool_row(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "id": row[0],
        "serverId": row[1],
        "toolName": row[2],
        "description": row[3],
        "inputSchema": row[4] or {},
        "outputSchema": row[5] or {},
        "riskLevel": row[6],
        "enabled": row[7],
        "lastSeenAt": row[8].isoformat() if row[8] else None,
        "createdAt": row[9].isoformat(),
        "updatedAt": row[10].isoformat(),
    }
