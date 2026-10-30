from __future__ import annotations

import json
from typing import Any

from fastapi import HTTPException

from app.core.config import settings
from app.core.database import postgres_connection
from app.mcp.client_manager import masked_config_ref, normalize_config_ref, normalize_secret_ref, normalize_server_type, validate_credential_references


def create_mcp_server(user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_postgres()
    name = str(payload.get("name") or "").strip()
    server_type = normalize_server_type(payload.get("serverType") or payload.get("server_type"))
    config_ref = normalize_config_ref(server_type, payload.get("configRef") or payload.get("config_ref"))
    secret_ref = normalize_secret_ref(payload.get("secretRef") or payload.get("secret_ref"))
    if server_type == "builtin" and secret_ref is not None:
        raise HTTPException(status_code=400, detail="内置 MCP Server 不接受凭据引用。")
    validate_credential_references(config_ref, secret_ref)
    if not name:
        raise HTTPException(status_code=400, detail="MCP Server 名称不能为空。")
    with postgres_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO mcp_server (user_id, name, server_type, config_ref, secret_ref, status)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id, name)
            DO UPDATE SET server_type = EXCLUDED.server_type, config_ref = EXCLUDED.config_ref,
                          secret_ref = COALESCE(EXCLUDED.secret_ref, mcp_server.secret_ref),
                          status = EXCLUDED.status, updated_at = now()
            RETURNING id
            """,
            (user_id, name[:256], server_type, config_ref, secret_ref, str(payload.get("status") or "disabled")),
        ).fetchone()
        conn.commit()
    return get_mcp_server(user_id, int(row[0]))


def list_mcp_servers(user_id: str) -> list[dict[str, Any]]:
    _ensure_postgres()
    with postgres_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, name, server_type, config_ref, secret_ref, status, last_health_status,
                   last_health_message, last_checked_at, created_at, updated_at
            FROM mcp_server
            WHERE user_id = %s
            ORDER BY updated_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [_server_row(row) for row in rows]


def get_mcp_server(user_id: str, server_id: int) -> dict[str, Any]:
    _ensure_postgres()
    with postgres_connection() as conn:
        row = conn.execute(
            """
            SELECT id, name, server_type, config_ref, secret_ref, status, last_health_status,
                   last_health_message, last_checked_at, created_at, updated_at
            FROM mcp_server
            WHERE user_id = %s AND id = %s
            """,
            (user_id, server_id),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="MCP Server 不存在。")
    return _server_row(row)


def get_mcp_server_config(user_id: str, server_id: int) -> dict[str, Any]:
    _ensure_postgres()
    with postgres_connection() as conn:
        row = conn.execute(
            """
            SELECT id, name, server_type, config_ref, secret_ref, status, last_health_status,
                   last_health_message, last_checked_at, created_at, updated_at
            FROM mcp_server
            WHERE user_id = %s AND id = %s
            """,
            (user_id, server_id),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="MCP Server 不存在。")
    return _server_row(row, mask_config=False)


def update_mcp_server_status(user_id: str, server_id: int, status: str) -> dict[str, Any]:
    _ensure_postgres()
    normalized = status.strip().lower()
    if normalized not in {"enabled", "disabled"}:
        raise HTTPException(status_code=400, detail="MCP Server 状态必须是 enabled 或 disabled。")
    with postgres_connection() as conn:
        row = conn.execute(
            """
            UPDATE mcp_server
            SET status = %s, updated_at = now()
            WHERE user_id = %s AND id = %s
            RETURNING id
            """,
            (normalized, user_id, server_id),
        ).fetchone()
        conn.commit()
    if row is None:
        raise HTTPException(status_code=404, detail="MCP Server 不存在。")
    if normalized == "disabled":
        from app.mcp.session_pool import close_server
        close_server(server_id)
    return get_mcp_server(user_id, server_id)


def sync_mcp_tools(user_id: str, server_id: int, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _ensure_postgres()
    get_mcp_server(user_id, server_id)
    with postgres_connection() as conn:
        for tool in tools:
            name = str(tool.get("toolName") or tool.get("name") or "").strip()
            if not name:
                continue
            conn.execute(
                """
                INSERT INTO mcp_tool
                  (user_id, server_id, tool_name, description, input_schema, output_schema,
                   risk_level, enabled, last_seen_at)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, now())
                ON CONFLICT (server_id, tool_name)
                DO UPDATE SET description = EXCLUDED.description,
                              input_schema = EXCLUDED.input_schema,
                              output_schema = EXCLUDED.output_schema,
                              last_seen_at = now(),
                              updated_at = now()
                """,
                (
                    user_id,
                    server_id,
                    name[:256],
                    tool.get("description"),
                    json.dumps(tool.get("inputSchema") or tool.get("input_schema") or {}, ensure_ascii=False),
                    json.dumps(tool.get("outputSchema") or tool.get("output_schema") or {}, ensure_ascii=False),
                    str(tool.get("riskLevel") or tool.get("risk_level") or "read_only"),
                    bool(tool.get("enabled", True)),
                ),
            )
        conn.commit()
    return list_mcp_tools(user_id, server_id)


def list_mcp_tools(user_id: str, server_id: int | None = None) -> list[dict[str, Any]]:
    _ensure_postgres()
    clauses = ["user_id = %s"]
    params: list[Any] = [user_id]
    if server_id is not None:
        clauses.append("server_id = %s")
        params.append(server_id)
    with postgres_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT id, server_id, tool_name, description, input_schema, output_schema,
                   risk_level, enabled, last_seen_at, created_at, updated_at
            FROM mcp_tool
            WHERE {' AND '.join(clauses)}
            ORDER BY updated_at DESC
            """,
            tuple(params),
        ).fetchall()
    return [_tool_row(row) for row in rows]


def update_tool_policy(user_id: str, tool_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_postgres()
    risk_level = str(payload.get("riskLevel") or payload.get("risk_level") or "read_only")
    enabled = bool(payload.get("enabled", True))
    with postgres_connection() as conn:
        row = conn.execute(
            """
            UPDATE mcp_tool
            SET risk_level = %s, enabled = %s, updated_at = now()
            WHERE user_id = %s AND id = %s
            RETURNING server_id
            """,
            (risk_level, enabled, user_id, tool_id),
        ).fetchone()
        conn.commit()
    if row is None:
        raise HTTPException(status_code=404, detail="MCP Tool 不存在。")
    return [tool for tool in list_mcp_tools(user_id, int(row[0])) if tool["id"] == tool_id][0]


def mark_health(user_id: str, server_id: int, status: str, message: str = "") -> dict[str, Any]:
    _ensure_postgres()
    with postgres_connection() as conn:
        row = conn.execute(
            """
            UPDATE mcp_server
            SET last_health_status = %s, last_health_message = %s, last_checked_at = now(), updated_at = now()
            WHERE user_id = %s AND id = %s
            RETURNING id
            """,
            (status, message, user_id, server_id),
        ).fetchone()
        conn.commit()
    if row is None:
        raise HTTPException(status_code=404, detail="MCP Server 不存在。")
    return get_mcp_server(user_id, server_id)


def _server_row(row: tuple[Any, ...], *, mask_config: bool = True) -> dict[str, Any]:
    return {
        "id": row[0],
        "name": row[1],
        "serverType": row[2],
        "configRef": masked_config_ref(str(row[2]), row[3]) if mask_config else row[3],
        "secretRef": row[4] if not mask_config else None,
        "credentialConfigured": row[4] is not None,
        "status": row[5],
        "lastHealthStatus": row[6],
        "lastHealthMessage": row[7],
        "lastCheckedAt": row[8].isoformat() if row[8] else None,
        "createdAt": row[9].isoformat(),
        "updatedAt": row[10].isoformat(),
    }


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


def _ensure_postgres() -> None:
    if not settings.postgres_enabled:
        raise HTTPException(status_code=503, detail="PostgreSQL 未启用。")
