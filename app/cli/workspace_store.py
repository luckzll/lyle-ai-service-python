from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import HTTPException

from app.core.config import settings
from app.core.database import postgres_connection


def create_workspace(user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_postgres()
    name = str(payload.get("name") or "").strip()
    root_path = str(payload.get("rootPath") or payload.get("root_path") or "").strip()
    if not name or not root_path:
        raise HTTPException(status_code=400, detail="工作区名称和路径不能为空。")
    resolved = str(Path(root_path).expanduser().resolve())
    with postgres_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO workspace (user_id, name, root_path, status, default_shell)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (user_id, name)
            DO UPDATE SET root_path = EXCLUDED.root_path, status = EXCLUDED.status,
                          default_shell = EXCLUDED.default_shell, updated_at = now()
            RETURNING id
            """,
            (user_id, name[:256], resolved, str(payload.get("status") or "active"), str(payload.get("defaultShell") or "zsh")),
        ).fetchone()
        conn.commit()
    return get_workspace(user_id, int(row[0]))


def list_workspaces(user_id: str) -> list[dict[str, Any]]:
    _ensure_postgres()
    with postgres_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, name, root_path, status, default_shell, created_at, updated_at
            FROM workspace
            WHERE user_id = %s
            ORDER BY updated_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [_workspace_row(row) for row in rows]


def get_workspace(user_id: str, workspace_id: int) -> dict[str, Any]:
    _ensure_postgres()
    with postgres_connection() as conn:
        row = conn.execute(
            """
            SELECT id, name, root_path, status, default_shell, created_at, updated_at
            FROM workspace
            WHERE user_id = %s AND id = %s
            """,
            (user_id, workspace_id),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="工作区不存在。")
    return _workspace_row(row)


def create_workspace_policy(user_id: str, workspace_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_postgres()
    get_workspace(user_id, workspace_id)
    command_pattern = str(payload.get("commandPattern") or payload.get("command_pattern") or "").strip()
    policy = str(payload.get("policy") or "").strip().lower()
    risk_level = str(payload.get("riskLevel") or payload.get("risk_level") or "read_only").strip()
    if not command_pattern:
        raise HTTPException(status_code=400, detail="命令规则不能为空。")
    if policy not in {"allow", "confirm", "deny"}:
        raise HTTPException(status_code=400, detail="命令策略必须是 allow、confirm 或 deny。")
    if risk_level not in {"read_only", "write_low", "write_medium", "write_high", "dangerous"}:
        raise HTTPException(status_code=400, detail="命令风险等级无效。")
    with postgres_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO workspace_command_policy
              (user_id, workspace_id, command_pattern, policy, risk_level, requires_confirmation)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                user_id,
                workspace_id,
                command_pattern[:512],
                policy,
                risk_level,
                bool(payload.get("requiresConfirmation", policy == "confirm")),
            ),
        ).fetchone()
        conn.commit()
    return get_workspace_policy(user_id, workspace_id, int(row[0]))


def list_workspace_policies(user_id: str, workspace_id: int) -> list[dict[str, Any]]:
    _ensure_postgres()
    get_workspace(user_id, workspace_id)
    with postgres_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, command_pattern, policy, risk_level, requires_confirmation, created_at, updated_at
            FROM workspace_command_policy
            WHERE user_id = %s AND workspace_id = %s
            ORDER BY created_at ASC, id ASC
            """,
            (user_id, workspace_id),
        ).fetchall()
    return [_policy_row(row, workspace_id) for row in rows]


def get_workspace_policy(user_id: str, workspace_id: int, policy_id: int) -> dict[str, Any]:
    _ensure_postgres()
    with postgres_connection() as conn:
        row = conn.execute(
            """
            SELECT id, command_pattern, policy, risk_level, requires_confirmation, created_at, updated_at
            FROM workspace_command_policy
            WHERE user_id = %s AND workspace_id = %s AND id = %s
            """,
            (user_id, workspace_id, policy_id),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="工作区命令策略不存在。")
    return _policy_row(row, workspace_id)


def delete_workspace_policy(user_id: str, workspace_id: int, policy_id: int) -> dict[str, Any]:
    _ensure_postgres()
    with postgres_connection() as conn:
        row = conn.execute(
            """
            DELETE FROM workspace_command_policy
            WHERE user_id = %s AND workspace_id = %s AND id = %s
            RETURNING id
            """,
            (user_id, workspace_id, policy_id),
        ).fetchone()
        conn.commit()
    if row is None:
        raise HTTPException(status_code=404, detail="工作区命令策略不存在。")
    return {"ok": True, "id": int(row[0])}


def _workspace_row(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "id": row[0],
        "name": row[1],
        "rootPath": row[2],
        "status": row[3],
        "defaultShell": row[4],
        "createdAt": row[5].isoformat(),
        "updatedAt": row[6].isoformat(),
    }


def _policy_row(row: tuple[Any, ...], workspace_id: int) -> dict[str, Any]:
    return {
        "id": row[0],
        "workspaceId": workspace_id,
        "commandPattern": row[1],
        "policy": row[2],
        "riskLevel": row[3],
        "requiresConfirmation": row[4],
        "createdAt": row[5].isoformat(),
        "updatedAt": row[6].isoformat(),
    }


def _ensure_postgres() -> None:
    if not settings.postgres_enabled:
        raise HTTPException(status_code=503, detail="PostgreSQL 未启用。")
