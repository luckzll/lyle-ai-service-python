from __future__ import annotations

import re
from typing import Any

from fastapi import HTTPException

from app.cli.workspace_store import get_workspace, list_workspaces
from app.core.config import settings
from app.experience.events import record_event
from app.mcp.client_manager import run_mcp_health_check
from app.mcp.tool_registry import get_mcp_server_config, list_mcp_servers, mark_health


_HEALTHY_STATES = {"ok", "healthy"}
_UNCHECKED_STATES = {"", "unknown", "unchecked", "pending", "not_checked"}
_AUTH_ERROR = re.compile(
    r"\b401\b|unauthori[sz]ed|authentication|auth_required|reauthoriz|"
    r"invalid[\s_-]*(?:token|credentials?)|expired[\s_-]*(?:token|credentials?)|"
    r"credential|未授权|授权.*(?:失效|过期|失败)|凭据|凭证|认证失败|令牌.*(?:失效|过期)",
    re.IGNORECASE,
)
_PERMISSION_ERROR = re.compile(r"\b403\b|forbidden|permission[\s_-]*denied|权限不足|权限受限", re.IGNORECASE)
_REASONS = {
    "healthy": "连接检查通过，可读取已授权资源。",
    "disabled": "连接已停用，可按需启用。",
    "unchecked": "连接已配置，尚未运行检查。",
    "reauthorize_required": "连接授权已失效或凭据不可用，请更新授权后重试。",
    "restricted": "连接权限受限，请检查授权范围。",
    "unavailable": "连接检查未通过，请检查配置或稍后重试。",
}


def recommended_connections(user_id: str) -> dict[str, Any]:
    diagnostics = connection_diagnostics(user_id)["items"]
    kinds = {item["type"] for item in diagnostics}
    items = diagnostics[:]
    if "local_project" not in kinds:
        items.insert(0, {
            "id": "local-project", "type": "local_project", "name": "本地项目",
            "status": "not_configured", "configured": False, "needsAttention": False,
            "displayReason": "尚未授权本地工作区。", "nextAction": "前往添加工作区",
            "actionTarget": "/advanced-settings?tab=cli", "diagnosable": False,
            "capabilities": ["读取 Git 状态", "查找 TODO"], "lastCheckedAt": None,
        })
    if "github" not in kinds:
        github_configured = bool(settings.connector_github_token.strip())
        items.append({
            "id": "github", "type": "github", "name": "GitHub",
            "status": "unchecked" if github_configured else "not_configured",
            "configured": github_configured, "needsAttention": False,
            "displayReason": "GitHub 只读连接已配置，尚未检查。" if github_configured else "当前没有可用的 GitHub 连接。",
            "nextAction": "查看连接器" if github_configured else "添加 GitHub 连接",
            "actionTarget": "/advanced-settings?tab=connectors" if github_configured else "/advanced-settings?tab=mcp",
            "diagnosable": False,
            "capabilities": ["读取 PR 与 Issue"], "lastCheckedAt": None,
        })
    return {"items": items}


def connection_diagnostics(user_id: str) -> dict[str, Any]:
    _ensure_postgres()
    items: list[dict[str, Any]] = []
    for workspace in list_workspaces(user_id):
        active = workspace.get("status") == "active"
        # A saved authorization cannot prove that a user's local host is reachable
        # from this service. Do not present its updated_at as a completed check.
        items.append({
            "id": f"workspace-{workspace['id']}", "sourceId": workspace["id"],
            "type": "local_project", "name": workspace["name"],
            "status": "unchecked" if active else "disabled", "configured": True,
            "needsAttention": False, "diagnosable": False,
            "displayReason": "工作区已授权；本地环境尚未检查。" if active else "工作区已停用。",
            "nextAction": "查看工作区设置", "actionTarget": "/advanced-settings?tab=cli",
            "capabilities": ["读取 Git 状态", "查找 TODO"], "lastCheckedAt": None,
        })
    items.extend(_mcp_diagnostic(server) for server in list_mcp_servers(user_id))
    return {"items": items}


def diagnose_connection(user_id: str, connection_id: str) -> dict[str, Any]:
    _ensure_postgres()
    workspace_match = re.fullmatch(r"workspace-([1-9]\d*)", connection_id)
    if workspace_match:
        get_workspace(user_id, int(workspace_match[1]))
        raise HTTPException(status_code=409, detail="本地工作区需在本地项目中检查，请前往工作区设置。")
    server_match = re.fullmatch(r"mcp-([1-9]\d*)", connection_id)
    if not server_match:
        raise HTTPException(status_code=404, detail="连接不存在。")

    server_id = int(server_match[1])
    # This lookup is scoped to the current user and must precede any network or
    # process execution. Use the stored configuration, never a client endpoint.
    server = get_mcp_server_config(user_id, server_id)
    if server.get("status") != "enabled":
        raise HTTPException(status_code=409, detail="连接已停用，请先启用后再运行检查。")
    record_event(user_id, "connection_diagnostic_started")
    try:
        result = run_mcp_health_check(server)
        status = "healthy" if result.get("ok") is True else _failure_status(str(result.get("message") or ""))
    except HTTPException as exc:
        status = _failure_status(str(exc.detail), exc.status_code)
    except Exception:
        # Transport errors can include credentials, commands or private URLs.
        # Store and expose only a bounded, user-facing diagnostic category.
        status = "unavailable"
    stored_status = {"healthy": "ok", "reauthorize_required": "auth_required", "restricted": "restricted"}.get(status, "error")
    checked = mark_health(user_id, server_id, stored_status, _REASONS[status])
    return _mcp_diagnostic(checked)


def _failure_status(message: str, status_code: int | None = None) -> str:
    if status_code == 401 or _AUTH_ERROR.search(message):
        return "reauthorize_required"
    if status_code == 403 or _PERMISSION_ERROR.search(message):
        return "restricted"
    return "unavailable"


def _mcp_diagnostic(server: dict[str, Any]) -> dict[str, Any]:
    raw_status = str(server.get("status") or "").lower()
    last_health = str(server.get("lastHealthStatus") or "").lower()
    if raw_status != "enabled":
        status = "disabled"
    elif last_health in _HEALTHY_STATES:
        status = "healthy"
    elif last_health in _UNCHECKED_STATES:
        status = "unchecked"
    elif last_health in {"auth_required", "reauthorize_required", "unauthorized"}:
        status = "reauthorize_required"
    elif last_health == "restricted":
        status = "restricted"
    else:
        status = _failure_status(str(server.get("lastHealthMessage") or ""))
    server_type = "github" if "github" in str(server.get("name") or "").lower() else "mcp"
    enabled = raw_status == "enabled"
    return {
        "id": f"mcp-{server['id']}", "sourceId": server["id"], "type": server_type,
        "name": server["name"], "status": status, "configured": True,
        "needsAttention": enabled and status in {"unavailable", "reauthorize_required", "restricted"},
        "displayReason": _REASONS[status],
        "nextAction": "运行连接诊断" if enabled else "前往启用连接",
        **({} if enabled else {"actionTarget": "/advanced-settings?tab=mcp"}),
        "diagnosable": enabled, "capabilities": ["读取已授权资源"],
        "lastCheckedAt": server.get("lastCheckedAt"),
    }


def _ensure_postgres() -> None:
    if not settings.postgres_enabled:
        raise HTTPException(status_code=503, detail="PostgreSQL 未启用。")
