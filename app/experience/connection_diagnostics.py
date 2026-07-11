from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from app.cli.workspace_store import list_workspaces
from app.core.config import settings
from app.mcp.tool_registry import list_mcp_servers
from app.experience.events import record_event


def recommended_connections(user_id: str) -> dict[str, Any]:
    diagnostics = connection_diagnostics(user_id)["items"]
    kinds = {item["type"] for item in diagnostics}
    items = diagnostics[:]
    if "local_project" not in kinds:
        items.insert(0, {"id": "local-project", "type": "local_project", "name": "本地项目", "status": "unavailable", "displayReason": "尚未授权本地工作区。", "nextAction": "前往添加工作区", "actionTarget": "/advanced-settings?tab=cli", "diagnosable": False, "capabilities": ["读取 Git 状态", "查找 TODO"]})
    if "github" not in kinds:
        items.append({"id": "github", "type": "github", "name": "GitHub", "status": "unavailable", "displayReason": "当前没有可用的 GitHub 连接。", "nextAction": "添加 GitHub 连接", "actionTarget": "/advanced-settings?tab=mcp", "diagnosable": False, "capabilities": ["读取 PR 与 Issue"]})
    return {"items": items}


def connection_diagnostics(user_id: str) -> dict[str, Any]:
    if not settings.postgres_enabled:
        raise HTTPException(status_code=503, detail="PostgreSQL 未启用。")
    items: list[dict[str, Any]] = []
    for workspace in list_workspaces(user_id):
        healthy = workspace.get("status") == "active"
        items.append({"id": f"workspace-{workspace['id']}", "sourceId": workspace["id"], "type": "local_project", "name": workspace["name"], "status": "healthy" if healthy else "restricted", "displayReason": "可读取已授权工作区。" if healthy else "工作区已停用。", "nextAction": "运行连接诊断", "diagnosable": True, "capabilities": ["读取 Git 状态", "查找 TODO"], "lastCheckedAt": workspace.get("updatedAt")})
    for server in list_mcp_servers(user_id):
        raw_status = str(server.get("status") or "")
        last_health = str(server.get("lastHealthStatus") or "")
        status = "healthy" if raw_status == "enabled" and last_health in {"ok", "healthy"} else "reauthorize_required" if "auth" in str(server.get("lastHealthMessage") or "").lower() else "unavailable"
        server_type = "github" if "github" in f"{server.get('name', '')} {server.get('endpoint', '')}".lower() else "mcp"
        items.append({"id": f"mcp-{server['id']}", "sourceId": server["id"], "type": server_type, "name": server["name"], "status": status, "displayReason": server.get("lastHealthMessage") or ("连接可用。" if status == "healthy" else "需要运行连接诊断。"), "nextAction": "运行连接诊断", "diagnosable": True, "capabilities": ["读取已授权资源"], "lastCheckedAt": server.get("lastHealthAt")})
    return {"items": items}


def diagnose_connection(user_id: str, connection_id: str) -> dict[str, Any]:
    record_event(user_id, "connection_diagnostic_started")
    item = next((item for item in connection_diagnostics(user_id)["items"] if item["id"] == connection_id), None)
    if item is None:
        raise HTTPException(status_code=404, detail="连接不存在。")
    return item
