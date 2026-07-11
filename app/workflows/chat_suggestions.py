from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.core.database import postgres_connection


def build_execution_suggestions(user_id: str, message: str) -> list[dict[str, Any]]:
    if not settings.postgres_enabled:
        return []
    with postgres_connection() as conn:
        workspace_row = conn.execute(
            """
            SELECT id, name FROM workspace
            WHERE user_id = %s AND status = 'active'
            ORDER BY updated_at DESC LIMIT 1
            """,
            (user_id,),
        ).fetchone()
        tool_rows = conn.execute(
            """
            SELECT tool.id, tool.tool_name, tool.description, tool.risk_level
            FROM mcp_tool tool
            JOIN mcp_server server ON server.id = tool.server_id
            WHERE tool.user_id = %s AND tool.enabled = true AND server.status = 'enabled'
            ORDER BY tool.updated_at DESC LIMIT 50
            """,
            (user_id,),
        ).fetchall()
    workspace = {"id": workspace_row[0], "name": workspace_row[1]} if workspace_row else None
    tools = [
        {"id": row[0], "toolName": row[1], "description": row[2], "riskLevel": row[3]}
        for row in tool_rows
    ]
    return suggestions_for_context(message, workspace, tools)


def suggestions_for_context(
    message: str,
    workspace: dict[str, Any] | None,
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    text = message.strip().lower()
    suggestions: list[dict[str, Any]] = []
    command = _suggested_command(text)
    if workspace and command:
        suggestions.append(
            {
                "id": "cli-preview",
                "type": "cli",
                "title": f"在 {workspace['name']} 检查项目",
                "description": f"建议先预览并运行只读命令：{command}",
                "riskLevel": "read_only",
                "status": "pending",
                "action": "open_workbench",
                "payload": {"workspaceId": workspace["id"], "command": command},
            }
        )
    tool = _suggested_tool(text, tools)
    if tool:
        suggestions.append(
            {
                "id": f"mcp-{tool['id']}",
                "type": "mcp",
                "title": f"使用 MCP 工具 {tool['toolName']}",
                "description": str(tool.get("description") or "在工具工作台检查参数后调用。"),
                "riskLevel": str(tool.get("riskLevel") or "read_only"),
                "status": "pending",
                "action": "open_workbench",
                "payload": {"toolId": tool["id"], "toolName": tool["toolName"]},
            }
        )
    if _wants_workflow(text):
        workflow_steps: list[dict[str, Any]] = []
        if command:
            workflow_steps.append(
                {"title": "检查项目状态", "stepType": "cli", "commandTemplate": command, "outputKey": "inspect"}
            )
        workflow_steps.append(
            {
                "title": "汇总结果",
                "stepType": "summarize",
                "outputKey": "summary",
                "inputTemplate": {"request": message[:500]},
            }
        )
        suggestions.append(
            {
                "id": "workflow-draft",
                "type": "workflow",
                "title": "保存为可复用工作流",
                "description": "创建后仍由你在高级设置检查、编辑和运行。",
                "riskLevel": "read_only",
                "status": "pending",
                "action": "create_workflow",
                "payload": {
                    "workflow": {
                        "name": _workflow_name(message),
                        "description": message[:500],
                        "status": "active",
                        "triggerType": "manual",
                        "steps": workflow_steps,
                    }
                },
            }
        )
    return suggestions[:3]


def _suggested_command(text: str) -> str | None:
    if any(keyword in text for keyword in ("todo", "待办", "未完成", "遗留")):
        return "rg TODO"
    if any(keyword in text for keyword in ("测试", "test")):
        return "git status --short"
    if any(keyword in text for keyword in ("代码", "git", "未提交", "项目状态", "仓库", "检查")):
        return "git status --short"
    return None


def _suggested_tool(text: str, tools: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not tools:
        return None
    explicit = any(keyword in text for keyword in ("mcp", "工具", "github", "浏览器", "数据库", "文件"))
    candidates = []
    for tool in tools:
        name = str(tool.get("toolName") or "").lower()
        if any(token in text for token in name.replace(".", "_").split("_") if len(token) >= 3):
            candidates.append(tool)
    if candidates:
        return candidates[0]
    return tools[0] if explicit else None


def _wants_workflow(text: str) -> bool:
    return any(keyword in text for keyword in ("每天", "每周", "定时", "重复", "自动化", "工作流", "以后都", "每次"))


def _workflow_name(message: str) -> str:
    compact = " ".join(message.strip().split())
    return f"{compact[:40]} 工作流" if compact else "聊天建议工作流"
