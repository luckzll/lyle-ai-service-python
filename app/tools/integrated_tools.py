"""One explicitly selected, tenant-scoped tool boundary for chat and projects."""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import shlex
import shutil
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from app.core.config import settings
from app.tools import agent_mcp
from app.cli.workspace_store import get_workspace, list_workspaces
from app.cli.command_executor import preview_command, run_command
from app.tools.operation_confirmation import create_pending_operation
from app.security.secret_masker import mask_secrets


class ToolSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mcpToolIds: list[PositiveInt] = Field(default_factory=list, max_length=20)
    workspaceId: PositiveInt | None = None
    localCli: bool = False


_selection = ContextVar("integrated_tool_selection", default=None)
_execution_id = ContextVar("integration_execution_id", default=None)


@contextmanager
def selection_scope(selection):
    token = _selection.set(selection)
    execution = _execution_id.set(str(uuid4()))
    try:
        yield
    finally:
        _selection.reset(token)
        _execution_id.reset(execution)


def current_selection():
    return _selection.get()


CLI_ACTIONS = {
    "project_git_status": ("查看 Git 状态", "git", ["git", "status", "--short"]),
    "project_git_diff": ("查看代码差异", "git", ["git", "diff", "--no-ext-diff", "--no-textconv", "--stat"]),
    "project_search": ("搜索项目源码", "rg", None),
    "project_github_issues": ("查看 GitHub Issues", "gh", ["gh", "issue", "list", "--limit", "20", "--json", "number,title,state,url"]),
    "project_github_prs": ("查看 GitHub PR", "gh", ["gh", "pr", "list", "--limit", "20", "--json", "number,title,state,url"]),
    "project_run_tests": ("运行项目测试（npm / pytest / Maven）", "npm", None),
}
TEST_COMMANDS = {"npm": ["npm", "test", "--", "--run"], "pytest": ["pytest", "-q"], "maven": ["mvn", "test", "-B"]}


def cli_schemas():
    return [{"type": "function", "function": {
        "name": name, "description": f"{label}。在用户绑定的执行工作区运行，不能用于浏览器本机目录；命令执行前会生成确认预览。" ,
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 256}} if name == "project_search" else {"runner": {"type": "string", "enum": list(TEST_COMMANDS)}} if name == "project_run_tests" else {},
                       "required": ["query"] if name == "project_search" else ["runner"] if name == "project_run_tests" else [], "additionalProperties": False},
    }} for name, (label, _, _) in CLI_ACTIONS.items()]


def selected_mcp(user_id, selection):
    if selection is not None and not selection.mcpToolIds:
        return {}
    tools = agent_mcp.catalog(user_id)
    if selection is None:  # Keep existing clients' chat behavior.
        return tools
    ids = set(selection.mcpToolIds)
    return {name: tool for name, tool in tools.items() if tool["id"] in ids}


def schemas(user_id, selection, *, read_only=False):
    tools = selected_mcp(user_id, selection)
    if read_only:
        tools = {name: tool for name, tool in tools.items() if agent_mcp.effective_risk(tool) == "read_only"}
    result = [{"type": "function", "function": {"name": name,
              "description": str(tool.get("description") or tool["toolName"])[:1000],
              "parameters": tool["inputSchema"]}} for name, tool in tools.items()]
    if selection and selection.workspaceId and not read_only:
        get_workspace(user_id, selection.workspaceId)  # Never accept another tenant's ID.
        result.extend(cli_schemas())
    return result


def command_for(name, arguments):
    if name not in CLI_ACTIONS or not isinstance(arguments, dict):
        raise ValueError("工具不存在。")
    expected = {"query"} if name == "project_search" else {"runner"} if name == "project_run_tests" else set()
    if set(arguments) != expected:
        raise ValueError("工具参数无效；不能指定路径、确认标志或自定义命令。")
    if name == "project_search":
        query = arguments["query"]
        if not isinstance(query, str) or not query.strip() or len(query) > 256 or any(ord(c) < 32 for c in query):
            raise ValueError("请输入 1–256 个字符的搜索文字。")
        parts = ["rg", "--line-number", "--max-count", "20", "--max-filesize", "1M", "--fixed-strings", "--", query, "."]
    elif name == "project_run_tests":
        runner = arguments["runner"]
        if not isinstance(runner, str) or runner not in TEST_COMMANDS:
            raise ValueError("请选择 npm、pytest 或 Maven 测试。")
        parts = TEST_COMMANDS[runner]
    else:
        parts = CLI_ACTIONS[name][2]
    return shlex.join(parts)


def workspace_fingerprint(workspace):
    return hashlib.sha256(json.dumps({key: workspace.get(key) for key in ("id", "rootPath", "status")}, sort_keys=True).encode()).hexdigest()


def invoke(user_id, conversation_id, name, arguments, selection, question="", invocation_id=None):
    if selection is None:
        raise ValueError("请先在本轮选择工具。")
    if name.startswith("mcp_"):
        tool = selected_mcp(user_id, selection).get(name)
        if not tool:
            raise ValueError("工具未启用或未在本轮选择。")
        return agent_mcp.invoke(user_id, conversation_id, tool, arguments, question)
    if not selection.workspaceId:
        raise ValueError("请先绑定执行工作区；本机目录连接不等于服务器工作区。")
    command = command_for(name, arguments)
    preview = preview_command(user_id, selection.workspaceId, command)
    if not preview["permission"].get("allowed") or preview["risk"].get("policy") == "deny":
        return {"ok": False, "content": "工作区策略禁止此命令。"}
    workspace = preview["workspace"]
    result = create_pending_operation(user_id=user_id, conversation_id=conversation_id,
        tool_name=name, operation_type="cli.selected", risk_level=str(preview["risk"]["riskLevel"]),
        summary=f"在执行工作区 {workspace['name']}（服务端）运行：{command}",
        payload={"workspaceId": selection.workspaceId, "name": name, "arguments": arguments,
                 "fingerprint": workspace_fingerprint(workspace),
                 "executionId": invocation_id or _execution_id.get() or str(uuid4())}, source_text=question)
    operation = result.get("operation") or {}
    if operation.get("status") in {"confirmed", "failed"} and operation.get("result"):
        return {**operation["result"], "tool": name, "operation": operation, "idempotentReplay": True}
    return result


def execute_confirmed(user_id, operation):
    payload = operation["payload"]
    workspace = get_workspace(user_id, payload["workspaceId"])
    if workspace_fingerprint(workspace) != payload["fingerprint"]:
        return {"ok": False, "content": "工作区绑定已变化，请重新提交。"}
    command = command_for(payload["name"], payload["arguments"])
    result = run_command(user_id, payload["workspaceId"], command, confirmed=True)
    if payload["name"] == "project_search" and result.get("exitCode") == 1:
        result = {**result, "ok": True, "summary": "没有匹配结果。"}
    return {**result, "content": mask_secrets(str(result.get("summary") or result.get("permission", {}).get("reason") or "命令未完成"))[:12000]}


def catalog(user_id):
    warning = ""
    try:
        tools = agent_mcp.catalog(user_id)
        workspaces = list_workspaces(user_id) if settings.postgres_enabled else []
    except Exception:
        tools, workspaces = {}, []
        warning = "服务端连接列表暂不可用，请稍后刷新。本机命令仍可单独连接。"
    return {"tools": [{"name": name, "id": tool["id"], "label": tool["toolName"],
                        "description": tool.get("description") or "", "riskLevel": agent_mcp.effective_risk(tool),
                        "location": "server"} for name, tool in tools.items()],
            "workspaces": [{"id": w["id"], "name": w["name"], "location": "server"} for w in workspaces if w["status"] == "active"],
            "availabilityError": warning,
            "cli": [{"name": name, "label": label,
                     "available": any(shutil.which(parts[0]) for parts in TEST_COMMANDS.values()) if name == "project_run_tests" else bool(shutil.which(binary)),
                     "binary": "npm / pytest / mvn" if name == "project_run_tests" else binary}
                    for name, (label, binary, _) in CLI_ACTIONS.items()]}
