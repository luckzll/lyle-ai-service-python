"""Tenant-scoped MCP tool bridge, with durable previews for external writes."""
import hashlib
import json

from jsonschema import Draft202012Validator

from app.core.config import settings
from app.chat_tools import mcp_display_name, run_observed_tool
from app.mcp.tool_registry import list_mcp_tools, get_mcp_server_config
from app.mcp.tool_executor import invoke_mcp_tool, _get_tool
from app.mcp.client_manager import infer_tool_risk
from app.security.permission_engine import decide_permission, RISK_ORDER
from app.security.secret_masker import mask_secrets
from app.tools.operation_confirmation import create_pending_operation


def fingerprint(tool: dict) -> str:
    data = {key: tool.get(key) for key in ("id", "serverId", "toolName", "inputSchema", "riskLevel")}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def catalog(user_id: str) -> dict[str, dict]:
    if not settings.postgres_enabled or not settings.agent_loop_mcp_enabled:
        return {}
    result = {}
    for tool in list_mcp_tools(user_id):
        if not tool["enabled"]:
            continue
        server = get_mcp_server_config(user_id, int(tool["serverId"]))
        if server["status"] != "enabled":
            continue
        policy = permission(tool)
        if not policy.allowed:
            continue
        schema = tool.get("inputSchema") or {}
        try:
            Draft202012Validator.check_schema(schema)
        except Exception:
            continue
        result[f"mcp_{tool['id']}"] = tool
    return result


def effective_risk(tool: dict) -> str:
    inferred = infer_tool_risk(tool["toolName"], tool.get("description", ""))
    return max((tool["riskLevel"], inferred), key=lambda value: RISK_ORDER.get(value, 3))


def permission(tool: dict):
    risk = effective_risk(tool)
    return decide_permission(executor_type="mcp_tool", executor_name=tool["toolName"],
                             operation="invoke", risk_level=risk, trigger_type="manual")


def server_fingerprint(user_id: str, tool: dict) -> str:
    server = get_mcp_server_config(user_id, int(tool["serverId"]))
    if server["status"] != "enabled":
        raise ValueError("MCP Server 未启用。")
    config = {key: server.get(key) for key in ("serverType", "configRef", "secretRef")}
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def validate(tool: dict, arguments: dict):
    # Do not resolve remote JSON Schema references (SSRF and non-deterministic schemas).
    def check(value):
        if isinstance(value, dict):
            if "$ref" in value and not str(value["$ref"]).startswith("#"):
                raise ValueError("不支持外部 Schema 引用。")
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
    check(tool["inputSchema"])
    Draft202012Validator(tool["inputSchema"]).validate(arguments)


def invoke(user_id: str, conversation_id: str | None, selected: dict, arguments: dict, question: str) -> dict:
    return run_observed_tool(mcp_display_name(selected), lambda: _invoke(user_id, conversation_id, selected, arguments, question))


def _invoke(user_id: str, conversation_id: str | None, selected: dict, arguments: dict, question: str) -> dict:
    tool = _get_tool(user_id, int(selected["id"]))
    if not tool["enabled"] or fingerprint(tool) != fingerprint(selected):
        raise ValueError("工具配置已改变，请重新生成调用。")
    validate(tool, arguments)
    policy = permission(tool)
    if not policy.allowed:
        return {"ok": False, "content": policy.reason}
    if policy.requires_confirmation:
        return create_pending_operation(
            user_id=user_id, conversation_id=conversation_id, tool_name=f"mcp_{tool['id']}",
            operation_type="mcp.invoke", risk_level=effective_risk(tool),
            summary=mask_secrets(f"{tool['toolName']}：{json.dumps(arguments, ensure_ascii=False)}")[:4000],
            payload={"toolId": tool["id"], "arguments": arguments, "fingerprint": fingerprint(tool),
                     "serverFingerprint": server_fingerprint(user_id, tool)},
            source_text=question,
        )
    result = invoke_mcp_tool(user_id, int(tool["id"]), {"arguments": arguments})
    return {**result, "tool": f"mcp_{tool['id']}",
            "content": mask_secrets(json.dumps(result.get("result", {}), ensure_ascii=False))[:12000]}


def execute_confirmed(user_id: str, operation: dict) -> dict:
    return run_observed_tool(str(operation.get("toolName") or "mcp_tool"), lambda: _execute_confirmed(user_id, operation))


def _execute_confirmed(user_id: str, operation: dict) -> dict:
    payload = operation["payload"]
    tool = _get_tool(user_id, int(payload["toolId"]))
    if not tool["enabled"] or fingerprint(tool) != payload["fingerprint"]:
        return {"ok": False, "content": "工具配置已变化，请重新提交操作。"}
    if server_fingerprint(user_id, tool) != payload.get("serverFingerprint"):
        return {"ok": False, "content": "连接配置已变化，请重新提交操作。"}
    validate(tool, payload["arguments"])
    policy = permission(tool)
    if not policy.allowed:
        return {"ok": False, "content": policy.reason}
    if policy.requires_second_confirmation and operation["operationType"] != "mcp.invoke.second":
        return create_pending_operation(
            user_id=user_id, conversation_id=operation.get("conversationId"),
            tool_name=operation["toolName"], operation_type="mcp.invoke.second",
            risk_level=effective_risk(tool), summary="二次确认：" + operation["summary"],
            payload={**payload, "firstApprovalId": operation["id"]}, source_text="高风险操作二次确认",
        )
    result = invoke_mcp_tool(user_id, int(tool["id"]), {
        "arguments": payload["arguments"], "confirmed": True,
        "secondConfirmed": operation["operationType"] == "mcp.invoke.second",
    })
    return {**result, "content": mask_secrets(json.dumps(result.get("result", {}), ensure_ascii=False))[:12000]}
