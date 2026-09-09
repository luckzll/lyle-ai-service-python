from __future__ import annotations

import asyncio
import json
import os
import re
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException


SUPPORTED_SERVER_TYPES = {"stdio", "streamable_http"}
SECRET_REF_PATTERN = re.compile(r"^env:([A-Z][A-Z0-9_]{0,127})$")
SENSITIVE_CONFIG_KEY = re.compile(r"(authorization|token|secret|password|api[_-]?key|credential)", re.IGNORECASE)


def normalize_server_type(value: object) -> str:
    server_type = str(value or "builtin").strip().lower().replace("-", "_")
    if server_type in {"http", "streamablehttp"}:
        return "streamable_http"
    if server_type not in {"builtin", *SUPPORTED_SERVER_TYPES}:
        raise HTTPException(status_code=400, detail=f"不支持的 MCP 连接类型：{server_type}")
    return server_type


def normalize_config_ref(server_type: str, value: object) -> str | None:
    if server_type == "builtin":
        return None
    if isinstance(value, str):
        try:
            config = json.loads(value)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="MCP 连接配置必须是 JSON 对象。") from exc
    elif isinstance(value, dict):
        config = value
    else:
        raise HTTPException(status_code=400, detail="MCP 连接配置不能为空。")
    _validate_config(server_type, config)
    return json.dumps(config, ensure_ascii=False)


def parse_config_ref(server: dict[str, Any]) -> dict[str, Any]:
    raw = server.get("configRef")
    if isinstance(raw, dict):
        config = raw
    else:
        try:
            config = json.loads(str(raw or "{}"))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="MCP Server 连接配置无效。") from exc
    _validate_config(str(server.get("serverType") or ""), config)
    return _resolve_secret_reference(config, server.get("secretRef"))


def normalize_secret_ref(value: object) -> str | None:
    if value in (None, ""):
        return None
    reference = str(value).strip()
    if not SECRET_REF_PATTERN.fullmatch(reference):
        raise HTTPException(status_code=400, detail="secretRef 必须是 env:UPPERCASE_ENVIRONMENT_VARIABLE 格式。")
    return reference


def validate_credential_references(config_ref: str | None, secret_ref: str | None) -> None:
    if not config_ref:
        return
    config = json.loads(config_ref)
    for location in ("headers", "env"):
        values = config.get(location)
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if not SENSITIVE_CONFIG_KEY.search(str(key)):
                continue
            if secret_ref is None or "{{secret_ref}}" not in str(value):
                raise HTTPException(
                    status_code=400,
                    detail="敏感 MCP 凭据必须使用 secretRef，并在配置中写 {{secret_ref}} 占位符。",
                )


def run_mcp_health_check(server: dict[str, Any]) -> dict[str, Any]:
    if server.get("serverType") == "builtin":
        return {"ok": True, "message": "内置 MCP Server 可用。", "toolCount": None}
    return asyncio.run(_health_check(server))


def run_mcp_tool_discovery(server: dict[str, Any]) -> list[dict[str, Any]]:
    return asyncio.run(_discover_tools(server))


def run_mcp_tool_call(server: dict[str, Any], tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(_call_tool(server, tool_name, arguments))


async def _health_check(server: dict[str, Any]) -> dict[str, Any]:
    try:
        async with _session(server) as session:
            await session.send_ping()
            tools = await session.list_tools()
        return {"ok": True, "message": f"连接成功，发现 {len(tools.tools)} 个工具。", "toolCount": len(tools.tools)}
    except HTTPException:
        raise
    except Exception as exc:
        return {"ok": False, "message": _connection_error(exc), "toolCount": None}


async def _discover_tools(server: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        async with _session(server) as session:
            response = await session.list_tools()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_connection_error(exc)) from exc
    return [
        {
            "toolName": tool.name,
            "description": tool.description or "",
            "inputSchema": tool.inputSchema or {},
            "outputSchema": getattr(tool, "outputSchema", None) or {},
            "riskLevel": infer_tool_risk(tool.name, tool.description or ""),
            "enabled": True,
        }
        for tool in response.tools
    ]


async def _call_tool(server: dict[str, Any], tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    try:
        async with _session(server) as session:
            result = await session.call_tool(tool_name, arguments=arguments)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_connection_error(exc)) from exc
    payload = result.model_dump(mode="json", by_alias=True, exclude_none=True)
    return {
        "ok": not bool(result.isError),
        "isError": bool(result.isError),
        "content": payload.get("content", []),
        "structuredContent": payload.get("structuredContent") or payload.get("structured_content"),
    }


@asynccontextmanager
async def _session(server: dict[str, Any]) -> AsyncIterator[Any]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client

    server_type = normalize_server_type(server.get("serverType"))
    config = parse_config_ref(server)
    if server_type == "stdio":
        env = {**os.environ, **{str(key): str(value) for key, value in (config.get("env") or {}).items()}}
        params = StdioServerParameters(
            command=str(config["command"]),
            args=[str(item) for item in config.get("args") or []],
            env=env,
            cwd=str(config["cwd"]) if config.get("cwd") else None,
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
        return
    if server_type == "streamable_http":
        headers = {str(key): str(value) for key, value in (config.get("headers") or {}).items()}
        async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=15) as http_client:
            async with streamable_http_client(str(config["url"]), http_client=http_client) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session
        return
    raise HTTPException(status_code=400, detail="内置 MCP Server 不需要建立外部连接。")


def infer_tool_risk(name: str, description: str = "") -> str:
    text = f"{name} {description}".lower()
    dangerous = ("drop", "reset", "payment", "permission", "revoke", "destroy", "支付", "权限")
    write_high = ("delete", "remove", "删除", "批量", "batch", "send", "发邮件", "push", "publish", "deploy", "merge")
    write_medium = ("create", "update", "write", "edit", "add", "set", "upload", "move", "rename")
    read_only = ("get", "list", "read", "search", "find", "fetch", "query", "status", "inspect", "view")
    if any(token in text for token in dangerous):
        return "dangerous"
    if any(token in text for token in write_high):
        return "write_high"
    if any(token in text for token in write_medium):
        return "write_medium"
    if any(token in text for token in read_only):
        return "read_only"
    return "write_medium"


def masked_config_ref(server_type: str, raw: object) -> str | None:
    if not raw or server_type == "builtin":
        return None
    try:
        config = json.loads(str(raw))
    except json.JSONDecodeError:
        return "[配置已保存]"
    if isinstance(config.get("headers"), dict):
        config["headers"] = {key: "***" for key in config["headers"]}
    if isinstance(config.get("env"), dict):
        config["env"] = {key: "***" for key in config["env"]}
    return json.dumps(config, ensure_ascii=False)


def _validate_config(server_type: str, config: dict[str, Any]) -> None:
    if server_type == "stdio":
        if not str(config.get("command") or "").strip():
            raise HTTPException(status_code=400, detail="stdio MCP 必须配置 command。")
        if config.get("args") is not None and not isinstance(config.get("args"), list):
            raise HTTPException(status_code=400, detail="stdio MCP 的 args 必须是数组。")
    elif server_type == "streamable_http":
        url = str(config.get("url") or "").strip()
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise HTTPException(status_code=400, detail="Streamable HTTP MCP 必须配置有效的 http(s) URL。")
    if config.get("headers") is not None and not isinstance(config.get("headers"), dict):
        raise HTTPException(status_code=400, detail="headers 必须是对象。")
    if config.get("env") is not None and not isinstance(config.get("env"), dict):
        raise HTTPException(status_code=400, detail="env 必须是对象。")


def _resolve_secret_reference(config: dict[str, Any], secret_ref: object) -> dict[str, Any]:
    reference = normalize_secret_ref(secret_ref)
    if reference is None:
        return config
    variable_name = SECRET_REF_PATTERN.fullmatch(reference).group(1)
    secret = os.getenv(variable_name)
    if not secret:
        raise HTTPException(status_code=503, detail="连接凭据当前不可用，请在运行环境配置对应密钥后重试。")

    def replace(value: Any) -> Any:
        if isinstance(value, str):
            return value.replace("{{secret_ref}}", secret)
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        return value

    return replace(config)


def _connection_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    return f"MCP Server 连接失败：{message[:500]}"
