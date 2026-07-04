from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

from app.connectors.base import ConnectorInfo, tool_result
from app.core.config import settings
from app.tools.operation_confirmation import create_pending_operation


class GitHubConnector:
    name = "github"
    label = "GitHub"
    capabilities = ["list", "get", "search", "preview_create"]

    def status(self) -> ConnectorInfo:
        configured = bool(settings.connector_github_token.strip())
        return ConnectorInfo(
            name=self.name,
            label=self.label,
            status="connected" if configured else "needs_auth",
            capabilities=self.capabilities,
            message="已配置 GitHub 只读访问。" if configured else "未配置 CONNECTOR_GITHUB_TOKEN，仅可查看连接器状态。",
        )

    def list(self, query: dict[str, Any]) -> dict[str, Any]:
        repo = str(query.get("repo") or "").strip()
        if not repo:
            return tool_result(connector=self.name, ok=False, content="请提供 repo，例如 owner/name。", error_code="PARAM_MISSING")
        state = str(query.get("state") or "open")
        path = f"/repos/{repo}/issues?state={urllib.parse.quote(state)}&per_page={_limit(query.get('limit'), 20)}"
        response = _request(path)
        if not response["ok"]:
            return response
        items = [_issue_item(item) for item in response["data"] if isinstance(item, dict)]
        return tool_result(
            connector=self.name,
            ok=True,
            content=f"读取 GitHub issue/PR {len(items)} 条。",
            items=items,
            sources=[{"type": "github", "ref": item["url"], "title": item["title"]} for item in items],
        )

    def get(self, resource_id: str) -> dict[str, Any]:
        parsed = _parse_issue_ref(resource_id)
        if not parsed:
            return tool_result(connector=self.name, ok=False, content="资源 ID 格式应为 owner/repo#number。", error_code="PARAM_MISSING")
        owner_repo, number = parsed
        response = _request(f"/repos/{owner_repo}/issues/{number}")
        if not response["ok"]:
            return response
        item = _issue_item(response["data"])
        item["body"] = str(response["data"].get("body") or "")[:1200]
        return tool_result(
            connector=self.name,
            ok=True,
            content=f"已读取 GitHub 条目：{item['title']}",
            items=[item],
            sources=[{"type": "github", "ref": item["url"], "title": item["title"]}],
        )

    def search(self, query: dict[str, Any]) -> dict[str, Any]:
        q = str(query.get("q") or query.get("query") or "").strip()
        repo = str(query.get("repo") or "").strip()
        if not q:
            return tool_result(connector=self.name, ok=False, content="请提供搜索关键词。", error_code="PARAM_MISSING")
        search_query = q if not repo else f"{q} repo:{repo}"
        path = f"/search/issues?q={urllib.parse.quote(search_query)}&per_page={_limit(query.get('limit'), 20)}"
        response = _request(path)
        if not response["ok"]:
            return response
        raw_items = response["data"].get("items") if isinstance(response["data"], dict) else []
        items = [_issue_item(item) for item in raw_items if isinstance(item, dict)]
        return tool_result(
            connector=self.name,
            ok=True,
            content=f"GitHub 搜索命中 {len(items)} 条。",
            items=items,
            sources=[{"type": "github", "ref": item["url"], "title": item["title"]} for item in items],
        )

    def preview_create(self, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        repo = str(payload.get("repo") or "").strip()
        title = str(payload.get("title") or "").strip()
        if not repo or not title:
            return tool_result(connector=self.name, ok=False, content="请提供 repo 和 title。", error_code="PARAM_MISSING")
        return create_pending_operation(
            user_id=user_id,
            conversation_id=str(payload.get("conversationId") or ""),
            tool_name=self.name,
            operation_type="connector.github.issue.create",
            summary=f"准备在 {repo} 创建 GitHub issue：{title}",
            payload={"repo": repo, "title": title, "body": str(payload.get("body") or "")},
            source_text=str(payload.get("sourceText") or title),
            risk_level="external_write",
        )


def _request(path: str) -> dict[str, Any]:
    token = settings.connector_github_token.strip()
    if not token:
        return tool_result(connector="github", ok=False, content="GitHub token 未配置。", error_code="AUTH_REQUIRED")
    url = settings.connector_github_api_base_url.rstrip("/") + path
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "Lyle-V3-Connector",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
            return {"ok": True, "data": data}
    except Exception as exc:
        return tool_result(
            connector="github",
            ok=False,
            content=f"GitHub 读取失败：{exc}",
            error_code="CONNECTOR_ERROR",
            retryable=True,
        )


def execute_github_issue_create(payload: dict[str, Any]) -> dict[str, Any]:
    repo = str(payload.get("repo") or "").strip()
    title = str(payload.get("title") or "").strip()
    body = str(payload.get("body") or "")
    if not repo or not title:
        return tool_result(connector="github", ok=False, content="请提供 repo 和 title。", error_code="PARAM_MISSING")
    response = _request_json("POST", f"/repos/{repo}/issues", {"title": title, "body": body})
    if not response["ok"]:
        return response
    item = _issue_item(response["data"])
    return tool_result(
        connector="github",
        ok=True,
        content=f"已创建 GitHub issue：{item['title']}",
        items=[item],
        sources=[{"type": "github", "ref": item["url"], "title": item["title"]}],
    )


def _request_json(method: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    token = settings.connector_github_token.strip()
    if not token:
        return tool_result(connector="github", ok=False, content="GitHub token 未配置。", error_code="AUTH_REQUIRED")
    url = settings.connector_github_api_base_url.rstrip("/") + path
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "Lyle-V3-Connector",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
            return {"ok": True, "data": data}
    except Exception as exc:
        return tool_result(
            connector="github",
            ok=False,
            content=f"GitHub 写入失败：{exc}",
            error_code="CONNECTOR_ERROR",
            retryable=True,
        )


def _issue_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(item.get("id") or item.get("number") or ""),
        "resourceType": "pull_request" if item.get("pull_request") else "issue",
        "number": item.get("number"),
        "title": item.get("title") or "",
        "state": item.get("state") or "",
        "url": item.get("html_url") or item.get("url") or "",
        "updatedAt": item.get("updated_at"),
        "author": (item.get("user") or {}).get("login") if isinstance(item.get("user"), dict) else None,
    }


def _parse_issue_ref(resource_id: str) -> tuple[str, str] | None:
    if "#" not in resource_id:
        return None
    repo, number = resource_id.split("#", 1)
    if repo.count("/") != 1 or not number.strip().isdigit():
        return None
    return repo.strip(), number.strip()


def _limit(value: Any, default: int) -> int:
    try:
        return max(1, min(int(value), 100))
    except (TypeError, ValueError):
        return default
