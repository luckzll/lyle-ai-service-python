"""Reviewed upstream recipes. Registration does not imply installation or readiness."""
from copy import deepcopy
import os
import re

PRESETS = [
    {"id": "github", "name": "GitHub", "kind": "mcp", "description": "读取仓库、Issue 和 PR；需要你自己的 GitHub 凭据。",
     "repository": "https://github.com/github/github-mcp-server", "license": "MIT",
     "serverType": "streamable_http", "secretRef": "env:GITHUB_MCP_AUTHORIZATION", "config": {"url": "https://api.githubcopilot.com/mcp/", "headers": {"Authorization": "{{secret_ref}}", "X-MCP-Readonly": "true"}},
     "requirements": "在服务端配置 GITHUB_MCP_AUTHORIZATION（Bearer 加令牌），默认只读。"},
    {"id": "playwright", "name": "Playwright 浏览器", "kind": "mcp", "description": "网页浏览与页面检查，浏览器运行在工具服务端。",
     "repository": "https://github.com/microsoft/playwright-mcp", "license": "Apache-2.0",
     "serverType": "stdio", "config": {"command": "npx", "args": ["-y", "@playwright/mcp@0.0.83", "--headless", "--isolated"]},
     "requirements": "需要 Node.js 18+、网络和浏览器运行依赖。"},
    {"id": "fetch", "name": "网页正文 Fetch", "kind": "mcp", "description": "提取网页正文供对话引用；官方参考实现，需要部署环境评估。",
     "repository": "https://github.com/modelcontextprotocol/servers/tree/main/src/fetch", "license": "MIT",
     "serverType": "stdio", "config": {"command": "uvx", "args": ["mcp-server-fetch==2026.10.24"]},
     "requirements": "需要 uv 和出站网络；部署端限制内网访问。"},
    {"id": "ripgrep", "name": "ripgrep", "kind": "cli", "description": "在已绑定工作区快速搜索源码。", "repository": "https://github.com/BurntSushi/ripgrep", "license": "MIT / Unlicense", "requirements": "执行端需要安装 rg。"},
    {"id": "github-cli", "name": "GitHub CLI", "kind": "cli", "description": "在项目中查询 Issue 和 PR。", "repository": "https://github.com/cli/cli", "license": "MIT", "requirements": "执行端需要安装 gh 并完成授权。"},
]


def presets_for_user(user_id):
    items = deepcopy(PRESETS)
    # Transport discovery cannot establish that this host can run a browser
    # alongside the AI service. Operators can withhold the recipe until tested.
    if os.getenv("LYLE_ENABLE_PLAYWRIGHT_PRESET", "true").strip().lower() not in {"true", "1", "yes"}:
        items = [item for item in items if item["id"] != "playwright"]
    # A deployment must configure each account's own credential. Never share an
    # operator's GitHub token with every tenant that clicks a recommendation.
    suffix = str(user_id)
    if not re.fullmatch(r"[0-9]{1,20}", suffix):
        return [item for item in items if item["id"] != "github"]
    github = next(item for item in items if item["id"] == "github")
    variable = f"LYLE_GITHUB_MCP_{suffix}"
    github["secretRef"] = f"env:{variable}"
    github["requirements"] = "需要配置你个人的 GitHub 授权，默认只读。连接未配置时会保持停用。"
    return items
