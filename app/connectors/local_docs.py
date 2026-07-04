from __future__ import annotations

from pathlib import Path
from typing import Any

from app.connectors.base import ConnectorInfo, tool_result
from app.core.config import settings
from app.tools.operation_confirmation import create_pending_operation


class LocalDocsConnector:
    name = "local_docs"
    label = "本地文档"
    capabilities = ["list", "get", "search", "preview_create"]
    extensions = {".md", ".txt"}

    def status(self) -> ConnectorInfo:
        roots = _allowed_roots()
        available = any(root.exists() for root in roots)
        return ConnectorInfo(
            name=self.name,
            label=self.label,
            status="connected" if available else "unavailable",
            capabilities=self.capabilities,
            message="可读取本地 Markdown/TXT 文档。" if available else "未找到可读取的本地文档目录。",
        )

    def list(self, query: dict[str, Any]) -> dict[str, Any]:
        limit = _limit(query.get("limit"), 50)
        files = _iter_files(limit=limit)
        items = [_file_item(path) for path in files]
        return tool_result(
            connector=self.name,
            ok=True,
            content=f"找到 {len(items)} 个本地文档。",
            items=items,
            sources=[{"type": "local_path", "ref": item["id"], "title": item["title"]} for item in items],
        )

    def get(self, resource_id: str) -> dict[str, Any]:
        path = _resolve_resource(resource_id)
        if path is None:
            return tool_result(connector=self.name, ok=False, content="文档不存在或不在允许目录内。", error_code="NOT_FOUND")
        text = _read_text(path)
        item = {**_file_item(path), "content": text[: settings.connector_local_docs_max_chars]}
        return tool_result(
            connector=self.name,
            ok=True,
            content=f"已读取文档：{path.name}",
            items=[item],
            sources=[{"type": "local_path", "ref": item["id"], "title": item["title"]}],
        )

    def search(self, query: dict[str, Any]) -> dict[str, Any]:
        keyword = str(query.get("q") or query.get("query") or "").strip().lower()
        if not keyword:
            return self.list(query)
        limit = _limit(query.get("limit"), 20)
        items: list[dict[str, Any]] = []
        for path in _iter_files(limit=settings.connector_local_docs_max_files):
            title_hit = keyword in path.name.lower()
            text = _read_text(path)
            body_hit = keyword in text.lower()
            if not title_hit and not body_hit:
                continue
            snippet = _snippet(text, keyword)
            items.append({**_file_item(path), "snippet": snippet, "score": 1.0 if title_hit else 0.7})
            if len(items) >= limit:
                break
        return tool_result(
            connector=self.name,
            ok=True,
            content=f"本地文档搜索命中 {len(items)} 条。",
            items=items,
            sources=[{"type": "local_path", "ref": item["id"], "title": item["title"]} for item in items],
        )

    def preview_create(self, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        title = str(payload.get("title") or "新建本地文档").strip()
        body = str(payload.get("content") or "").strip()
        return create_pending_operation(
            user_id=user_id,
            conversation_id=str(payload.get("conversationId") or ""),
            tool_name=self.name,
            operation_type="connector.local_docs.create",
            summary=f"准备新建本地文档：{title}",
            payload={"title": title, "content": body},
            source_text=str(payload.get("sourceText") or title),
            risk_level="write",
        )


def execute_local_doc_create(payload: dict[str, Any]) -> dict[str, Any]:
    title = str(payload.get("title") or "新建本地文档").strip()
    body = str(payload.get("content") or "").strip()
    roots = _allowed_roots()
    if not roots:
        return tool_result(connector="local_docs", ok=False, content="未配置本地文档目录。", error_code="CONFIG_ERROR")
    target_dir = roots[0] / ".local" / "connector-docs"
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = _safe_filename(title) + ".md"
    target = (target_dir / filename).resolve()
    if not _is_allowed(target):
        return tool_result(connector="local_docs", ok=False, content="目标路径不在允许目录内。", error_code="SECURITY_ERROR")
    if target.exists():
        return tool_result(connector="local_docs", ok=False, content="同名本地文档已存在。", error_code="CONFLICT")
    target.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
    item = _file_item(target)
    return tool_result(
        connector="local_docs",
        ok=True,
        content=f"已新建本地文档：{target.name}",
        items=[item],
        sources=[{"type": "local_path", "ref": item["id"], "title": item["title"]}],
    )


def _allowed_roots() -> list[Path]:
    roots = []
    for raw in settings.connector_local_docs_roots.split(","):
        text = raw.strip()
        if not text:
            continue
        roots.append(Path(text).expanduser().resolve())
    return roots


def _iter_files(limit: int) -> list[Path]:
    files: list[Path] = []
    for root in _allowed_roots():
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if len(files) >= limit:
                return files
            if path.is_file() and path.suffix.lower() in LocalDocsConnector.extensions and _is_allowed(path):
                files.append(path)
    return files


def _resolve_resource(resource_id: str) -> Path | None:
    try:
        path = Path(resource_id).expanduser().resolve()
    except (OSError, RuntimeError):
        return None
    if not path.is_file() or path.suffix.lower() not in LocalDocsConnector.extensions:
        return None
    return path if _is_allowed(path) else None


def _is_allowed(path: Path) -> bool:
    try:
        resolved = path.resolve()
        return any(resolved.is_relative_to(root) for root in _allowed_roots())
    except (OSError, RuntimeError):
        return False


def _file_item(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "id": str(path),
        "title": path.name,
        "resourceType": "file",
        "path": str(path),
        "updatedAt": stat.st_mtime,
        "size": stat.st_size,
    }


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _snippet(text: str, keyword: str) -> str:
    lower = text.lower()
    index = lower.find(keyword)
    if index < 0:
        return text[:240]
    start = max(0, index - 80)
    end = min(len(text), index + 220)
    return text[start:end].replace("\n", " ").strip()


def _limit(value: Any, default: int) -> int:
    try:
        return max(1, min(int(value), 100))
    except (TypeError, ValueError):
        return default


def _safe_filename(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value.strip())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return (cleaned or "local-doc")[:80]
