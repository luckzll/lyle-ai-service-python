import logging
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

from app.core.config import settings
from app.rag.document_loader import load_document_text
from app.rag.store import storage_object_belongs_to_user

logger = logging.getLogger(__name__)

PATH_PATTERN = re.compile(r"(s3://\S+|file://\S+|[A-Za-z]:[\\/]\S+|/[\w\-./%\u4e00-\u9fff]+)")


def parse_file_from_question(question: str, user_id: str | None = None) -> dict[str, object]:
    storage_path = _extract_path(question)
    if not storage_path:
        return {
            "ok": False,
            "content": "没有找到可解析的文件路径。请在问题中包含 file://、s3:// 或本地绝对路径。",
        }

    return parse_file(storage_path, user_id)


def parse_file(storage_path: str, user_id: str | None = None) -> dict[str, object]:
    if not _is_allowed_storage_path(storage_path, user_id):
        logger.warning("file_parse rejected path outside whitelist: %s", storage_path)
        return {
            "ok": False,
            "content": "该文件路径不在允许解析的目录白名单内，或对象不属于当前用户，已拒绝解析。",
        }

    title, text = load_document_text(storage_path)
    text = text[: settings.tool_file_parse_max_chars]
    return {
        "ok": True,
        "content": f"文件 {title} 解析完成，前 {len(text)} 字内容：\n{text}",
        "file": {
            "title": title,
            "storagePath": storage_path,
            "textPreview": text,
        },
    }


def _extract_path(question: str) -> str | None:
    match = PATH_PATTERN.search(question)
    if not match:
        return None
    return match.group(1).rstrip("。），,)")


def _is_allowed_storage_path(storage_path: str, user_id: str | None = None) -> bool:
    """用户问题里的路径是不可信输入：本地路径走白名单，s3:// 必须属于当前用户已入库对象。"""
    parsed = urlparse(storage_path)
    if parsed.scheme == "s3":
        return bool(user_id and storage_object_belongs_to_user(user_id, storage_path))

    candidate = Path(unquote(parsed.path)) if parsed.scheme == "file" else Path(storage_path)
    try:
        # resolve() 会展开符号链接和 ..，防止通过链接或相对跳转逃出白名单目录。
        resolved = candidate.expanduser().resolve()
    except OSError:
        return False

    for raw_root in settings.tool_file_parse_allowed_dirs.split(","):
        raw_root = raw_root.strip()
        if not raw_root:
            continue
        try:
            root = Path(raw_root).expanduser().resolve()
        except OSError:
            continue
        if resolved.is_relative_to(root):
            return True
    return False
