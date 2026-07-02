import re

from app.core.config import settings
from app.rag.document_loader import load_document_text


PATH_PATTERN = re.compile(r"(s3://\S+|file://\S+|/[\w\-./%\u4e00-\u9fff]+)")


def parse_file_from_question(question: str) -> dict[str, object]:
    storage_path = _extract_path(question)
    if not storage_path:
        return {
            "ok": False,
            "content": "没有找到可解析的文件路径。请在问题中包含 file://、s3:// 或本地绝对路径。",
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
