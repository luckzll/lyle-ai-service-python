"""Memory-only local document parsing, without attachment, RAG, memory or model calls."""
import base64
import binascii
import json
from pathlib import PurePath

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from app.rag.document_loader import DocumentReadError, load_local_document_text

router = APIRouter()
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_ENCODED_CHARS = 6_990_508
MAX_BODY_BYTES = MAX_ENCODED_CHARS + 4_096
FORMATS = {"pdf", "docx", "xlsx", "xlsm", "xls", "pptx"}


def extract_document(payload: object) -> dict:
    if not isinstance(payload, dict) or set(payload) != {"name", "data"}:
        raise HTTPException(400, "请提供文档名称和 Base64 文件内容。")
    name, encoded = payload["name"], payload["data"]
    if (not isinstance(name, str) or not name.strip() or len(name) > 128 or name.startswith(".")
            or any(ord(char) < 32 or ord(char) == 127 for char in name) or any(char in name for char in "/\\:")):
        raise HTTPException(400, "文档名称无效，请重新选择文件。")
    format = PurePath(name).suffix.lower().removeprefix(".")
    if format not in FORMATS:
        raise HTTPException(400, "仅支持 PDF、DOCX、XLSX、XLSM、XLS 和 PPTX；DOC 和 PPT 请转换为 DOCX 和 PPTX。")
    if not isinstance(encoded, str) or not encoded:
        raise HTTPException(400, "文件内容无效，请重新选择文档。")
    if len(encoded) > MAX_ENCODED_CHARS:
        raise HTTPException(413, "单个文档不能超过 5 MB。")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(400, "文件内容不是有效的 Base64 数据。") from None
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(413, "单个文档不能超过 5 MB。")
    if base64.b64encode(data).decode("ascii") != encoded:
        raise HTTPException(400, "文件内容不是有效的 Base64 数据。")
    try:
        return load_local_document_text(data, format)
    except DocumentReadError as exc:
        raise HTTPException(413 if exc.code == "FILE_TOO_LARGE" else 400, str(exc)) from None
    except Exception:
        # Parser diagnostics can contain document text, file paths or provider data.
        raise HTTPException(400, "文档无法解析，可能已损坏、加密或与扩展名不一致，请重新导出文件。") from None


@router.post("/local-workspace/document/extract")
async def extract(request: Request):
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_BODY_BYTES:
            raise HTTPException(413, "单个文档不能超过 5 MB。")
        body.extend(chunk)
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeError, RecursionError):
        raise HTTPException(400, "文档请求格式无效，请重新选择文件。") from None
    return await run_in_threadpool(extract_document, payload)
