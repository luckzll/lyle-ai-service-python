from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlparse
from urllib.request import url2pathname
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from docx import Document
from minio import Minio
from openpyxl import load_workbook
from pypdf import PdfReader

from app.core.config import settings

TEXT_SUFFIXES = {".txt", ".md", ".csv", ".json", ".log"}
PDF_SUFFIXES = {".pdf"}
DOCX_SUFFIXES = {".docx"}
XLSX_SUFFIXES = {".xlsx", ".xlsm"}
PPTX_SUFFIXES = {".pptx"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


@dataclass(frozen=True)
class StorageObjectRef:
    storage_path: str
    storage_provider: str | None = None
    bucket: str | None = None
    object_key: str | None = None
    original_name: str | None = None
    content_type: str | None = None
    file_size: int | None = None
    visibility: str | None = None


@dataclass(frozen=True)
class TextChunk:
    text: str
    index: int
    parent_title: str
    start_char: int
    end_char: int


def load_document_text(storage_path: str | StorageObjectRef) -> tuple[str, str]:
    ref = storage_path if isinstance(storage_path, StorageObjectRef) else StorageObjectRef(storage_path=storage_path)
    path = _resolve_storage_path(ref)
    title = ref.original_name or path.name

    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return title, path.read_text(encoding="utf-8", errors="ignore")

    if suffix in PDF_SUFFIXES:
        return title, _read_pdf(path)

    if suffix in DOCX_SUFFIXES:
        return title, _read_docx(path)

    if suffix in XLSX_SUFFIXES:
        return title, _read_xlsx(path)

    if suffix == ".xls":
        return title, _read_xls(path)

    if suffix in PPTX_SUFFIXES:
        return title, _read_pptx(path)

    if suffix in IMAGE_SUFFIXES:
        return title, _read_image_ocr(path)

    if suffix == ".doc":
        return title, (
            f"文件 {title} 已上传，但当前暂不支持旧版 .doc 格式。"
            "请转换为 .docx 后再上传。"
        )

    return title, f"文件 {title} 已上传，但当前暂不支持 {suffix or '未知'} 格式解析。"


def resolve_storage_path(storage_path: str | StorageObjectRef) -> Path:
    """Resolve a trusted server-side storage reference for turn-scoped attachment use."""
    ref = storage_path if isinstance(storage_path, StorageObjectRef) else StorageObjectRef(storage_path=storage_path)
    return _resolve_storage_path(ref)


def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    return [item.text for item in split_text_with_metadata(text, chunk_size, overlap)]


def split_text_with_metadata(text: str, chunk_size: int = 800, overlap: int = 120) -> list[TextChunk]:
    blocks = _semantic_blocks(text)
    if not blocks:
        return [TextChunk("文档内容为空。", 1, "文档内容", 0, 0)]

    chunks: list[TextChunk] = []
    current = ""
    current_title = ""
    current_start = 0
    for block in blocks:
        block_text = str(block["text"])
        block_title = str(block["heading"] or "")
        block_start = int(block["start"])
        if current and block_title != current_title:
            # A chunk must not inherit the previous section's context label.
            chunks.append(_chunk(current, len(chunks) + 1, current_title, current_start))
            current = ""
        if len(block_text) > chunk_size:
            if current:
                chunks.append(_chunk(current, len(chunks) + 1, current_title, current_start))
                current = ""
            chunks.extend(_window_split(block_text, chunk_size, overlap, len(chunks), block_title, block_start))
            continue
        # Retain exact source whitespace so offsets identify the citable text.
        candidate = text[current_start:int(block["end"])] if current else block_text
        if len(candidate) <= chunk_size:
            if not current:
                current_start = block_start
                current_title = block_title
            current = candidate
            current_title = current_title or block_title
            continue
        if current:
            chunks.append(_chunk(current, len(chunks) + 1, current_title, current_start))
        current = block_text
        current_title = block_title
        current_start = block_start
    if current:
        chunks.append(_chunk(current, len(chunks) + 1, current_title, current_start))

    return chunks


def _semantic_blocks(text: str) -> list[dict[str, object]]:
    paragraphs: list[dict[str, object]] = []
    current = False
    current_start = 0
    current_end = 0
    current_heading = ""
    active_heading = ""
    cursor = 0
    for raw_line in text.splitlines(keepends=True):
        line = raw_line.strip()
        if not line:
            if current:
                paragraphs.append({"text": text[current_start:current_end], "start": current_start, "end": current_end, "heading": current_heading})
                current = False
                current_heading = active_heading
            cursor += len(raw_line)
            continue
        if _looks_like_heading(line) and current:
            paragraphs.append({"text": text[current_start:current_end], "start": current_start, "end": current_end, "heading": current_heading})
            current = True
            current_start = cursor
            current_end = cursor + len(raw_line.rstrip("\r\n"))
            current_heading = line
            active_heading = line
            cursor += len(raw_line)
            continue
        if not current:
            current_start = cursor
            current_heading = active_heading
        if _looks_like_heading(line):
            active_heading = line
            current_heading = line
        current = True
        current_end = cursor + len(raw_line.rstrip("\r\n"))
        cursor += len(raw_line)
    if current:
        paragraphs.append({"text": text[current_start:current_end], "start": current_start, "end": current_end, "heading": current_heading})
    return [item for item in paragraphs if str(item["text"]).strip()]


def _looks_like_heading(line: str) -> bool:
    if len(line) > 80:
        return False
    heading_prefixes = ("#", "第", "一、", "二、", "三、", "四、", "五、", "1.", "2.", "3.", "4.", "5.")
    return line.startswith(heading_prefixes) or line.endswith(("：", ":"))


def _window_split(
    text: str,
    chunk_size: int,
    overlap: int,
    existing_count: int = 0,
    parent_title: str = "",
    base_start: int = 0,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(TextChunk(
            text=text[start:end],
            index=existing_count + len(chunks) + 1,
            parent_title=parent_title or "长段落",
            start_char=base_start + start,
            end_char=base_start + end,
        ))
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _chunk(text: str, index: int, parent_title: str, start_char: int) -> TextChunk:
    return TextChunk(
        text=text,
        index=index,
        parent_title=parent_title or "文档内容",
        start_char=start_char,
        end_char=start_char + len(text),
    )


def _resolve_storage_path(ref: StorageObjectRef) -> Path:
    provider = (ref.storage_provider or "").upper()
    if provider == "LOCAL" and ref.storage_path:
        parsed = urlparse(ref.storage_path)
        return Path(url2pathname(parsed.path)) if parsed.scheme == "file" else Path(ref.storage_path)
    if provider in {"MINIO", "S3"} and ref.bucket and ref.object_key:
        return _download_minio_object(ref.bucket, ref.object_key)
    if provider == "TOS":
        raise ValueError("知识库文件收到 TOS 对象引用，但当前 Python 解析器尚未配置 TOS 私有下载。")

    parsed = urlparse(ref.storage_path)
    if parsed.scheme == "file":
        return Path(url2pathname(parsed.path))
    if parsed.scheme == "s3":
        return _download_minio_object(parsed.netloc, parsed.path.lstrip("/"))
    return Path(ref.storage_path)


def _download_minio_object(bucket: str, object_name: str) -> Path:
    if not settings.minio_enabled:
        raise ValueError("收到 s3:// 文件路径，但 MINIO_ENABLED 未开启。")

    client = Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )
    suffix = Path(object_name).suffix
    temp = NamedTemporaryFile(delete=False, suffix=suffix)
    temp.close()
    client.fget_object(bucket, object_name, temp.name)
    return Path(temp.name)


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"第 {index} 页\n{text}")
    return "\n\n".join(pages) or "PDF 未解析到可用文本。"


def _read_docx(path: Path) -> str:
    document = Document(str(path))
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    table_text = []
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                table_text.append(" | ".join(cells))
    content = paragraphs + table_text
    return "\n".join(content) or "DOCX 未解析到可用文本。"


def _read_xlsx(path: Path) -> str:
    workbook = load_workbook(filename=str(path), read_only=True, data_only=True)
    lines: list[str] = []
    try:
        for worksheet in workbook.worksheets:
            lines.append(f"工作表：{worksheet.title}")
            for row in worksheet.iter_rows(values_only=True):
                values = [str(cell).strip() for cell in row if cell is not None and str(cell).strip()]
                if values:
                    lines.append(" | ".join(values))
    finally:
        workbook.close()
    return "\n".join(lines) or "XLSX 未解析到可用文本。"


def _read_xls(path: Path) -> str:
    try:
        import xlrd
    except Exception:
        return "旧版 XLS 解析需要安装可选依赖 xlrd。当前环境未启用 XLS 解析。"

    workbook = xlrd.open_workbook(str(path))
    lines: list[str] = []
    for sheet in workbook.sheets():
        lines.append(f"工作表：{sheet.name}")
        for row_index in range(sheet.nrows):
            values = [
                str(sheet.cell_value(row_index, col_index)).strip()
                for col_index in range(sheet.ncols)
                if str(sheet.cell_value(row_index, col_index)).strip()
            ]
            if values:
                lines.append(" | ".join(values))
    return "\n".join(lines) or "XLS 未解析到可用文本。"


def _read_pptx(path: Path) -> str:
    try:
        with ZipFile(path) as archive:
            slide_entries = sorted(
                (
                    item
                    for item in archive.infolist()
                    if item.filename.startswith("ppt/slides/slide") and item.filename.endswith(".xml")
                ),
                key=_pptx_slide_number,
            )
            slides: list[str] = []
            for index, entry in enumerate(slide_entries, start=1):
                if entry.file_size > 8 * 1024 * 1024:
                    raise ValueError(f"PPTX 第 {index} 页内容过大，无法安全解析。")
                root = ElementTree.fromstring(archive.read(entry))
                text_items = [
                    str(node.text).strip()
                    for node in root.iter()
                    if node.tag.endswith("}t") and node.text and str(node.text).strip()
                ]
                if text_items:
                    slides.append(f"第 {index} 页\n" + "\n".join(text_items))
    except (BadZipFile, ElementTree.ParseError) as exc:
        raise ValueError("PPTX 文件已损坏或不是有效的 PowerPoint 文件。") from exc
    return "\n\n".join(slides) or "PPTX 未解析到可用文本。"


def _pptx_slide_number(entry) -> int:
    stem = Path(entry.filename).stem
    suffix = stem.removeprefix("slide")
    return int(suffix) if suffix.isdigit() else 0


def _read_image_ocr(path: Path) -> str:
    try:
        import pytesseract
        from PIL import Image
    except Exception:
        return (
            "图片 OCR 需要安装可选依赖 pillow、pytesseract，并在系统安装 tesseract。"
            "当前环境未启用 OCR。"
        )

    try:
        text = pytesseract.image_to_string(Image.open(path), lang="chi_sim+eng")
    except Exception as exc:
        return f"图片 OCR 解析失败：{exc}"
    return text.strip() or "图片 OCR 未识别到文本。"
