from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import unquote, urlparse

from docx import Document
from minio import Minio
from openpyxl import load_workbook
from pypdf import PdfReader

from app.core.config import settings

TEXT_SUFFIXES = {".txt", ".md", ".csv", ".json", ".log"}
PDF_SUFFIXES = {".pdf"}
DOCX_SUFFIXES = {".docx"}
XLSX_SUFFIXES = {".xlsx", ".xlsm"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def load_document_text(storage_path: str) -> tuple[str, str]:
    path = _resolve_storage_path(storage_path)
    title = path.name

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

    if suffix in IMAGE_SUFFIXES:
        return title, _read_image_ocr(path)

    if suffix == ".doc":
        return title, (
            f"文件 {title} 已上传，但当前暂不支持旧版 .doc 格式。"
            "请转换为 .docx 后再上传。"
        )

    return title, f"文件 {title} 已上传，但当前暂不支持 {suffix or '未知'} 格式解析。"


def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not normalized:
        return ["文档内容为空。"]

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunks.append(normalized[start:end])
        if end == len(normalized):
            break
        start = max(end - overlap, start + 1)

    return chunks


def _resolve_storage_path(storage_path: str) -> Path:
    parsed = urlparse(storage_path)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if parsed.scheme == "s3":
        return _download_minio_object(parsed.netloc, parsed.path.lstrip("/"))
    return Path(storage_path)


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
