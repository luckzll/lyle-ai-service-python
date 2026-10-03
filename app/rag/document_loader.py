from dataclasses import dataclass
from io import BytesIO
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

TEXT_SUFFIXES = {".txt", ".md", ".csv", ".json", ".log", *(
    "." + item for item in "ts tsx js jsx mjs cjs vue svelte py java kt kts go rs c h cpp hpp cs php rb swift sh bash zsh ps1 sql yaml yml toml ini cfg xml html htm css scss sass less graphql gql proto properties gradle scala r dart ipynb".split()
)}
TEXT_FILENAMES = {"Dockerfile", "Makefile", "Containerfile", "Jenkinsfile", "Procfile", "README", "LICENSE",
                  ".gitignore", ".dockerignore", ".editorconfig"}
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


class DocumentReadError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def is_supported_document_name(name: str) -> bool:
    suffix = Path(name).suffix.lower()
    return (Path(name).name in TEXT_FILENAMES or suffix in
            TEXT_SUFFIXES | PDF_SUFFIXES | DOCX_SUFFIXES | XLSX_SUFFIXES | PPTX_SUFFIXES | IMAGE_SUFFIXES | {".xls"})


def load_document_text(storage_path: str | StorageObjectRef, *, max_bytes: int | None = None) -> tuple[str, str]:
    ref = storage_path if isinstance(storage_path, StorageObjectRef) else StorageObjectRef(storage_path=storage_path)
    path = _resolve_storage_path(ref)
    title = ref.original_name or path.name

    # UUID storage objects may have no suffix; the server-validated upload name
    # defines the format, including exact names such as Dockerfile.
    suffix = Path(title).suffix.lower()
    if max_bytes is not None and path.stat().st_size > max_bytes:
        raise DocumentReadError("FILE_TOO_LARGE", "文件超过允许读取的大小。")
    if suffix in TEXT_SUFFIXES or Path(title).name in TEXT_FILENAMES:
        with path.open("rb") as source:
            data = source.read(max_bytes + 1) if max_bytes is not None else source.read()
        if max_bytes is not None and len(data) > max_bytes:
            raise DocumentReadError("FILE_TOO_LARGE", "文件超过允许读取的大小。")
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DocumentReadError("INVALID_UTF8", "文本文件不是有效的 UTF-8 编码。") from exc
        if any(ord(char) < 32 and char not in "\t\n\r" for char in text) or "\x7f" in text:
            raise DocumentReadError("BINARY_CONTENT", "文件包含二进制控制字符，无法按文本读取。")
        return title, text

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

    raise DocumentReadError("UNSUPPORTED_FORMAT", "当前文件格式不支持解析；旧版 DOC 请转换为 DOCX。")


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


# Separate from storage-backed RAG ingestion: caller-supplied bytes never acquire
# a storage reference and cannot create attachments, embeddings or memory.
LOCAL_DOCUMENT_MAX_TEXT = 120_000
LOCAL_DOCUMENT_MAX_ZIP_BYTES = 20 * 1024 * 1024
LOCAL_DOCUMENT_MAX_ZIP_ENTRY = 8 * 1024 * 1024
LOCAL_DOCUMENT_MAX_ZIP_ENTRIES = 2_048
LOCAL_DOCUMENT_MAX_PAGES = 100
LOCAL_DOCUMENT_MAX_SHEETS = 20
LOCAL_DOCUMENT_MAX_ROWS = 5_000
LOCAL_DOCUMENT_MAX_COLUMNS = 200
LOCAL_DOCUMENT_MAX_BLOCKS = 5_000


class _LocalText:
    def __init__(self):
        self.parts: list[str] = []
        self.length = 0
        self.truncated = False
        self.warnings: list[str] = []

    def add(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return True
        available = LOCAL_DOCUMENT_MAX_TEXT - self.length
        separator = "\n" if self.parts else ""
        value = separator + text
        if len(value) > available:
            self.parts.append(value[:available])
            self.length += available
            self.truncated = True
            return False
        self.parts.append(value)
        self.length += len(value)
        return True

    def result(self, format: str) -> dict:
        text = "".join(self.parts)
        warnings = list(self.warnings)
        if self.truncated:
            warnings.append("文档内容较多，已返回有界的部分文字。")
        if not text:
            warnings.append("未解析到可用文字；扫描版 PDF 需要先识别文字或转换为含文字的文件。"
                            if format == "pdf" else "未解析到可用文字，文件可能为空或只包含图片。")
        return {"text": text, "format": format, "truncated": self.truncated, "warnings": warnings}


def _local_zip(data: bytes, format: str) -> ZipFile:
    archive = ZipFile(BytesIO(data))
    try:
        entries = archive.infolist()
        if len(entries) > LOCAL_DOCUMENT_MAX_ZIP_ENTRIES:
            raise DocumentReadError("FILE_TOO_LARGE", "文档包含过多内容，无法安全解析。")
        total = 0
        names: set[str] = set()
        for entry in entries:
            name = entry.filename
            if (name in names or name.startswith(("/", "\\")) or "\\" in name or ".." in name.split("/")
                    or entry.flag_bits & 1):
                raise DocumentReadError("INVALID_DOCUMENT", "文档结构无效或已加密，请重新导出文件。")
            names.add(name)
            total += entry.file_size
            if (entry.file_size > LOCAL_DOCUMENT_MAX_ZIP_ENTRY or total > LOCAL_DOCUMENT_MAX_ZIP_BYTES
                    or entry.file_size > max(1_000_000, entry.compress_size * 200)):
                raise DocumentReadError("FILE_TOO_LARGE", "文档展开后过大，无法安全解析。")
            if name.endswith(".xml") or name.endswith(".rels"):
                # No external entities/DTDs are accepted, including unused OOXML parts.
                with archive.open(entry) as source:
                    content = source.read(LOCAL_DOCUMENT_MAX_ZIP_ENTRY + 1)
                if len(content) > LOCAL_DOCUMENT_MAX_ZIP_ENTRY:
                    raise DocumentReadError("FILE_TOO_LARGE", "文档展开后过大，无法安全解析。")
                # XML declarations may use UTF-16/32; remove their NUL padding
                # when detecting forbidden DTD/entity declarations.
                upper = content.replace(b"\x00", b"").upper()
                if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
                    raise DocumentReadError("INVALID_DOCUMENT", "文档包含不支持的 XML 结构，请重新导出文件。")
        required = {"docx": "word/document.xml", "xlsx": "xl/workbook.xml", "xlsm": "xl/workbook.xml",
                    "pptx": "ppt/presentation.xml"}[format]
        if "[Content_Types].xml" not in names or required not in names:
            raise DocumentReadError("INVALID_DOCUMENT", "文件内容与文档格式不一致，请重新导出文件。")
        return archive
    except Exception:
        archive.close()
        raise


def _local_pdf(data: bytes, output: _LocalText) -> None:
    from pypdf.generic import ArrayObject
    if not data.startswith(b"%PDF-"):
        raise DocumentReadError("INVALID_DOCUMENT", "文件不是有效的 PDF 文档。")
    reader = PdfReader(BytesIO(data))
    if reader.is_encrypted:
        raise DocumentReadError("INVALID_DOCUMENT", "PDF 已加密，请先解密后再读取。")
    count = len(reader.pages)
    output.truncated = count > LOCAL_DOCUMENT_MAX_PAGES
    expanded = 0
    seen_streams: set[int] = set()

    def check_stream(stream):
        nonlocal expanded
        stream = stream.get_object()
        if id(stream) in seen_streams:
            return
        seen_streams.add(id(stream))
        if len(seen_streams) > 256:
            raise DocumentReadError("FILE_TOO_LARGE", "PDF 文字结构过多，无法安全解析。")
        size = len(stream.get_data())
        expanded += size
        if size > LOCAL_DOCUMENT_MAX_ZIP_ENTRY or expanded > LOCAL_DOCUMENT_MAX_ZIP_BYTES:
            raise DocumentReadError("FILE_TOO_LARGE", "PDF 文字内容过大，无法安全解析。")

    def check_resources(resources, depth=0):
        if not resources:
            return
        if depth > 8:
            raise DocumentReadError("FILE_TOO_LARGE", "PDF 文字结构过多，无法安全解析。")
        resources = resources.get_object()
        objects = resources.get("/XObject")
        if objects:
            objects = objects.get_object()
            if len(objects) > 64:
                raise DocumentReadError("FILE_TOO_LARGE", "PDF 文字结构过多，无法安全解析。")
            for reference in objects.values():
                obj = reference.get_object()
                if obj.get("/Subtype") == "/Form" and id(obj) not in seen_streams:
                    check_stream(obj)
                    check_resources(obj.get("/Resources"), depth + 1)
        fonts = resources.get("/Font")
        if fonts:
            fonts = fonts.get_object()
            if len(fonts) > 64:
                raise DocumentReadError("FILE_TOO_LARGE", "PDF 字体结构过多，无法安全解析。")
            for reference in fonts.values():
                mapping = reference.get_object().get("/ToUnicode")
                if mapping:
                    check_stream(mapping)

    for index in range(min(count, LOCAL_DOCUMENT_MAX_PAGES)):
        page = reader.pages[index]
        # pypdf also enforces stream decompression limits. Bound text content per
        # page and across this extraction before building the text representation.
        content = page.get("/Contents")
        if content is not None:
            content = content.get_object()
            streams = content if isinstance(content, ArrayObject) else [content]
            if len(streams) > 64:
                raise DocumentReadError("FILE_TOO_LARGE", "PDF 文字结构过多，无法安全解析。")
            for stream in streams:
                check_stream(stream)
        check_resources(page.get("/Resources"))
        text = page.extract_text() or ""
        if text.strip() and not output.add(f"第 {index + 1} 页\n{text}"):
            break


def _local_docx(data: bytes, output: _LocalText) -> None:
    from docx.text.paragraph import Paragraph
    document = Document(BytesIO(data))
    blocks = 0
    # Keep tables beside their surrounding paragraphs instead of collecting all
    # tables at the end, which can change the meaning of a document.
    for item in document.iter_inner_content():
        if isinstance(item, Paragraph):
            blocks += 1
            if blocks > LOCAL_DOCUMENT_MAX_BLOCKS:
                output.truncated = True
                return
            if not output.add(item.text):
                return
        else:
            for row in item.rows:
                blocks += 1
                if blocks > LOCAL_DOCUMENT_MAX_BLOCKS:
                    output.truncated = True
                    return
                if len(row.cells) > LOCAL_DOCUMENT_MAX_COLUMNS:
                    output.truncated = True
                cells = [cell.text.strip() for cell in row.cells[:LOCAL_DOCUMENT_MAX_COLUMNS] if cell.text.strip()]
                if cells and not output.add(" | ".join(cells)):
                    return


def _local_spreadsheet(data: bytes, format: str, output: _LocalText) -> None:
    if format == "xls":
        import xlrd
        output.warnings.append("XLS 仅读取文件保存的单元格值，不执行宏或重新计算公式。")
        if not data.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
            raise DocumentReadError("INVALID_DOCUMENT", "文件不是有效的 XLS 工作簿。")
        workbook = xlrd.open_workbook(file_contents=data, on_demand=True)
        try:
            output.truncated = workbook.nsheets > LOCAL_DOCUMENT_MAX_SHEETS
            remaining_rows = LOCAL_DOCUMENT_MAX_ROWS
            for index in range(min(workbook.nsheets, LOCAL_DOCUMENT_MAX_SHEETS)):
                sheet = workbook.sheet_by_index(index)
                rows = min(sheet.nrows, remaining_rows)
                output.truncated |= sheet.nrows > rows or sheet.ncols > LOCAL_DOCUMENT_MAX_COLUMNS
                heading = f"工作表：{sheet.name}\n"
                for row_index in range(rows):
                    values = [str(sheet.cell_value(row_index, col)).strip() for col in range(min(sheet.ncols, LOCAL_DOCUMENT_MAX_COLUMNS))]
                    values = [value for value in values if value]
                    if values:
                        if not output.add(heading + " | ".join(values)):
                            return
                        heading = ""
                remaining_rows -= rows
                workbook.unload_sheet(index)
                if remaining_rows <= 0:
                    output.truncated |= index + 1 < workbook.nsheets
                    break
        finally:
            workbook.release_resources()
    else:
        # Preserve formula expressions when cached results are absent. Never
        # execute formulas/macros or resolve external workbook links.
        output.warnings.append("仅读取单元格内容和公式表达式，不执行宏或计算公式。")
        workbook = load_workbook(BytesIO(data), read_only=True, data_only=False, keep_links=False)
        try:
            output.truncated = len(workbook.worksheets) > LOCAL_DOCUMENT_MAX_SHEETS
            remaining_rows = LOCAL_DOCUMENT_MAX_ROWS
            for index, sheet in enumerate(workbook.worksheets[:LOCAL_DOCUMENT_MAX_SHEETS]):
                rows = min(sheet.max_row if sheet.max_row is not None else remaining_rows, remaining_rows)
                columns = min(sheet.max_column if sheet.max_column is not None else LOCAL_DOCUMENT_MAX_COLUMNS, LOCAL_DOCUMENT_MAX_COLUMNS)
                output.truncated |= sheet.max_row is None or sheet.max_column is None
                output.truncated |= (sheet.max_row or 0) > rows or (sheet.max_column or 0) > columns
                heading = f"工作表：{sheet.title}\n"
                if rows and columns:
                    for row in sheet.iter_rows(min_row=1, max_row=rows, min_col=1, max_col=columns, values_only=True):
                        values = [str(cell).strip() for cell in row if cell is not None and str(cell).strip()]
                        if values:
                            if not output.add(heading + " | ".join(values)):
                                return
                            heading = ""
                remaining_rows -= rows
                if remaining_rows <= 0:
                    output.truncated |= index + 1 < len(workbook.worksheets)
                    break
        finally:
            workbook.close()


def load_local_document_text(data: bytes, format: str) -> dict:
    """Bounded, memory-only document extraction; does not resolve storage or ingest."""
    if len(data) > 5 * 1024 * 1024:
        raise DocumentReadError("FILE_TOO_LARGE", "单个文档不能超过 5 MB。")
    output = _LocalText()
    if format in {"docx", "xlsx", "xlsm", "pptx"}:
        with _local_zip(data, format) as archive:
            if any(entry.file_size and ("/media/" in entry.filename or "/charts/" in entry.filename)
                   for entry in archive.infolist()):
                output.warnings.append("仅提取文档文字和单元格，未识别内嵌图片或图表内容。")
            if format == "pptx":
                # Follow the presentation order, which can differ from slide filenames.
                rel_namespace = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                slide_namespace = "http://schemas.openxmlformats.org/presentationml/2006/main"
                presentation = ElementTree.fromstring(archive.read("ppt/presentation.xml"))
                relationships = ElementTree.fromstring(archive.read("ppt/_rels/presentation.xml.rels"))
                references = {node.get("Id"): node for node in relationships}
                slides = []
                for node in presentation.iter(f"{{{slide_namespace}}}sldId"):
                    reference = references.get(node.get(f"{{{rel_namespace}}}id"))
                    if reference is None or reference.get("TargetMode") == "External" or not (reference.get("Type") or "").endswith("/slide"):
                        raise DocumentReadError("INVALID_DOCUMENT", "PPTX 页面结构无效，请重新导出文件。")
                    target = reference.get("Target") or ""
                    path = target[1:] if target.startswith("/ppt/") else "ppt/" + target
                    if not path.startswith("ppt/slides/") or ".." in path.split("/") or "\\" in path:
                        raise DocumentReadError("INVALID_DOCUMENT", "PPTX 页面结构无效，请重新导出文件。")
                    slides.append(archive.getinfo(path))
                output.truncated = len(slides) > LOCAL_DOCUMENT_MAX_PAGES
                for index, entry in enumerate(slides[:LOCAL_DOCUMENT_MAX_PAGES], 1):
                    root = ElementTree.fromstring(archive.read(entry))
                    items = [node.text.strip() for node in root.iter() if node.tag.endswith("}t") and node.text and node.text.strip()]
                    if items and not output.add(f"第 {index} 页\n" + "\n".join(items)):
                        break
            elif format == "docx":
                _local_docx(data, output)
            else:
                _local_spreadsheet(data, format, output)
    elif format == "pdf":
        _local_pdf(data, output)
    elif format == "xls":
        _local_spreadsheet(data, format, output)
    else:
        raise DocumentReadError("UNSUPPORTED_FORMAT", "不支持该文档格式；DOC 和 PPT 请转换为 DOCX 和 PPTX。")
    return output.result(format)
