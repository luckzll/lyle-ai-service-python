import base64
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

from docx import Document
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openpyxl import Workbook
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
import pytest
import xlwt

from app.api import local_workspace_document as module
from app.api.local_workspace import router as workspace_router, TOOLS, PROMPT
from app.rag import document_loader as loader


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(workspace_router, prefix="/ai")
    return TestClient(app)


def payload(data, name="sample.docx"):
    return {"name": name, "data": base64.b64encode(data).decode("ascii")}


def post(client, data, name="sample.docx"):
    return client.post("/ai/local-workspace/document/extract", json=payload(data, name))


def docx(text="Project scope: local documents"):
    document = Document()
    document.add_paragraph(text)
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Task"
    table.cell(0, 1).text = "Review"
    output = BytesIO(); document.save(output)
    return output.getvalue()


def xlsx(rows=None):
    workbook = Workbook(); sheet = workbook.active; sheet.title = "项目"
    for row in rows or [["名称", "进度"], ["文档支持", "完成"]]:
        sheet.append(row)
    output = BytesIO(); workbook.save(output); workbook.close()
    return output.getvalue()


def xls():
    workbook = xlwt.Workbook(); sheet = workbook.add_sheet("旧表")
    sheet.write(0, 0, "旧版工作簿"); sheet.write(0, 1, "可以读取")
    output = BytesIO(); workbook.save(output)
    return output.getvalue()


def pptx(texts=("Presentation first page", "Presentation second page")):
    # A real OOXML package with content types, package relationships, presentation
    # and slide relationships; no optional python-pptx dependency is required.
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>'
            + ''.join(f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>' for i in range(1, len(texts) + 1)) + '</Types>')
        archive.writestr("_rels/.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/></Relationships>')
        archive.writestr("ppt/presentation.xml", '<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><p:sldIdLst>'
            + ''.join(f'<p:sldId id="{255 + i}" r:id="rId{i}"/>' for i in range(1, len(texts) + 1)) + '</p:sldIdLst><p:sldSz cx="9144000" cy="6858000"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>')
        archive.writestr("ppt/_rels/presentation.xml.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + ''.join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>' for i in range(1, len(texts) + 1)) + '</Relationships>')
        for index, text in enumerate(texts, 1):
            archive.writestr(f"ppt/slides/slide{index}.xml", '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:cSld><p:spTree>'
                '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
                '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
                '<p:sp><p:nvSpPr><p:cNvPr id="2" name="Text 1"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr/>'
                '<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:t>' + text + '</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>')
    return output.getvalue()


def pdf(text="PDF document text", pages=1, encrypted=False):
    writer = PdfWriter()
    for _ in range(pages):
        page = writer.add_blank_page(width=300, height=200)
        if text:
            font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
            page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
            content = DecodedStreamObject()
            content.set_data(f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode("ascii"))
            page[NameObject("/Contents")] = writer._add_object(content)
    if encrypted:
        writer.encrypt("fixture-password")
    output = BytesIO(); writer.write(output)
    return output.getvalue()


@pytest.mark.parametrize("format,builder,expected", [
    ("docx", docx, "Project scope"), ("xlsx", xlsx, "文档支持 | 完成"),
    ("xlsm", xlsx, "文档支持 | 完成"), ("xls", xls, "旧版工作簿 | 可以读取"),
    ("pptx", pptx, "Presentation second page"), ("pdf", pdf, "PDF document text"),
])
def test_actual_document_bytes_are_extracted_via_http(client, format, builder, expected):
    response = post(client, builder(), f"项目文档.{format.upper()}")
    assert response.status_code == 200
    result = response.json()
    assert expected in result["text"]
    assert result["format"] == format
    assert result["truncated"] is False
    if format in {"xls", "xlsx", "xlsm"}:
        assert any("不执行宏" in warning for warning in result["warnings"])
    else:
        assert result["warnings"] == []
    assert set(result) == {"text", "format", "truncated", "warnings"}


def test_blank_pdf_reports_no_recognized_text_without_fabricating_readable_content(client):
    result = post(client, pdf(text=""), "scan.pdf").json()
    assert result["text"] == "" and result["truncated"] is False
    assert "扫描版 PDF" in result["warnings"][0]


def test_empty_spreadsheet_does_not_treat_sheet_title_as_document_text(client):
    workbook = Workbook(); workbook.active.title = "Empty sheet"
    output = BytesIO(); workbook.save(output); workbook.close()
    result = post(client, output.getvalue(), "empty.xlsx").json()
    assert result["text"] == "" and result["warnings"]


@pytest.mark.parametrize("name", ["legacy.doc", "legacy.ppt", "script.exe", "../sample.pdf", "folder\\sample.pdf",
                                     ".hidden.pdf", "bad:name.pdf", "bad\x7f.pdf", "x" * 125 + ".pdf"])
def test_unsupported_or_path_names_return_friendly_400(client, name):
    response = post(client, b"not parsed", name)
    assert response.status_code == 400
    assert "not parsed" not in response.text
    if name.endswith((".doc", ".ppt")):
        assert "DOCX" in response.json()["detail"] and "PPTX" in response.json()["detail"]


@pytest.mark.parametrize("body", [None, [], {}, {"name": "x.pdf", "data": "!secret!"},
    {"name": "x.pdf", "data": "data:application/pdf;base64,AAAA"},
    {"name": "x.pdf", "data": ""}, {"name": "x.pdf", "data": "非ASCII"},
    {"name": "x.pdf", "data": "Zh=="}, {"name": "x.pdf", "data": "Zg==="},
    {"name": "x.pdf", "data": "AAAA", "storage_path": "secret-path"}])
def test_invalid_payloads_return_400_without_echoing_file_content(client, body):
    response = client.post("/ai/local-workspace/document/extract", json=body)
    assert response.status_code == 400
    assert "!secret!" not in response.text and "secret-path" not in response.text


def test_invalid_json_returns_400(client):
    response = client.post("/ai/local-workspace/document/extract", content=b'{"name":')
    assert response.status_code == 400
    response = client.post("/ai/local-workspace/document/extract", content=b"[" * 2_000 + b"]" * 2_000)
    assert response.status_code == 400


@pytest.mark.parametrize("format", ["pdf", "docx", "xlsx", "xls", "pptx"])
def test_corrupt_or_mime_spoofed_files_are_not_accepted(client, format):
    response = post(client, b"SECRET_PROVIDER_DETAIL /home/service/file", f"bad.{format}")
    assert response.status_code == 400
    assert "SECRET_PROVIDER_DETAIL" not in response.text and "/home/" not in response.text


def test_encrypted_pdf_requires_decryption(client):
    response = post(client, pdf(encrypted=True), "protected.pdf")
    assert response.status_code == 400
    assert "解密" in response.json()["detail"]
    assert "fixture-password" not in response.text


def test_raw_encoded_and_streamed_request_size_limits_return_413(client):
    response = post(client, b"x" * (module.MAX_FILE_BYTES + 1), "large.pdf")
    assert response.status_code == 413
    response = client.post("/ai/local-workspace/document/extract", json={"name": "x.pdf", "data": "A" * (module.MAX_ENCODED_CHARS + 1)})
    assert response.status_code == 413
    response = client.post("/ai/local-workspace/document/extract", content=b" " * (module.MAX_BODY_BYTES + 1))
    assert response.status_code == 413


def test_zip_expansion_and_entry_count_limits_reject_bombs(client):
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", b"x" * 1_100_000)
    assert post(client, output.getvalue()).status_code == 413
    output = BytesIO()
    with ZipFile(output, "w") as archive:
        for index in range(loader.LOCAL_DOCUMENT_MAX_ZIP_ENTRIES + 1):
            archive.writestr(f"parts/{index}", b"")
    assert post(client, output.getvalue()).status_code == 413


@pytest.mark.parametrize("part,data", [
    ("../outside.xml", b"x"), ("word/document.xml", b'<!DOCTYPE x [<!ENTITY e SYSTEM "file:///secret">]><x/>'),
    ("word/document.xml", '<!DOCTYPE x [<!ENTITY e SYSTEM "file:///secret">]><x/>'.encode("utf-16")),
])
def test_unsafe_zip_members_and_external_entities_are_rejected(client, part, data):
    output = BytesIO()
    with ZipFile(output, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr(part, data)
    response = post(client, output.getvalue())
    assert response.status_code == 400 and "file:///secret" not in response.text


def test_wrong_ooxml_format_is_rejected(client):
    assert post(client, xlsx(), "spoof.docx").status_code == 400


def test_character_output_is_capped_and_marked_truncated(client, monkeypatch):
    monkeypatch.setattr(loader, "LOCAL_DOCUMENT_MAX_TEXT", 32)
    result = post(client, docx("A meaningful paragraph with more than thirty-two characters")).json()
    assert len(result["text"]) == 32
    assert result["truncated"] and result["warnings"]


def test_pdf_pages_and_spreadsheet_rows_are_bounded(client, monkeypatch):
    monkeypatch.setattr(loader, "LOCAL_DOCUMENT_MAX_PAGES", 2)
    result = post(client, pdf(pages=3), "pages.pdf").json()
    assert "第 2 页" in result["text"] and "第 3 页" not in result["text"] and result["truncated"]
    monkeypatch.setattr(loader, "LOCAL_DOCUMENT_MAX_ROWS", 2)
    result = post(client, xlsx([["row-one"], ["row-two"], ["row-three"]]), "rows.xlsx").json()
    assert "row-two" in result["text"] and "row-three" not in result["text"] and result["truncated"]


def test_uncached_spreadsheet_formulas_are_preserved_without_claiming_calculated_results(client):
    result = post(client, xlsx([["Input", "Formula"], [2, "=A2*3"]]), "formulas.xlsx").json()
    assert "=A2*3" in result["text"]
    assert "2 | =A2*3" in result["text"] and "2 | 6" not in result["text"]
    assert any("不执行宏或计算公式" in warning for warning in result["warnings"])


def test_embedded_media_gets_an_explicit_unrecognized_content_warning(client):
    original = docx()
    output = BytesIO()
    with ZipFile(BytesIO(original)) as source, ZipFile(output, "w", ZIP_DEFLATED) as target:
        for entry in source.infolist():
            target.writestr(entry.filename, source.read(entry))
        target.writestr("word/media/image1.png", b"image bytes are not interpreted")
    result = post(client, output.getvalue()).json()
    assert "Project scope" in result["text"]
    assert any("未识别内嵌图片或图表" in warning for warning in result["warnings"])


def test_presentation_slides_are_bounded(client, monkeypatch):
    monkeypatch.setattr(loader, "LOCAL_DOCUMENT_MAX_PAGES", 1)
    result = post(client, pptx(), "slides.pptx").json()
    assert "first page" in result["text"] and "second page" not in result["text"] and result["truncated"]


def test_presentation_order_follows_slide_relationships_not_filenames(client):
    output = BytesIO()
    with ZipFile(BytesIO(pptx())) as source, ZipFile(output, "w", ZIP_DEFLATED) as target:
        for entry in source.infolist():
            data = source.read(entry)
            if entry.filename == "ppt/presentation.xml":
                data = data.replace(b'<p:sldId id="256" r:id="rId1"/><p:sldId id="257" r:id="rId2"/>',
                                    b'<p:sldId id="257" r:id="rId2"/><p:sldId id="256" r:id="rId1"/>')
            target.writestr(entry.filename, data)
    result = post(client, output.getvalue(), "reordered.pptx").json()
    assert result["text"].index("second page") < result["text"].index("first page")


def test_pdf_content_stream_bytes_are_bounded_before_text_extraction(client, monkeypatch):
    monkeypatch.setattr(loader, "LOCAL_DOCUMENT_MAX_ZIP_ENTRY", 64)
    response = post(client, pdf("large content " * 20), "large-content.pdf")
    assert response.status_code == 413


def test_spreadsheet_without_dimension_metadata_still_reads_actual_cells(client):
    output = BytesIO()
    with ZipFile(BytesIO(xlsx([["dimensionless content"]]))) as source, ZipFile(output, "w", ZIP_DEFLATED) as target:
        for entry in source.infolist():
            data = source.read(entry)
            if entry.filename == "xl/worksheets/sheet1.xml":
                data = data.replace(b'<dimension ref="A1:A1"/>', b'')
            target.writestr(entry.filename, data)
    result = post(client, output.getvalue(), "no-dimensions.xlsx").json()
    assert "dimensionless content" in result["text"]
    assert result["truncated"]


def test_parser_failure_is_sanitized_and_does_not_echo_the_exception(client, monkeypatch):
    def fail(*args):
        raise RuntimeError("secret path and content: /home/private user text")
    monkeypatch.setattr(module, "load_local_document_text", fail)
    response = post(client, docx())
    assert response.status_code == 400
    assert "secret" not in response.text and "/home/" not in response.text


def test_extraction_never_uses_storage_resolution_ingestion_or_temporary_files(client, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Persistent attachment or temporary storage must not be used")
    for name in ["_resolve_storage_path", "_download_minio_object", "load_document_text", "NamedTemporaryFile"]:
        monkeypatch.setattr(loader, name, forbidden)
    assert post(client, docx()).status_code == 200
    assert post(client, xlsx(), "table.xlsx").status_code == 200
    assert post(client, pdf(), "read.pdf").status_code == 200


def test_model_tool_descriptions_explain_document_read_only_and_empty_pdf_feedback():
    read = next(tool["function"]["description"] for tool in TOOLS if tool["function"]["name"] == "workspace_read")
    assert "PDF" in read and "PPTX" in read and "不能修改" in read
    assert "扫描版 PDF 未识别文字时不能宣称读到了内容" in PROMPT
