from zipfile import ZipFile

import pytest
from openpyxl import Workbook

from app.core.config import settings
from app.memory.store import export_memory
from app.rag.store import add_document
from app.rag.document_loader import StorageObjectRef, load_document_text, resolve_storage_path, split_text, split_text_with_metadata


def test_all_source_formats_and_exact_names_use_original_name_on_suffixless_storage(tmp_path):
    from app.rag.document_loader import TEXT_FILENAMES
    names = ["project." + suffix for suffix in (
        "ts tsx js jsx mjs cjs vue svelte py java kt kts go rs c h cpp hpp cs php rb swift sh bash zsh ps1 "
        "sql yaml yml toml ini cfg xml html htm css scss sass less graphql gql proto properties gradle scala r dart ipynb"
    ).split()] + list(TEXT_FILENAMES)
    path = tmp_path / "opaque-storage-uuid"
    path.write_bytes("# UTF8 中文\nvalue = 123\n".encode())
    for name in names:
        assert load_document_text(StorageObjectRef(str(path), original_name=name)) == (name, "# UTF8 中文\nvalue = 123\n")


@pytest.mark.parametrize("data,code", [(b"\xff\xfe", "INVALID_UTF8"), (b"hello\0world", "BINARY_CONTENT"),
                                      (b"hello\x0cworld", "BINARY_CONTENT"), (b"hello\x7f", "BINARY_CONTENT")])
def test_source_files_reject_invalid_utf8_and_binary_control_bytes(tmp_path, data, code):
    from app.rag.document_loader import DocumentReadError
    path = tmp_path / "opaque"
    path.write_bytes(data)
    with pytest.raises(DocumentReadError) as error:
        load_document_text(StorageObjectRef(str(path), original_name="main.py"))
    assert error.value.code == code


def test_unsupported_files_raise_and_empty_source_is_valid(tmp_path):
    from app.rag.document_loader import DocumentReadError
    path = tmp_path / "opaque"
    path.write_bytes(b"")
    with pytest.raises(DocumentReadError) as error:
        load_document_text(StorageObjectRef(str(path), original_name="archive.zip"))
    assert error.value.code == "UNSUPPORTED_FORMAT"
    assert load_document_text(StorageObjectRef(str(path), original_name="Dockerfile")) == ("Dockerfile", "")
    path.write_bytes(b"1234")
    with pytest.raises(DocumentReadError) as error:
        load_document_text(StorageObjectRef(str(path), original_name="main.py"), max_bytes=3)
    assert error.value.code == "FILE_TOO_LARGE"


@pytest.mark.parametrize("provider", [None, "LOCAL"])
def test_file_uri_reads_native_path_with_unicode_spaces_and_literal_percent(tmp_path, provider):
    # as_uri yields /C:/... on Windows and /tmp/... on POSIX. Percent-encoded
    # literal sequences must be decoded exactly once, never into a new slash.
    path = tmp_path / "附件 空格 中文 100% %20 %2F.txt"
    path.write_text("附件验收内容 V6.2", encoding="utf-8")
    ref = StorageObjectRef(storage_path=path.as_uri(), storage_provider=provider)

    assert resolve_storage_path(ref) == path
    assert load_document_text(ref) == (path.name, "附件验收内容 V6.2")


@pytest.mark.parametrize("provider", [None, "LOCAL"])
def test_plain_native_path_is_not_uri_decoded(tmp_path, provider):
    path = tmp_path / "保留原文件名 %20 %2F.txt"
    path.write_text("直接路径", encoding="utf-8")
    ref = StorageObjectRef(storage_path=str(path), storage_provider=provider)

    assert resolve_storage_path(ref) == path
    assert load_document_text(ref) == (path.name, "直接路径")


def test_load_document_text_reads_xlsx(tmp_path):
    path = tmp_path / "demo.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "用户"
    sheet.append(["姓名", "爱好"])
    sheet.append(["周乐乐", "西瓜"])
    workbook.save(path)

    title, text = load_document_text(str(path))

    assert title == "demo.xlsx"
    assert "工作表：用户" in text
    assert "周乐乐 | 西瓜" in text


def test_load_document_text_reads_pptx_slides_in_numeric_order(tmp_path):
    path = tmp_path / "quarterly-review.pptx"
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "ppt/slides/slide10.xml",
            '<p:sld xmlns:p="urn:p" xmlns:a="urn:a"><a:t>第十页结论</a:t></p:sld>',
        )
        archive.writestr(
            "ppt/slides/slide2.xml",
            '<p:sld xmlns:p="urn:p" xmlns:a="urn:a"><a:t>第二页指标</a:t><a:t>增长 28%</a:t></p:sld>',
        )

    title, text = load_document_text(str(path))

    assert title == "quarterly-review.pptx"
    assert "第二页指标" in text
    assert "增长 28%" in text
    assert text.index("第二页指标") < text.index("第十页结论")


def test_load_document_text_reports_broken_pptx(tmp_path):
    path = tmp_path / "broken.pptx"
    path.write_bytes(b"not-a-presentation")

    with pytest.raises(ValueError, match="PPTX 文件已损坏"):
        load_document_text(str(path))


def test_split_text_keeps_short_paragraphs_together_and_splits_long_blocks():
    text = "\n\n".join([
        "一、MVP 范围：\n聊天、知识库、记忆确认都属于首期范围。",
        "二、暂不包含：\n复杂多租户权限和计费系统。",
        "长段：" + "持续优化RAG切片。" * 120,
    ])

    chunks = split_text(text, chunk_size=80, overlap=10)

    assert any("一、MVP 范围" in chunk and "记忆确认" in chunk for chunk in chunks)
    assert any("二、暂不包含" in chunk and "计费系统" in chunk for chunk in chunks)
    assert len(chunks) > 3
    assert all(len(chunk) <= 80 for chunk in chunks)


def test_split_text_with_metadata_tracks_parent_and_offsets():
    text = "一、MVP 范围：\n聊天、知识库、记忆确认都属于首期范围。\n\n二、暂不包含：\n计费系统。"

    chunks = split_text_with_metadata(text, chunk_size=80, overlap=10)

    assert chunks[0].parent_title == "一、MVP 范围："
    assert chunks[0].start_char == 0
    assert chunks[0].end_char > chunks[0].start_char


def test_load_document_text_reads_xls(tmp_path):
    import xlwt

    path = tmp_path / "legacy.xls"
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet("旧表")
    sheet.write(0, 0, "姓名")
    sheet.write(0, 1, "职业")
    sheet.write(1, 0, "周乐乐")
    sheet.write(1, 1, "开发者")
    workbook.save(str(path))

    title, text = load_document_text(str(path))

    assert title == "legacy.xls"
    assert "工作表：旧表" in text
    assert "周乐乐 | 开发者" in text


def test_ingested_document_extracts_pending_memory_candidates(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "rule")

    path = tmp_path / "profile.txt"
    path.write_text("我喜欢西瓜。", encoding="utf-8")

    result = add_document("file-1", "kb-1", str(path), user_id="doc-memory-user")
    memory = export_memory("doc-memory-user", "knowledge:kb-1")

    assert result["candidate_memories"]
    assert result["candidate_memories"][0]["targetKey"] == "food_preference"
    assert result["candidate_memories"][0]["status"] == "pending"
    assert memory["structured"] == []
