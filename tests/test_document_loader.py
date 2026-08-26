from zipfile import ZipFile

import pytest
from openpyxl import Workbook

from app.core.config import settings
from app.memory.store import export_memory
from app.rag.store import add_document
from app.rag.document_loader import load_document_text, split_text, split_text_with_metadata


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
