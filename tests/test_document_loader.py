from openpyxl import Workbook

from app.core.config import settings
from app.memory.store import export_memory
from app.rag.store import add_document
from app.rag.document_loader import load_document_text


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
