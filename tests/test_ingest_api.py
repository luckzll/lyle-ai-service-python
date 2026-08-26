import pytest
from fastapi import HTTPException

from app.api import ingest as ingest_api
from app.schemas.ingest import IngestRequest


def request() -> IngestRequest:
    return IngestRequest(
        fileId="file-1",
        knowledgeBaseId="kb-1",
        storagePath="file:///storage/knowledge-files/kb-1/report.pdf",
    )


def test_ingest_reports_missing_shared_storage(monkeypatch):
    def missing_storage(_request):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(ingest_api, "enqueue_ingest_task", missing_storage)

    with pytest.raises(HTTPException) as error:
        ingest_api.ingest(request())

    assert error.value.status_code == 422
    assert "未同步到解析服务" in error.value.detail


def test_ingest_preserves_document_validation_message(monkeypatch):
    def invalid_document(_request):
        raise ValueError("PPTX 文件已损坏")

    monkeypatch.setattr(ingest_api, "enqueue_ingest_task", invalid_document)

    with pytest.raises(HTTPException) as error:
        ingest_api.ingest(request())

    assert error.value.status_code == 422
    assert error.value.detail == "PPTX 文件已损坏"
