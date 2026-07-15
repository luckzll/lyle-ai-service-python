import pytest

from app.core.config import settings
from app.embedding.client import get_embedding_readiness, warm_up_embedding_model


def test_hash_embedding_warmup_marks_service_ready(monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "hash")
    monkeypatch.setattr(settings, "embedding_model", "test-hash")

    status = warm_up_embedding_model()

    assert status["ready"] is True
    assert status["provider"] == "hash"
    assert status["message"] == "向量模型已加载。"


def test_invalid_remote_embedding_config_stays_unready(monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "openai_compatible")
    monkeypatch.setattr(settings, "embedding_api_key", "")

    with pytest.raises(ValueError, match="EMBEDDING_API_KEY"):
        warm_up_embedding_model()

    assert get_embedding_readiness()["ready"] is False
