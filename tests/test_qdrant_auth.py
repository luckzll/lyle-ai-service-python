from app.core.config import settings
from app.core.qdrant import qdrant_headers


def test_qdrant_headers_are_empty_without_api_key(monkeypatch):
    monkeypatch.setattr(settings, "qdrant_api_key", "")
    assert qdrant_headers() == {}


def test_qdrant_headers_include_configured_api_key(monkeypatch):
    monkeypatch.setattr(settings, "qdrant_api_key", "test-key")
    assert qdrant_headers() == {"api-key": "test-key"}
