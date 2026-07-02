from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app


def test_python_rate_limit(monkeypatch):
    monkeypatch.setattr(settings, "internal_api_token", "test-internal-token")
    monkeypatch.setattr(settings, "python_rate_limit_per_minute", 1)
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    client = TestClient(app)

    headers = {"X-Internal-Token": "test-internal-token"}
    assert client.get("/ai/memory/rate-user", headers=headers).status_code == 200
    assert client.get("/ai/memory/rate-user", headers=headers).status_code == 429
