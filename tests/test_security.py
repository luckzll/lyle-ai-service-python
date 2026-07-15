from fastapi.testclient import TestClient

from app.core.config import settings
from app import main as main_module
from app.main import app


def test_insecure_internal_token_rejects_all_ai_routes(monkeypatch):
    for insecure in ("", "lyle-internal-dev-token-change-me", "change-this-internal-token"):
        monkeypatch.setattr(settings, "internal_api_token", insecure)

        response = TestClient(app).get("/ai/memory/user-1")

        assert response.status_code == 503


def test_internal_token_required_for_ai_routes(monkeypatch):
    monkeypatch.setattr(settings, "internal_api_token", "test-internal-token")

    response = TestClient(app).get("/ai/memory/user-1")

    assert response.status_code == 401


def test_internal_token_allows_ai_routes(monkeypatch):
    monkeypatch.setattr(settings, "internal_api_token", "test-internal-token")
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)

    response = TestClient(app).get(
        "/ai/memory/user-1",
        headers={"X-Internal-Token": "test-internal-token"},
    )

    assert response.status_code == 200
    assert response.json()["short_term"] == []


def test_readiness_probe_is_public_but_requires_secure_service_configuration(monkeypatch):
    monkeypatch.setattr(main_module, "get_embedding_readiness", lambda: {
        "ready": True,
        "provider": "hash",
        "model": "test",
        "message": "向量模型已加载。",
    })
    monkeypatch.setattr(settings, "internal_api_token", "test-internal-token")

    response = TestClient(app).get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
