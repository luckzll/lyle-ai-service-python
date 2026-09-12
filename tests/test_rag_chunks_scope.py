from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.api.rag_debug import router
from app.core.config import settings
from app.core.security import internal_token_middleware
from app.rag import store
from app.rag.cache_scope import current_retrieval_cache_scope


TOKEN = "chunks-test-internal-token"
HEADERS = {"X-Internal-Token": TOKEN}


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(settings, "internal_api_token", TOKEN)
    monkeypatch.setattr(store, "_chunks", [
        store.KnowledgeChunk("1", "10", "old", "stale revision", retrieval_revision="old"),
        store.KnowledgeChunk("1", "10", "new", "current revision", retrieval_revision="new"),
        store.KnowledgeChunk("2", "10", "deleted", "deleted source", retrieval_revision="new"),
        store.KnowledgeChunk("1", "20", "other", "another knowledge base", retrieval_revision="new"),
    ])
    application = FastAPI()
    application.middleware("http")(internal_token_middleware)
    application.include_router(router, prefix="/ai")
    with TestClient(application) as client:
        yield client


def request_payload():
    return {
        "userId": "7", "knowledgeBaseIds": ["10"], "limit": 50,
        "knowledgeScope": {
            "userId": "7", "authorizationRevision": "owned-by-user-7",
            "knowledgeBaseRevisions": {"10": "current-manifest"},
            "allowedDocumentIds": {"10": ["1"]}, "documentRevisions": {"10": {"1": "new"}},
        },
    }


def test_scoped_chunk_listing_filters_old_revision_deleted_document_and_other_kb(api):
    response = api.post("/ai/rag/chunks", json=request_payload(), headers=HEADERS)
    assert response.status_code == 200
    assert [row["chunk"] for row in response.json()["items"]] == ["current revision"]
    assert current_retrieval_cache_scope() is None


def test_scope_filtering_happens_before_limit(api):
    body = request_payload()
    body["limit"] = 1
    assert [row["chunk"] for row in api.post("/ai/rag/chunks", json=body, headers=HEADERS).json()["items"]] == ["current revision"]


def test_scoped_listing_rechecks_each_new_manifest_without_reusing_old_rows(api):
    body = request_payload()
    assert api.post("/ai/rag/chunks", json=body, headers=HEADERS).json()["items"]
    body["knowledgeScope"]["documentRevisions"] = {"10": {"1": "future-revision"}}
    assert api.post("/ai/rag/chunks", json=body, headers=HEADERS).json()["items"] == []
    body["knowledgeScope"]["allowedDocumentIds"] = {"10": []}
    body["knowledgeScope"]["documentRevisions"] = {"10": {}}
    assert api.post("/ai/rag/chunks", json=body, headers=HEADERS).json()["items"] == []


@pytest.mark.parametrize("mutation, expected_status", [
    (lambda body: body.update(userId="someone-else"), 403),
    (lambda body: body.update(knowledgeScope={}), 422),
    (lambda body: body["knowledgeScope"].pop("documentRevisions"), 422),
    (lambda body: body.update(knowledgeBaseIds=[]), 422),
    (lambda body: body.update(knowledgeBaseIds=["20"]), 422),
    (lambda body: body.update(limit=201), 422),
])
def test_invalid_chunks_scope_is_rejected_before_store_read(api, monkeypatch, mutation, expected_status):
    monkeypatch.setattr("app.api.rag_debug.list_chunks", lambda *_: pytest.fail("invalid scope reached store"))
    body = request_payload()
    mutation(body)
    assert api.post("/ai/rag/chunks", json=body, headers=HEADERS).status_code == expected_status


def test_both_routes_require_internal_token_and_legacy_get_rejects_kb_requests(api):
    assert api.post("/ai/rag/chunks", json=request_payload()).status_code == 401
    assert api.get("/ai/rag/chunks").status_code == 401
    assert api.get("/ai/rag/chunks?knowledgeBaseId=10", headers=HEADERS).status_code == 422
    # This route is reachable only through Java's ADMIN-only global diagnostics.
    assert len(api.get("/ai/rag/chunks", headers=HEADERS).json()["items"]) == 4
