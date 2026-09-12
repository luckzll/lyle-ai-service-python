from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.api.rag_debug import router
from app.api import agent, chat
from app.core.config import settings
from app.core.security import internal_token_middleware
from app.rag import retriever, store
from app.rag.cache_scope import candidate_cache, current_retrieval_cache_scope, invalidate_candidate_cache


@pytest.fixture
def api(monkeypatch):
    invalidate_candidate_cache()
    for name, value in {
        "qdrant_enabled": False, "rerank_enabled": False,
        "rag_query_rewrite_enabled": False, "rag_sufficiency_check_enabled": False,
        "rag_adaptive_retrieval_enabled": False, "rag_optimized_retrieval_enabled": True,
        "rag_candidate_cache_enabled": True, "internal_api_token": "integration-internal-token-only",
    }.items():
        monkeypatch.setattr(settings, name, value)
    monkeypatch.setattr(store, "_chunks", [
        store.KnowledgeChunk("kept", "kb", "Source one", "beta current passage", retrieval_revision="revision-one"),
        store.KnowledgeChunk("deleted", "kb", "Source two", "beta deleted passage", retrieval_revision="revision-two"),
    ])
    application = FastAPI()
    application.middleware("http")(internal_token_middleware)
    application.include_router(router, prefix="/ai")
    application.include_router(agent.router, prefix="/ai")
    application.include_router(chat.router, prefix="/ai")
    with TestClient(application) as client:
        yield client
    invalidate_candidate_cache()


def payload(documents=None, version="manifest-one", revisions=None):
    docs = ["kept"] if documents is None else documents
    return {
        "question": "beta", "userId": "user", "knowledgeBaseIds": ["kb"],
        "knowledgeScope": {
            "userId": "user", "authorizationRevision": "owned-by-user", "knowledgeBaseRevisions": {"kb": version},
            "allowedDocumentIds": {"kb": docs},
            "documentRevisions": {"kb": revisions if revisions is not None else {doc: "revision-one" for doc in docs}},
        },
    }


def post(api, body):
    return api.post("/ai/rag/debug", json=body, headers={"X-Internal-Token": "integration-internal-token-only"})


def test_internal_api_manifest_reaches_store_and_candidate_cache(api, monkeypatch):
    calls = []
    original = retriever.search_candidates
    monkeypatch.setattr(retriever, "search_candidates", lambda *args: calls.append(1) or original(*args))
    first = post(api, payload())
    second = post(api, payload())
    assert first.status_code == second.status_code == 200
    assert {row["doc_id"] for row in first.json()["contexts"]} == {"kept"}
    assert second.json()["retrievalPlan"]["candidateCache"]["hit"] == 1
    assert len(calls) == 1
    assert first.json()["contexts"] == second.json()["contexts"]


def test_deleted_document_manifest_never_reuses_warm_results_or_stale_local_rows(api):
    assert post(api, payload()).json()["contexts"]
    removed = post(api, payload(documents=[], version="after-delete", revisions={}))
    assert removed.status_code == 200
    assert removed.json()["contexts"] == []
    assert removed.json()["retrievalPlan"]["candidateCache"]["hit"] == 0
    # The stale local row remains to simulate a different Python worker.
    assert any(chunk.doc_id == "kept" for chunk in store._chunks)


def test_rebuild_manifest_rejects_another_workers_stale_same_id_revision(api):
    assert post(api, payload()).json()["contexts"]
    rebuilt = post(api, payload(version="after-rebuild", revisions={"kept": "revision-new"}))
    assert rebuilt.status_code == 200
    assert rebuilt.json()["contexts"] == []
    # Only publishing the verified revision may become searchable under the new manifest.
    store._chunks[:] = [store.KnowledgeChunk("kept", "kb", "Source revised", "beta fresh revision", retrieval_revision="revision-new")]
    fresh = post(api, payload(version="after-rebuild", revisions={"kept": "revision-new"}))
    assert [row["chunk"] for row in fresh.json()["contexts"]] == ["beta fresh revision"]


def test_api_rejects_mismatched_or_incomplete_manifest_before_retrieval(api, monkeypatch):
    monkeypatch.setattr(retriever, "search_candidates", lambda *_: pytest.fail("invalid manifest reached retrieval"))
    mismatch = payload()
    mismatch["knowledgeScope"]["userId"] = "different-user"
    assert post(api, mismatch).status_code == 403
    incomplete = payload()
    del incomplete["knowledgeScope"]["documentRevisions"]
    assert post(api, incomplete).status_code == 422
    assert candidate_cache.stats()["entries"] == 0


def test_a_manifest_without_the_internal_service_token_cannot_establish_scope(api):
    assert api.post("/ai/rag/debug", json=payload()).status_code == 401
    assert candidate_cache.stats()["entries"] == 0


def test_selected_knowledge_bases_without_manifest_are_rejected(api, monkeypatch):
    monkeypatch.setattr(retriever, "search_candidates", lambda *_: pytest.fail("missing scope reached retrieval"))
    body = payload()
    del body["knowledgeScope"]
    assert post(api, body).status_code == 422


def test_agent_chat_also_enters_verified_scope_and_rejects_missing_manifest(api, monkeypatch):
    calls = []

    def graph(user, conversation, question, ids):
        calls.append(current_retrieval_cache_scope())
        return {"answer": "available", "contexts": [], "memory": {}}

    monkeypatch.setattr(agent, "run_multi_agent_graph", graph)
    monkeypatch.setattr(agent, "write_memory_in_background", lambda *_args, **_kwargs: None)
    body = payload()
    body["message"] = body.pop("question")
    headers = {"X-Internal-Token": "integration-internal-token-only"}
    valid = api.post("/ai/agent/chat", json=body, headers=headers)
    assert valid.status_code == 200
    assert calls[0].user_id == "user" and dict(calls[0].document_revisions)["kb"] == (("kept", "revision-one"),)
    del body["knowledgeScope"]
    invalid = api.post("/ai/agent/chat", json=body, headers=headers)
    assert invalid.status_code == 422 and len(calls) == 1


@pytest.mark.parametrize("manifest", [None, {}])
def test_stream_rejects_missing_manifest_before_sse_headers_or_worker(api, monkeypatch, manifest):
    def unexpected(*_args, **_kwargs):
        pytest.fail("invalid retrieval scope started a stream worker")

    monkeypatch.setattr(chat, "HeartbeatStreamingResponse", unexpected)
    monkeypatch.setattr(chat, "_text_chat_frames", unexpected)
    body = payload()
    body["message"] = body.pop("question")
    if manifest is None:
        del body["knowledgeScope"]
    else:
        body["knowledgeScope"] = manifest
    response = api.post("/ai/chat/stream", json=body,
                        headers={"X-Internal-Token": "integration-internal-token-only"})
    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
