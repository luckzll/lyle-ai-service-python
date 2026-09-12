import pytest
from fastapi import HTTPException

from app.rag.cache_scope import allowed_document_ids, current_retrieval_cache_scope
from app.rag.request_scope import request_retrieval_scope


def manifest(**updates):
    return {"userId": "7", "authorizationRevision": "auth1", "knowledgeBaseRevisions": {"10": "v1"},
            "allowedDocumentIds": {"10": ["1"]}, "documentRevisions": {"10": {"1": "v1"}}, **updates}


def test_scope_accepts_server_manifest_and_restores_context():
    with request_retrieval_scope("7", ["10"], manifest(), "conversation1"):
        assert allowed_document_ids("10") == frozenset({"1"})
        assert allowed_document_ids("20") == frozenset()
    assert current_retrieval_cache_scope() is None


@pytest.mark.parametrize("data", [manifest(userId="8"), manifest(allowedDocumentIds={}),
                                  manifest(authorizationRevision=""), manifest(knowledgeBaseRevisions={"20": "v1"})])
def test_explicit_invalid_scope_rejected_instead_of_unrestricted_fallback(data):
    with pytest.raises(HTTPException), request_retrieval_scope("7", ["10"], data):
        pytest.fail("must reject malformed or mismatched trusted metadata")


def test_call_without_selected_knowledge_bases_bypasses_cache():
    with request_retrieval_scope("7", []):
        assert current_retrieval_cache_scope() is None


@pytest.mark.parametrize("data", [None, {}, []])
def test_selected_knowledge_bases_require_verified_manifest(data):
    with pytest.raises(HTTPException) as error:
        with request_retrieval_scope("7", ["10"], data):
            pytest.fail("selected KBs cannot fall back to unrestricted legacy retrieval")
    assert error.value.status_code == 422


def test_empty_authorized_file_list_is_not_unrestricted():
    with request_retrieval_scope("7", ["10"], manifest(allowedDocumentIds={"10": []})):
        assert allowed_document_ids("10") == frozenset()
