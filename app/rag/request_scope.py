"""Validate the authenticated Java service's retrieval manifest at the API boundary."""
from contextlib import contextmanager
from collections.abc import Mapping

from fastapi import HTTPException

from app.rag.cache_scope import retrieval_cache_scope


@contextmanager
def request_retrieval_scope(user_id, knowledge_base_ids, manifest=None, conversation_id=None):
    manifest = {} if manifest is None else manifest
    if not isinstance(manifest, Mapping):
        raise HTTPException(status_code=422, detail="Invalid retrieval scope")
    if knowledge_base_ids and not manifest:
        raise HTTPException(status_code=422, detail="A verified retrieval scope is required for selected knowledge bases")
    if manifest and manifest.get("userId") != user_id:
        raise HTTPException(status_code=403, detail="Retrieval scope identity mismatch")
    with retrieval_cache_scope(
        user_id, knowledge_base_ids,
        revisions=manifest.get("knowledgeBaseRevisions"),
        authorization_revision=manifest.get("authorizationRevision"),
        conversation_id=conversation_id or "",
        allowed_document_ids=manifest.get("allowedDocumentIds"),
        document_revisions=manifest.get("documentRevisions"),
    ) as scope:
        if manifest and scope is None:
            raise HTTPException(status_code=422, detail="Incomplete retrieval scope")
        yield scope
