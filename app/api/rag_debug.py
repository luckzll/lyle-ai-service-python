from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.rag.retriever import retrieve_context_with_diagnostics
from app.rag.store import list_chunks
from app.rag.request_scope import request_retrieval_scope

router = APIRouter()


class RagDebugRequest(BaseModel):
    question: str
    userId: str = ""
    knowledgeScope: dict[str, object] = Field(default_factory=dict)
    knowledgeBaseIds: list[str] = Field(default_factory=list)


class RagChunksRequest(BaseModel):
    userId: str = Field(min_length=1)
    knowledgeBaseIds: list[str] = Field(min_length=1)
    knowledgeScope: dict[str, object]
    limit: int = Field(default=50, ge=1, le=200)


@router.post("/rag/debug")
def rag_debug(request: RagDebugRequest) -> dict[str, object]:
    with request_retrieval_scope(request.userId, request.knowledgeBaseIds, request.knowledgeScope):
        diagnostics = retrieve_context_with_diagnostics(request.question, request.knowledgeBaseIds)
    diagnostics["knowledgeBaseIds"] = request.knowledgeBaseIds
    return diagnostics


@router.get("/rag/chunks")
def rag_chunks(knowledgeBaseId: str | None = None, limit: int = 50) -> dict[str, object]:
    # Only Java's explicit ADMIN global diagnostics uses this internal-token route.
    if knowledgeBaseId is not None:
        raise HTTPException(status_code=422, detail="Knowledge-base chunks require a scoped POST request")
    return {"items": list_chunks(None, limit)}


@router.post("/rag/chunks")
def rag_chunks_scoped(request: RagChunksRequest) -> dict[str, object]:
    with request_retrieval_scope(request.userId, request.knowledgeBaseIds, request.knowledgeScope):
        return {"items": list_chunks(None, request.limit)}
