from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.rag.retriever import retrieve_context_with_diagnostics
from app.rag.store import list_chunks

router = APIRouter()


class RagDebugRequest(BaseModel):
    question: str
    knowledgeBaseIds: list[str] = Field(default_factory=list)


@router.post("/rag/debug")
def rag_debug(request: RagDebugRequest) -> dict[str, object]:
    diagnostics = retrieve_context_with_diagnostics(request.question, request.knowledgeBaseIds)
    diagnostics["knowledgeBaseIds"] = request.knowledgeBaseIds
    return diagnostics


@router.get("/rag/chunks")
def rag_chunks(knowledgeBaseId: str | None = None, limit: int = 50) -> dict[str, object]:
    return {"items": list_chunks(knowledgeBaseId, limit)}
