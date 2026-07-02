from pydantic import BaseModel
from fastapi import APIRouter

from app.memory.store import (
    delete_structured_memory,
    export_memory,
    get_structured_memory_history,
    get_structured_memory_merge_suggestions,
    purge_structured_memory,
    set_structured_memory_status,
    upsert_structured_memory,
)

router = APIRouter()


class StructuredMemoryRequest(BaseModel):
    key: str
    value: str
    confidence: float = 1.0
    status: str = "active"
    source: str = "manual"
    expiresAt: float | None = None


class StructuredMemoryStatusRequest(BaseModel):
    status: str


@router.get("/memory/{user_id}")
def get_memory(user_id: str, conversationId: str | None = None) -> dict[str, object]:
    return export_memory(user_id, conversationId)


@router.get("/memory/{user_id}/structured/{key}/history")
def structured_memory_history(user_id: str, key: str, limit: int = 30) -> dict[str, object]:
    return {"items": get_structured_memory_history(user_id, key, limit)}


@router.get("/memory/{user_id}/structured/merge-suggestions")
def structured_memory_merge_suggestions(user_id: str) -> dict[str, object]:
    return {"items": get_structured_memory_merge_suggestions(user_id)}


@router.put("/memory/{user_id}/structured")
def save_structured_memory(user_id: str, request: StructuredMemoryRequest) -> dict[str, object]:
    return upsert_structured_memory(
        user_id,
        request.key,
        request.value,
        source=request.source,
        confidence=request.confidence,
        status=request.status,
        expires_at=request.expiresAt,
    )


@router.put("/memory/{user_id}/structured/{key}/status")
def change_structured_memory_status(
    user_id: str,
    key: str,
    request: StructuredMemoryStatusRequest,
) -> dict[str, object]:
    return set_structured_memory_status(user_id, key, request.status)


@router.delete("/memory/{user_id}/structured/{key}")
def remove_structured_memory(user_id: str, key: str) -> dict[str, str]:
    delete_structured_memory(user_id, key)
    return {"status": "deleted"}


@router.delete("/memory/{user_id}/structured/{key}/purge")
def purge_structured_memory_api(user_id: str, key: str) -> dict[str, str]:
    purge_structured_memory(user_id, key)
    return {"status": "purged"}
