from pydantic import BaseModel
from fastapi import APIRouter

from app.memory.store import (
    delete_structured_memory,
    export_memory,
    get_structured_memory_history,
    get_structured_memory_merge_suggestions,
    purge_structured_memory,
    set_profile_memory_status,
    set_structured_memory_status,
    upsert_structured_memory,
)
from app.memory.evidence_store import list_profile_evidence
from app.memory.fragments import list_memory_fragments, set_memory_fragment_status
from app.memory.profile_store import list_user_profiles

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


class MemoryStatusRequest(BaseModel):
    status: str


@router.get("/memory/{user_id}")
def get_memory(user_id: str, conversationId: str | None = None) -> dict[str, object]:
    return export_memory(user_id, conversationId)


@router.get("/memory/{user_id}/profiles")
def memory_profiles(user_id: str, status: str | None = None, limit: int = 200) -> dict[str, object]:
    return {"items": list_user_profiles(user_id, status=status, limit=limit)}


@router.get("/memory/{user_id}/fragments")
def memory_fragments(
    user_id: str,
    status: str | None = None,
    category: str | None = None,
    limit: int = 200,
) -> dict[str, object]:
    return {"items": list_memory_fragments(user_id, status=status, category=category, limit=limit)}


@router.get("/memory/{user_id}/profiles/{profile_id}/evidence")
def memory_profile_evidence(user_id: str, profile_id: int, limit: int = 100) -> dict[str, object]:
    return {"items": list_profile_evidence(user_id, profile_id, limit=limit)}


@router.patch("/memory/{user_id}/profiles/{schema_path:path}/status")
def change_profile_status(
    user_id: str,
    schema_path: str,
    request: MemoryStatusRequest,
) -> dict[str, object]:
    return set_profile_memory_status(user_id, schema_path, request.status)


@router.patch("/memory/{user_id}/fragments/{fragment_id}/status")
def change_fragment_status(
    user_id: str,
    fragment_id: int,
    request: MemoryStatusRequest,
) -> dict[str, object]:
    return set_memory_fragment_status(user_id, fragment_id, request.status)


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
