from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

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
from app.memory.governance import (
    MemoryGovernanceConflict,
    MemoryGovernanceNotFound,
    MemoryGovernanceValidation,
    get_memory_record,
    get_purge_job,
    govern_memory_record,
    list_governance_events,
    list_memory_inbox,
    list_memory_records,
    list_memory_sources,
    request_memory_purge,
    resolve_memory_conflict,
    retry_purge_job,
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


class MemoryStatusRequest(BaseModel):
    status: str


class GovernanceRequest(BaseModel):
    clientRequestId: str = Field(min_length=8, max_length=64)
    reason: str | None = Field(default=None, max_length=240)


class CorrectMemoryRequest(GovernanceRequest):
    value: str = Field(min_length=1, max_length=4000)


class ResolveConflictRequest(GovernanceRequest):
    resolution: str
    mergedValue: str | None = Field(default=None, max_length=4000)
    expectedVersion: int = Field(ge=1)


@router.get("/memory/{user_id}")
def get_memory(user_id: str, conversationId: str | None = None) -> dict[str, object]:
    return export_memory(user_id, conversationId)


@router.get("/memory/{user_id}/records")
def memory_records(
    user_id: str,
    status: str | None = None,
    query: str | None = None,
    limit: int = 100,
) -> dict[str, object]:
    return _governance_call(list_memory_records, user_id, status=status, query=query, limit=limit)


@router.get("/memory/{user_id}/records/{memory_id}")
def memory_record(user_id: str, memory_id: str) -> dict[str, object]:
    return _governance_call(get_memory_record, user_id, memory_id)


@router.get("/memory/{user_id}/records/{memory_id}/sources")
def memory_sources(user_id: str, memory_id: str, limit: int = 100) -> dict[str, object]:
    return _governance_call(list_memory_sources, user_id, memory_id, limit=limit)


@router.get("/memory/{user_id}/inbox")
def memory_inbox(user_id: str) -> dict[str, object]:
    return _governance_call(list_memory_inbox, user_id)


@router.post("/memory/{user_id}/conflicts/{conflict_id}:resolve")
def resolve_conflict(
    user_id: str,
    conflict_id: int,
    request: ResolveConflictRequest,
) -> dict[str, object]:
    return _governance_call(
        resolve_memory_conflict,
        user_id,
        conflict_id,
        resolution=request.resolution,
        merged_value=request.mergedValue,
        expected_version=request.expectedVersion,
        client_request_id=request.clientRequestId,
    )


@router.post("/memory/{user_id}/records/{memory_id}:correct")
def correct_memory(user_id: str, memory_id: str, request: CorrectMemoryRequest) -> dict[str, object]:
    return _governance_call(
        govern_memory_record,
        user_id,
        memory_id,
        action="CORRECT",
        value=request.value,
        reason=request.reason,
        client_request_id=request.clientRequestId,
    )


def _record_action(user_id: str, memory_id: str, request: GovernanceRequest, action: str) -> dict[str, object]:
    return _governance_call(
        govern_memory_record,
        user_id,
        memory_id,
        action=action,
        reason=request.reason,
        client_request_id=request.clientRequestId,
    )


@router.post("/memory/{user_id}/records/{memory_id}:disable")
def disable_memory(user_id: str, memory_id: str, request: GovernanceRequest) -> dict[str, object]:
    return _record_action(user_id, memory_id, request, "DISABLE")


@router.post("/memory/{user_id}/records/{memory_id}:restore")
def restore_memory(user_id: str, memory_id: str, request: GovernanceRequest) -> dict[str, object]:
    return _record_action(user_id, memory_id, request, "RESTORE")


@router.post("/memory/{user_id}/records/{memory_id}:forget")
def forget_memory(user_id: str, memory_id: str, request: GovernanceRequest) -> dict[str, object]:
    return _record_action(user_id, memory_id, request, "FORGET")


@router.post("/memory/{user_id}/records/{memory_id}:purge", status_code=202)
def purge_memory(user_id: str, memory_id: str, request: GovernanceRequest) -> dict[str, object]:
    return _governance_call(
        request_memory_purge,
        user_id,
        memory_id,
        client_request_id=request.clientRequestId,
    )


@router.get("/memory/{user_id}/purge-jobs/{job_id}")
def purge_job(user_id: str, job_id: int) -> dict[str, object]:
    return _governance_call(get_purge_job, user_id, job_id)


@router.post("/memory/{user_id}/purge-jobs/{job_id}:retry")
def retry_purge(user_id: str, job_id: int) -> dict[str, object]:
    return _governance_call(retry_purge_job, user_id, job_id)


@router.get("/memory/{user_id}/governance-events")
def governance_events(user_id: str, limit: int = 100) -> dict[str, object]:
    return _governance_call(list_governance_events, user_id, limit=limit)


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


def _governance_call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except MemoryGovernanceNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MemoryGovernanceConflict as exc:
        raise HTTPException(status_code=409, detail={"message": str(exc), "current": exc.current}) from exc
    except MemoryGovernanceValidation as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
