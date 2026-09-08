from pydantic import BaseModel, Field
from typing import Literal
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


class FactRevisionRequest(BaseModel):
    expectedRevision: int = Field(ge=1)
    status: Literal['active', 'disabled', 'forgotten'] | None = None
    value: str | None = Field(default=None, min_length=1, max_length=4000)


@router.get('/memory/{user_id}/facts')
def memory_facts(user_id: str):
    from app.memory import fact_store
    if not fact_store.enabled():
        raise HTTPException(409, 'MEMORY_FACTS_DISABLED')
    return {'items': fact_store.list_facts(user_id), 'migration': fact_store.migration_report(user_id)}


@router.put('/memory/{user_id}/facts/{fact_id}')
def update_memory_fact(user_id: str, fact_id: str, request: FactRevisionRequest):
    from app.memory import fact_store
    if not fact_store.enabled():
        raise HTTPException(409, 'MEMORY_FACTS_DISABLED')
    if request.status is None and request.value is None:
        raise HTTPException(422, 'MEMORY_FACT_CHANGE_REQUIRED')
    try:
        return fact_store.revise_fact(user_id, fact_id, expected_revision=request.expectedRevision,
                                      status=request.status, value=request.value)
    except KeyError:
        raise HTTPException(404, 'MEMORY_FACT_NOT_FOUND')
    except fact_store.FactConflict as exc:
        raise HTTPException(409, str(exc))
    except ValueError as exc:
        raise HTTPException(422, str(exc))


class SchemaRecallRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    mode: Literal["auto", "single_pass", "agent"] = "auto"
    allowedPaths: list[str] | None = Field(default=None, max_length=100)


@router.get("/memory/{user_id}/schema")
def memory_schema_status(user_id: str):
    from app.memory.schema_store import read_state, build_status
    from app.core.config import settings
    from app.memory.fact_store import enabled as facts_enabled
    return {"enabled": settings.memory_schema_enabled, 'factsEnabled': facts_enabled(), **build_status(read_state(user_id))}


@router.post("/memory/{user_id}/schema/rebuild")
def rebuild_memory_schema(user_id: str):
    from app.memory.schema_store import request_build, build_status
    from app.core.config import settings
    if not settings.memory_schema_enabled:
        raise HTTPException(status_code=409, detail="MEMORY_SCHEMA_DISABLED")
    return build_status(request_build(user_id, force=True))


@router.post("/memory/{user_id}/schema/recall")
def recall_memory_schema(user_id: str, request: SchemaRecallRequest):
    from app.core.config import settings
    from app.memory.store import recall_memory
    if not settings.memory_schema_enabled:
        raise HTTPException(status_code=409, detail="MEMORY_SCHEMA_DISABLED")
    result = recall_memory(user_id, None, request.question, include_vector=False,
                           schema_mode=request.mode, allowed_paths=request.allowedPaths)
    return {"schema": result["schema_recall"], "context": result["profile_context"]["context"],
            "usedMemories": result["used_memories"]}


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
