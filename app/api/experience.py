from typing import Any

from fastapi import APIRouter, Header, Query

from app.experience.confirmation_inbox import confirm, get_confirmation, list_confirmations, reject
from app.experience.connection_diagnostics import connection_diagnostics, diagnose_connection, recommended_connections
from app.experience.draft_service import create_from_intent, create_from_template, discard_draft, enable_draft, get_draft
from app.experience.onboarding_store import get_onboarding, run_first_inspection, update_onboarding
from app.experience.result_summary_service import get_result, list_results, needs_attention
from app.experience.template_catalog import list_templates
from app.experience.analytics import experience_analytics


router = APIRouter()


@router.get("/experience/analytics")
def analytics(days: int = Query(default=7, ge=1, le=90)) -> dict[str, Any]:
    return experience_analytics(days)


@router.get("/experience/templates/{user_id}")
def templates(user_id: str) -> dict[str, Any]:
    return {"items": list_templates()}


@router.get("/experience/onboarding/{user_id}")
def onboarding(user_id: str) -> dict[str, Any]:
    return get_onboarding(user_id)


@router.patch("/experience/onboarding/{user_id}")
def onboarding_update(user_id: str, request: dict[str, Any]) -> dict[str, Any]:
    return update_onboarding(user_id, request)


@router.post("/experience/onboarding/{user_id}/first-run")
def onboarding_first_run(user_id: str, request: dict[str, Any]) -> dict[str, Any]:
    return run_first_inspection(user_id, request)


@router.post("/experience/drafts/{user_id}/from-template")
def draft_from_template(user_id: str, request: dict[str, Any]) -> dict[str, Any]:
    return create_from_template(user_id, request)


@router.post("/experience/drafts/{user_id}/from-intent")
def draft_from_intent(user_id: str, request: dict[str, Any]) -> dict[str, Any]:
    return create_from_intent(user_id, request)


@router.get("/experience/drafts/{user_id}/{draft_id}")
def draft(user_id: str, draft_id: int) -> dict[str, Any]:
    return get_draft(user_id, draft_id)


@router.post("/experience/drafts/{user_id}/{draft_id}/enable")
def draft_enable(user_id: str, draft_id: int, request: dict[str, Any], idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")) -> dict[str, Any]:
    return enable_draft(user_id, draft_id, idempotency_key or str(request.get("idempotencyKey") or "") or None)


@router.post("/experience/drafts/{user_id}/{draft_id}/discard")
def draft_discard(user_id: str, draft_id: int) -> dict[str, Any]:
    return discard_draft(user_id, draft_id)


@router.get("/experience/execution/{user_id}/results")
def results(user_id: str, status: str | None = None, projectId: int | None = None, limit: int = Query(default=50, ge=1, le=100)) -> dict[str, Any]:
    return list_results(user_id, status, projectId, limit)


@router.get("/experience/execution/{user_id}/results/{result_id}")
def result(user_id: str, result_id: int) -> dict[str, Any]:
    return get_result(user_id, result_id)


@router.get("/experience/execution/{user_id}/needs-attention")
def attention(user_id: str) -> dict[str, Any]:
    return needs_attention(user_id)


@router.get("/experience/confirmations/{user_id}")
def confirmations(user_id: str) -> dict[str, Any]:
    return list_confirmations(user_id)


@router.get("/experience/confirmations/{user_id}/{operation_id}")
def confirmation(user_id: str, operation_id: int) -> dict[str, Any]:
    return get_confirmation(user_id, operation_id)


@router.post("/experience/confirmations/{user_id}/{operation_id}/confirm")
def confirmation_confirm(user_id: str, operation_id: int) -> dict[str, Any]:
    return confirm(user_id, operation_id)


@router.post("/experience/confirmations/{user_id}/{operation_id}/reject")
def confirmation_reject(user_id: str, operation_id: int) -> dict[str, Any]:
    return reject(user_id, operation_id)


@router.get("/experience/connections/{user_id}/recommended")
def connections_recommended(user_id: str) -> dict[str, Any]:
    return recommended_connections(user_id)


@router.get("/experience/connections/{user_id}/diagnostics")
def connections_diagnostics(user_id: str) -> dict[str, Any]:
    return connection_diagnostics(user_id)


@router.post("/experience/connections/{user_id}/{connection_id}/diagnose")
def connection_diagnose(user_id: str, connection_id: str) -> dict[str, Any]:
    return diagnose_connection(user_id, connection_id)
