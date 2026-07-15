from typing import Any, Literal

from fastapi import APIRouter, Header, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.experience.confirmation_inbox import confirm, get_confirmation, list_confirmations, reject
from app.experience.connection_diagnostics import connection_diagnostics, diagnose_connection, recommended_connections
from app.experience.draft_service import create_from_intent, create_from_template, discard_draft, enable_draft, get_draft
from app.experience.onboarding_store import get_onboarding, run_first_inspection, update_onboarding
from app.experience.result_summary_service import get_result, list_results, needs_attention
from app.experience.template_catalog import get_template, list_templates
from app.experience.analytics import experience_analytics
from app.experience.events import EFFECTIVE_PROGRESS_ACTIONS, EVENT_TYPES, record_event


router = APIRouter()


ProductPage = Literal[
    "dashboard", "chat", "tasks", "projects", "automations", "execution-center",
    "connections", "knowledge-bases", "conversations", "memory", "advanced-settings",
    "settings", "evaluation", "prompts", "admin",
]
ProductAction = Literal[
    "freeform", "organize-project", "make-plan", "research", "inspect", "start_chat",
    "run", "pause", "resume", "delete", "edit", "attention", "view_result", "view",
    "confirm", "reject", "copy", "create_task", "create_automation", "run_inspection",
    "diagnose", "retry", "open", "close", "save", "enable", "disable", "skip",
    "refresh", "restore", "filter_change", "view_confirmations", "view_attention",
    "view_results", "view_history",
    "task_complete", "task_reschedule", "confirmation", "automation_enable",
    "automation_run", "result_action", "project_chat",
]
ProductStatus = Literal[
    "draft", "enabled", "disabled", "paused", "running", "completed", "partial",
    "failed", "awaiting_confirmation", "pending", "confirmed", "rejected", "deleted",
    "healthy", "reauthorize_required", "restricted", "unavailable", "unknown",
    "succeeded", "skipped", "open", "accepted",
]
ProductErrorCategory = Literal[
    "network", "http", "validation", "permission", "rate_limit", "not_found",
    "timeout", "server", "unknown",
]


class ProductEventRequest(BaseModel):
    """Strict event envelope: arbitrary user text has no field that can accept it."""

    model_config = ConfigDict(extra="ignore")

    eventType: str = Field(min_length=1, max_length=64)
    page: ProductPage | None = None
    actionType: ProductAction | None = None
    templateId: str | None = Field(default=None, max_length=64)
    projectId: int | None = Field(default=None, gt=0)
    status: ProductStatus | None = None
    durationMs: int | None = Field(default=None, ge=0, le=9_223_372_036_854_775_807)
    errorCategory: ProductErrorCategory | None = None

    @field_validator("eventType")
    @classmethod
    def validate_event_type(cls, value: str) -> str:
        if value not in EVENT_TYPES:
            raise ValueError("unsupported product event")
        return value

    @field_validator("templateId")
    @classmethod
    def validate_template_id(cls, value: str | None) -> str | None:
        if value is not None and get_template(value) is None:
            raise ValueError("unknown template")
        return value

    @model_validator(mode="after")
    def validate_effective_progress_action(self) -> "ProductEventRequest":
        if self.eventType != "effective_progress_action":
            return self
        if self.actionType not in EFFECTIVE_PROGRESS_ACTIONS:
            raise ValueError("unsupported effective progress action")
        expected_page = {
            "task_complete": "tasks",
            "task_reschedule": "tasks",
            "confirmation": "execution-center",
            "automation_enable": "automations",
            "automation_run": "automations",
            "result_action": "execution-center",
            "project_chat": "chat",
        }[self.actionType]
        if self.page != expected_page:
            raise ValueError("effective progress action does not match page")
        if self.actionType == "project_chat" and self.projectId is None:
            raise ValueError("project context chat requires project id")
        return self


@router.get("/experience/analytics")
def analytics(days: int = Query(default=7, ge=1, le=90)) -> dict[str, Any]:
    return experience_analytics(days)


@router.post("/experience/events/{user_id}")
def product_event(user_id: str, request: ProductEventRequest) -> dict[str, bool]:
    record_event(
        user_id,
        request.eventType,
        template_id=request.templateId,
        status=request.status,
        duration_ms=request.durationMs,
        metadata={
            "page": request.page,
            "actionType": request.actionType,
            "projectId": request.projectId,
            "errorCategory": request.errorCategory,
        },
    )
    return {"accepted": True}


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
def results(
    user_id: str,
    status: str | None = None,
    projectId: int | None = None,
    automationId: int | None = Query(default=None, gt=0),
    cursor: str | None = Query(default=None, max_length=256),
    limit: int = Query(default=50, ge=1, le=100),
) -> dict[str, Any]:
    return list_results(
        user_id,
        status,
        projectId,
        limit,
        cursor=cursor,
        automation_id=automationId,
    )


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
