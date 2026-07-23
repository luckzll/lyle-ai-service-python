"""Minimal, privacy-preserving product event recording."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from app.core.config import settings
from app.core.database import postgres_connection


logger = logging.getLogger(__name__)

EFFECTIVE_PROGRESS_ACTIONS = frozenset({
    "task_complete",
    "task_reschedule",
    "confirmation",
    "automation_enable",
    "automation_run",
    "result_action",
    "project_chat",
})

EVENT_TYPES = {
    "onboarding_started", "onboarding_skipped", "first_run_started", "first_run_completed",
    "template_viewed", "draft_created", "draft_enabled", "draft_discarded", "result_viewed",
    "confirmation_viewed", "confirmation_confirmed", "confirmation_rejected",
    "automation_failed", "retry_started", "connection_diagnostic_started",
    "dashboard_viewed", "dashboard_action_clicked", "quick_intent_submitted",
    "navigation_group_opened", "context_panel_opened", "automation_card_actioned",
    "execution_filter_applied", "result_source_opened", "result_dismissed", "confirmation_completed",
    "mobile_primary_action_used", "ui_error_recovered", "effective_progress_action",
    "auth_view_total", "auth_submit_total", "auth_return_to_target_total",
    "chat_drag_overlay_total", "chat_attachment_added_total", "chat_folder_scan_total",
    "chat_folder_skip_total",
    "chat_turn_submitted_total",
}


def record_event(
    user_id: str,
    event_type: str,
    *,
    template_id: str | None = None,
    status: str | None = None,
    duration_ms: int | None = None,
    metadata: dict[str, str | int | bool | None] | None = None,
) -> None:
    """Never let analytics influence the user-facing execution path."""
    if not settings.postgres_enabled or event_type not in EVENT_TYPES:
        return
    if template_id is not None and len(template_id) > 64:
        logger.warning("Dropping an overlong product-event template id")
        template_id = None
    if status is not None and len(status) > 32:
        logger.warning("Dropping an overlong product-event status")
        status = None
    if duration_ms is not None and not 0 <= duration_ms <= 9_223_372_036_854_775_807:
        logger.warning("Dropping an out-of-range product-event duration")
        duration_ms = None
    safe_metadata = {
        key: value for key, value in (metadata or {}).items()
        if key in {
            "source", "action", "actionType", "page", "projectId", "entityId",
            "errorCategory", "projectSelected", "workspaceSelected", "itemCount"
        }
        and isinstance(value, (str, int, bool, type(None)))
    }
    try:
        with postgres_connection() as conn:
            conn.execute(
                """INSERT INTO experience_product_event
                  (anonymous_user_id, event_type, template_id, status, duration_ms, metadata_json)
                  VALUES (%s, %s, %s, %s, %s, %s::jsonb)""",
                (_anonymous_user_id(user_id), event_type, template_id, status, duration_ms,
                 json.dumps(safe_metadata, ensure_ascii=False)),
            )
            conn.commit()
    except Exception:
        logger.exception("Unable to record product event %s", event_type)


def _anonymous_user_id(user_id: str) -> str:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:32]
