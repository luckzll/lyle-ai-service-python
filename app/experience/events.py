"""Minimal, privacy-preserving product event recording."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from app.core.config import settings
from app.core.database import postgres_connection


logger = logging.getLogger(__name__)

EVENT_TYPES = {
    "onboarding_started", "onboarding_skipped", "first_run_started", "first_run_completed",
    "template_viewed", "draft_created", "draft_enabled", "draft_discarded", "result_viewed",
    "confirmation_viewed", "confirmation_confirmed", "confirmation_rejected",
    "automation_failed", "retry_started", "connection_diagnostic_started",
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
    safe_metadata = {
        key: value for key, value in (metadata or {}).items()
        if key in {"source", "action", "errorCategory", "projectSelected", "workspaceSelected"}
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
