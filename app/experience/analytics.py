"""Aggregate product experience metrics without exposing user-level data."""

from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.core.database import postgres_connection


def experience_analytics(days: int = 7) -> dict[str, Any]:
    days = max(1, min(days, 90))
    empty = {
        "periodDays": days,
        "funnel": {"onboardingStarted": 0, "firstRunStarted": 0, "firstRunCompleted": 0, "draftCreated": 0, "draftEnabled": 0},
        "templates": [], "executionByStatus": [],
        "confirmations": {"viewed": 0, "confirmed": 0, "rejected": 0},
        "failureCategories": [],
    }
    if not settings.postgres_enabled:
        return empty
    with postgres_connection() as conn:
        event_rows = conn.execute(
            """SELECT event_type, COUNT(*) FROM experience_product_event
               WHERE created_at >= now() - (%s * interval '1 day') GROUP BY event_type""", (days,)
        ).fetchall()
        template_rows = conn.execute(
            """SELECT template_id, COUNT(*) FROM experience_product_event
               WHERE event_type='draft_enabled' AND template_id IS NOT NULL
               AND created_at >= now() - (%s * interval '1 day')
               GROUP BY template_id ORDER BY COUNT(*) DESC""", (days,)
        ).fetchall()
        execution_rows = conn.execute(
            """SELECT status, COUNT(*) FROM execution_result_summary
               WHERE created_at >= now() - (%s * interval '1 day')
               GROUP BY status ORDER BY status""", (days,)
        ).fetchall()
        failure_rows = conn.execute(
            """SELECT status, COUNT(*) FROM execution_result_summary
               WHERE status IN ('failed', 'partial') AND created_at >= now() - (%s * interval '1 day')
               GROUP BY status ORDER BY status""", (days,)
        ).fetchall()
    counts = {str(event_type): int(count) for event_type, count in event_rows}
    return {
        "periodDays": days,
        "funnel": {
            "onboardingStarted": counts.get("onboarding_started", 0),
            "firstRunStarted": counts.get("first_run_started", 0),
            "firstRunCompleted": counts.get("first_run_completed", 0),
            "draftCreated": counts.get("draft_created", 0),
            "draftEnabled": counts.get("draft_enabled", 0),
        },
        "templates": [{"templateId": template_id, "enabled": int(count)} for template_id, count in template_rows],
        "executionByStatus": [{"status": status, "count": int(count)} for status, count in execution_rows],
        "confirmations": {"viewed": counts.get("confirmation_viewed", 0), "confirmed": counts.get("confirmation_confirmed", 0), "rejected": counts.get("confirmation_rejected", 0)},
        "failureCategories": [{"category": status, "count": int(count)} for status, count in failure_rows],
    }
