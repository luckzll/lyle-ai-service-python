from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import httpx
from psycopg.types.json import Jsonb

from app.core.config import settings
from app.core.database import postgres_connection


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeliveryOutcome:
    delivered_ids: frozenset[str]
    rejected: dict[str, str]


def report_usage_event(event: dict[str, Any]) -> None:
    if not settings.ai_usage_reporting_enabled:
        return
    persisted = _persist(event)
    if persisted:
        # Production requests only append to the durable outbox. The worker owns
        # network delivery so provider callbacks never wait on the Java ledger.
        return
    # Local development can run without Postgres; retain a synchronous fallback
    # there so usage remains observable without pretending it is durable.
    _deliver([event])


def deliver_pending_events(limit: int = 100) -> int:
    if not settings.postgres_enabled or not settings.ai_usage_reporting_enabled:
        return 0
    try:
        with postgres_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT event_id, payload
                    FROM usage_event_outbox
                    WHERE delivered_at IS NULL
                      AND next_attempt_at <= NOW()
                    ORDER BY created_at
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                    """,
                    (max(1, min(limit, 200)),),
                )
                rows = cursor.fetchall()
    except Exception:
        logger.exception("Failed to read AI usage outbox.")
        return 0
    if not rows:
        return 0
    events = [dict(row[1]) for row in rows]
    event_ids = [str(row[0]) for row in rows]
    outcome = _deliver(events)
    if outcome is None:
        _mark_retry(event_ids)
        return 0

    known_ids = set(event_ids)
    delivered_ids = sorted(known_ids.intersection(outcome.delivered_ids))
    rejected = {
        event_id: reason
        for event_id, reason in outcome.rejected.items()
        if event_id in known_ids
    }
    accounted_ids = set(delivered_ids).union(rejected)
    omitted_ids = sorted(known_ids.difference(accounted_ids))

    _mark_delivered(delivered_ids)
    _mark_terminal_rejected(rejected)
    # A syntactically valid 2xx response that omits events is not proof of
    # delivery. Retry only those unaccounted events.
    _mark_retry(omitted_ids)
    return len(delivered_ids)


def _persist(event: dict[str, Any]) -> bool:
    if not settings.postgres_enabled:
        return False
    try:
        with postgres_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO usage_event_outbox (
                        event_id, payload, created_at, next_attempt_at
                    ) VALUES (%s, %s, NOW(), NOW())
                    ON CONFLICT (event_id) DO NOTHING
                    """,
                    (event["eventId"], Jsonb(event)),
                )
        return True
    except Exception:
        logger.exception("Failed to persist AI usage event. event_id=%s", event.get("eventId"))
        return False


def _deliver(events: list[dict[str, Any]]) -> DeliveryOutcome | None:
    if not events:
        return DeliveryOutcome(frozenset(), {})
    url = settings.ai_usage_java_base_url.rstrip("/") + "/internal/ai-usage/events:batch"
    try:
        response = httpx.post(
            url,
            headers={"X-Usage-Internal-Token": settings.ai_usage_internal_token},
            json={"events": events},
            timeout=settings.ai_usage_reporting_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        accepted_ids = {
            str(event_id)
            for event_id in (payload.get("acceptedEventIds") or [])
            if event_id
        }
        duplicate_ids = {
            str(event_id)
            for event_id in (payload.get("duplicateEventIds") or [])
            if event_id
        }
        rejected = {
            str(item.get("eventId")): str(item.get("reason") or "REJECTED")
            for item in (payload.get("rejectedEvents") or [])
            if item and item.get("eventId")
        }
        if rejected:
            logger.error("AI usage events rejected by Java ledger. rejected=%s", rejected)
        return DeliveryOutcome(frozenset(accepted_ids.union(duplicate_ids)), rejected)
    except Exception:
        logger.exception(
            "Failed to deliver AI usage events. count=%s endpoint=%s",
            len(events),
            url,
        )
        return None


def _mark_delivered(event_ids: list[str]) -> None:
    if not settings.postgres_enabled or not event_ids:
        return
    try:
        with postgres_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE usage_event_outbox
                    SET delivered_at = NOW(), last_error = NULL
                    WHERE event_id = ANY(%s)
                    """,
                    (event_ids,),
                )
    except Exception:
        logger.exception("Failed to mark AI usage events delivered.")


def _mark_terminal_rejected(rejected: dict[str, str]) -> None:
    if not settings.postgres_enabled or not rejected:
        return
    try:
        with postgres_connection() as connection:
            with connection.cursor() as cursor:
                for event_id, reason in rejected.items():
                    cursor.execute(
                        """
                        UPDATE usage_event_outbox
                        SET delivered_at = NOW(),
                            attempt_count = attempt_count + 1,
                            last_error = %s
                        WHERE event_id = %s
                        """,
                        (f"REJECTED:{reason}"[:255], event_id),
                    )
    except Exception:
        logger.exception("Failed to terminalize rejected AI usage events.")


def _mark_retry(event_ids: list[str]) -> None:
    if not settings.postgres_enabled or not event_ids:
        return
    try:
        with postgres_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE usage_event_outbox
                    SET attempt_count = attempt_count + 1,
                        next_attempt_at = NOW() + (
                            LEAST(300, POWER(2, LEAST(attempt_count + 1, 8))) * INTERVAL '1 second'
                        ),
                        last_error = 'DELIVERY_FAILED'
                    WHERE event_id = ANY(%s)
                    """,
                    (event_ids,),
                )
    except Exception:
        logger.exception("Failed to schedule AI usage outbox retry.")
