import asyncio
import logging

from app.core.config import settings
from app.tools.notification_service import generate_all_due_notifications

logger = logging.getLogger(__name__)


async def run_notification_worker(stop_event: asyncio.Event) -> None:
    if not settings.postgres_enabled:
        logger.info("Notification worker skipped because PostgreSQL is disabled.")
        return

    interval = max(5, settings.notification_worker_interval_seconds)
    logger.info("Notification worker started. interval_seconds=%s", interval)

    while not stop_event.is_set():
        try:
            created = generate_all_due_notifications()
            if created:
                logger.info("Notification worker generated %s due reminders.", created)
        except Exception:
            logger.exception("Notification worker failed to generate reminders.")

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except TimeoutError:
            continue

    logger.info("Notification worker stopped.")
