import asyncio
import logging

from app.automation.automation_runner import due_automations, prime_event_automations, run_automation
from app.core.config import settings

logger = logging.getLogger(__name__)


async def run_automation_scheduler_worker(stop_event: asyncio.Event) -> None:
    logger.info("Automation scheduler worker started. interval_seconds=%s", settings.automation_worker_interval_seconds)
    while not stop_event.is_set():
        try:
            prime_event_automations()
            for item in due_automations():
                try:
                    run_automation(str(item["userId"]), int(item["id"]), scheduled=True)
                except Exception:
                    logger.exception("Automation run failed. automation_id=%s user_id=%s", item["id"], item["userId"])
        except Exception:
            logger.exception("Automation scheduler scan failed.")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=settings.automation_worker_interval_seconds)
        except TimeoutError:
            continue
    logger.info("Automation scheduler worker stopped.")
