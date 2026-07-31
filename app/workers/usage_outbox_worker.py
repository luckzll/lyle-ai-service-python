import asyncio
import logging

from app.core.config import settings
from app.usage.reporter import deliver_pending_events


logger = logging.getLogger(__name__)


async def run_usage_outbox_worker(stop_event: asyncio.Event) -> None:
    interval = max(5, settings.ai_usage_outbox_worker_interval_seconds)
    while not stop_event.is_set():
        try:
            delivered = await asyncio.to_thread(deliver_pending_events)
            if delivered:
                logger.info("Delivered pending AI usage events. count=%s", delivered)
        except Exception:
            logger.exception("AI usage outbox worker failed.")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            continue
