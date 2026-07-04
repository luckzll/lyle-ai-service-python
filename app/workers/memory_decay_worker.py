import asyncio
import logging

from app.core.config import settings
from app.memory.profile_store import apply_profile_decay

logger = logging.getLogger(__name__)


async def run_memory_decay_worker(stop_event: asyncio.Event) -> None:
    interval = max(60, settings.memory_decay_worker_interval_seconds)
    logger.info("Memory decay worker started. interval_seconds=%s", interval)

    while not stop_event.is_set():
        try:
            result = apply_profile_decay()
            if result["updated"]:
                logger.info(
                    "Memory decay worker updated profiles. updated=%s expired=%s",
                    result["updated"],
                    result["expired"],
                )
        except Exception:
            logger.exception("Memory decay worker failed.")

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except TimeoutError:
            continue

    logger.info("Memory decay worker stopped.")
