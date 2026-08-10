from __future__ import annotations

import asyncio
import logging

from app.core.config import settings
from app.memory.governance import process_pending_purge_jobs


logger = logging.getLogger(__name__)


async def run_memory_purge_worker(stop_event: asyncio.Event) -> None:
    interval = max(1, settings.memory_purge_worker_interval_seconds)
    while not stop_event.is_set():
        try:
            result = await asyncio.to_thread(process_pending_purge_jobs, limit=5)
            if result["claimed"]:
                logger.info("Memory purge worker completed. result=%s", result)
        except Exception:
            logger.exception("Memory purge worker iteration failed")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except TimeoutError:
            continue
