import asyncio
import logging

from app.core.config import settings
from app.memory.schema_builder import process_schema_builds

logger = logging.getLogger(__name__)


async def run_memory_schema_worker(stop_event: asyncio.Event):
    while not stop_event.is_set():
        try:
            await asyncio.to_thread(process_schema_builds)
        except Exception:
            logger.warning("Memory schema queue unavailable", exc_info=False)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=settings.memory_schema_worker_interval)
        except asyncio.TimeoutError:
            pass
