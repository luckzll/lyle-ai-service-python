"""Bounded, context-preserving concurrency for independent retrieval work."""
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import ContextVar, copy_context
from threading import BoundedSemaphore
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")
_MAX_WORKERS = 8
_executor = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="rag-retrieval")
_slots = BoundedSemaphore(_MAX_WORKERS)
_inside_worker: ContextVar[bool] = ContextVar("rag_inside_worker", default=False)


def bounded_map(function: Callable[[T], R], items: Iterable[T], *, max_workers: int = 2) -> list[R]:
    """Return in input order; nested/saturated work falls back to its caller.

    Never enqueue unbounded futures or wait for another executor slot while
    occupying one. Each worker gets its own copy of request/timing/usage context.
    Individual operations retain their existing provider timeouts.
    """
    values = list(items)
    width = max(1, min(_MAX_WORKERS, int(max_workers)))
    if width == 1 or len(values) <= 1 or _inside_worker.get():
        return [function(item) for item in values]

    def invoke(item):
        token = _inside_worker.set(True)
        try:
            return function(item)
        finally:
            _inside_worker.reset(token)

    results: list[R] = []
    slots = _slots
    executor = _executor
    for start in range(0, len(values), width):
        batch: list[Future | tuple[R]] = []
        try:
            for item in values[start:start + width]:
                if not slots.acquire(blocking=False):
                    batch.append((function(item),))
                    continue
                try:
                    context = copy_context()
                    future = executor.submit(context.run, invoke, item)
                except BaseException:
                    slots.release()
                    raise
                # Release even if a queued task is cancelled before it starts.
                future.add_done_callback(lambda completed: slots.release())
                batch.append(future)
            for entry in batch:
                results.append(entry.result() if isinstance(entry, Future) else entry[0])
        except BaseException:
            for entry in batch:
                if isinstance(entry, Future):
                    entry.cancel()
            raise
    return results
