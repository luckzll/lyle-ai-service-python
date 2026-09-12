"""Request-scoped, content-free progress and retrieval timing instrumentation.

LangGraph's executor propagates ContextVars to its node workers. Callers that
create their own threads must enter progress_scope in that worker (or copy the
context). Neither the callback nor its state is stored in the shared graph.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import logging
import math
from threading import Lock
from time import perf_counter


logger = logging.getLogger(__name__)
ProgressCallback = Callable[[dict[str, object]], None]
_callback: ContextVar[ProgressCallback | None] = ContextVar("chat_progress_callback", default=None)

_RAG_PHASES = frozenset({
    "query_rewrite", "recall", "embedding", "vector_search", "keyword_search",
    "rerank", "sufficiency", "merge", "parent_expansion", "qdrant_prepare",
    "diagnostics", "total",
})
TIMING_KEYS = frozenset({f"{phase}_ms" for phase in _RAG_PHASES} | {
    "writer_first_token_ms", "first_delta_ms", "writer_ms", "review_ms", "prepare_context_ms",
} | {
    key for index in range(1, 5)
    for key in (
        f"query_{index}_ms",
        *(f"query_{index}_{phase}_ms" for phase in _RAG_PHASES - {"total", "query_rewrite", "sufficiency", "diagnostics"}),
    )
})


def sanitize_timings(value: object) -> dict[str, int | float]:
    """Only finite, nonnegative durations with bounded, known keys may cross API."""
    if not isinstance(value, dict):
        return {}
    return {
        key: round(float(duration), 3)
        for key, duration in value.items()
        if key in TIMING_KEYS and type(duration) in (int, float)
        and 0 <= duration <= 86_400_000 and math.isfinite(duration)
    }


@contextmanager
def progress_scope(callback: ProgressCallback | None) -> Iterator[None]:
    token = _callback.set(callback)
    try:
        yield
    finally:
        _callback.reset(token)


def emit_step(
    step: str,
    status: str,
    *,
    duration_ms: int | float | None = None,
    summary: str | None = None,
    timings: dict[str, object] | None = None,
) -> None:
    """Emit instrumentation without letting a disconnected observer fail a task.

    Summary must be application-authored status text, never provider output,
    prompts, user queries, or exception messages. API layer adds turnId/sequence.
    """
    callback = _callback.get()
    if callback is None or status not in {"running", "completed", "failed", "skipped"}:
        return
    payload: dict[str, object] = {"step": step, "status": status}
    if type(duration_ms) in (int, float) and 0 <= duration_ms <= 86_400_000 and math.isfinite(duration_ms):
        payload["durationMs"] = round(duration_ms, 3)
    if summary:
        payload["summary"] = summary[:240]
    if timings:
        payload["timings"] = sanitize_timings(timings)
    try:
        callback(payload)
    except Exception:
        # Do not log callback exception content: it may contain request data.
        logger.warning("Chat progress observer unavailable; task continues.")


@dataclass
class RagTimings:
    values: dict[str, float] = field(default_factory=dict)
    lock: Lock = field(default_factory=Lock)

    def add(self, key: str, duration: float) -> None:
        if key in TIMING_KEYS:
            with self.lock:
                self.values[key] = self.values.get(key, 0.0) + duration

    def snapshot(self) -> dict[str, int | float]:
        with self.lock:
            return sanitize_timings(self.values)


_rag_timings: ContextVar[RagTimings | None] = ContextVar("rag_timings", default=None)
_rag_query_index: ContextVar[int | None] = ContextVar("rag_query_index", default=None)


@contextmanager
def rag_timing_scope() -> Iterator[RagTimings]:
    timings = RagTimings()
    token = _rag_timings.set(timings)
    query_token = _rag_query_index.set(None)
    try:
        yield timings
    finally:
        _rag_query_index.reset(query_token)
        _rag_timings.reset(token)


@contextmanager
def rag_phase(phase: str) -> Iterator[None]:
    timings = _rag_timings.get()
    started = perf_counter()
    try:
        yield
    finally:
        if timings is not None:
            duration = (perf_counter() - started) * 1000
            timings.add(f"{phase}_ms", duration)
            index = _rag_query_index.get()
            if index is not None:
                timings.add(f"query_{index}_{phase}_ms", duration)


@contextmanager
def rag_query(index: int) -> Iterator[None]:
    token = _rag_query_index.set(index)
    timings = _rag_timings.get()
    started = perf_counter()
    try:
        yield
    finally:
        if timings is not None:
            timings.add(f"query_{index}_ms", (perf_counter() - started) * 1000)
        _rag_query_index.reset(token)
