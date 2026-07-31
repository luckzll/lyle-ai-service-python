from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from typing import Any
import uuid


@dataclass
class ModelCallContext:
    user_id: str
    conversation_id: str | None
    turn_id: str | None
    trace_id: str
    reservation_id: str | None
    route_version: str
    max_calls: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_cost_micros: int | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_micros: int = 0
    pending_calls: int = 0
    pending_input_tokens: int = 0
    pending_cost_micros: int = 0
    lock: Lock = field(default_factory=Lock, repr=False)


_current_context: ContextVar[ModelCallContext | None] = ContextVar(
    "lyle_model_call_context",
    default=None,
)


def activate_model_call_context(request: Any) -> ModelCallContext:
    budget = getattr(request, "usageBudget", None) or {}
    context = ModelCallContext(
        user_id=str(getattr(request, "user_id", None) or getattr(request, "userId", "")),
        conversation_id=getattr(request, "conversation_id", None),
        turn_id=getattr(request, "turn_id", None),
        trace_id=str(getattr(request, "traceId", None) or uuid.uuid4()),
        reservation_id=getattr(request, "reservationId", None),
        route_version="v6.4-1",
        max_calls=_positive_int(budget.get("maxCalls")),
        max_input_tokens=_positive_int(budget.get("maxInputTokens")),
        max_output_tokens=_positive_int(budget.get("maxOutputTokens")),
        max_cost_micros=_positive_int(budget.get("maxCostMicros")),
    )
    _current_context.set(context)
    return context


def current_model_call_context() -> ModelCallContext | None:
    return _current_context.get()


def set_model_call_context(context: ModelCallContext | None) -> None:
    _current_context.set(context)


def usage_totals(context: ModelCallContext | None = None) -> dict[str, int | str]:
    selected = context or current_model_call_context()
    if selected is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "call_count": 0,
            "estimated_cost_micros": 0,
            "measurement_source": "PENDING",
        }
    with selected.lock:
        sources = {str(item.get("usageSource") or "") for item in selected.events}
        source = (
            "PENDING"
            if not sources
            else "PROVIDER_REPORTED"
            if sources == {"PROVIDER_REPORTED"}
            else "ESTIMATED"
            if sources == {"ESTIMATED"}
            else "PARTIAL_ESTIMATE"
        )
        return {
            "prompt_tokens": selected.input_tokens,
            "completion_tokens": selected.output_tokens,
            "call_count": selected.call_count,
            "estimated_cost_micros": selected.estimated_cost_micros,
            "measurement_source": source,
        }


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
