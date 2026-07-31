from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any
import uuid

from app.core.config import settings
from app.usage.callbacks import UsageBudgetExceeded
from app.usage.context import ModelCallContext, current_model_call_context
from app.usage.reporter import report_usage_event


@dataclass(frozen=True)
class ExternalCallTicket:
    context: ModelCallContext
    started: float
    estimated_input_tokens: int
    reserved_cost_micros: int


def begin_external_call(
    *,
    estimated_input_tokens: int = 0,
    estimated_cost_micros: int = 0,
) -> ExternalCallTicket | None:
    context = current_model_call_context()
    if context is None or not context.user_id:
        return None
    estimated_input_tokens = max(0, int(estimated_input_tokens))
    estimated_cost_micros = max(0, int(estimated_cost_micros))
    with context.lock:
        if context.max_calls is not None and context.call_count + context.pending_calls >= context.max_calls:
            raise UsageBudgetExceeded("单轮 AI 调用次数已达到上限。")
        if (
            context.max_input_tokens is not None
            and context.input_tokens + context.pending_input_tokens + estimated_input_tokens
            > context.max_input_tokens
        ):
            raise UsageBudgetExceeded("本轮输入上下文超过上限。")
        if (
            context.max_cost_micros is not None
            and context.estimated_cost_micros + context.pending_cost_micros + estimated_cost_micros
            > context.max_cost_micros
        ):
            raise UsageBudgetExceeded("本轮 AI 估算费用已达到上限。")
        context.pending_calls += 1
        context.pending_input_tokens += estimated_input_tokens
        context.pending_cost_micros += estimated_cost_micros
    return ExternalCallTicket(
        context=context,
        started=perf_counter(),
        estimated_input_tokens=estimated_input_tokens,
        reserved_cost_micros=estimated_cost_micros,
    )


def finish_external_call(
    ticket: ExternalCallTicket | None,
    *,
    call_type: str,
    task_type: str,
    provider: str,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    usage_source: str = "ESTIMATED",
    estimated_cost_micros: int | None = None,
    status: str = "SUCCEEDED",
    error_category: str | None = None,
) -> None:
    if ticket is None:
        return
    context = ticket.context
    input_tokens = max(0, int(input_tokens or ticket.estimated_input_tokens))
    output_tokens = max(0, int(output_tokens))
    cost_micros = max(
        0,
        int(ticket.reserved_cost_micros if estimated_cost_micros is None else estimated_cost_micros),
    )
    event: dict[str, Any] = {
        "eventId": str(uuid.uuid4()),
        "userId": context.user_id,
        "conversationId": context.conversation_id,
        "turnId": context.turn_id,
        "traceId": context.trace_id,
        "reservationId": context.reservation_id,
        "callType": call_type.upper(),
        "taskType": task_type.upper(),
        "provider": provider,
        "model": model,
        "routeVersion": context.route_version or settings.ai_usage_route_version,
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "cacheReadTokens": 0,
        "cacheWriteTokens": 0,
        "usageSource": usage_source,
        "estimatedCostMicros": cost_micros,
        "pricingVersion": settings.ai_usage_pricing_version,
        "currency": "CNY",
        "latencyMillis": max(0, round((perf_counter() - ticket.started) * 1000)),
        "status": status.upper(),
        "errorCategory": error_category[:64].upper() if error_category else None,
        "occurredAt": datetime.now(timezone.utc).isoformat(),
    }
    with context.lock:
        context.pending_calls = max(0, context.pending_calls - 1)
        context.pending_input_tokens = max(0, context.pending_input_tokens - ticket.estimated_input_tokens)
        context.pending_cost_micros = max(0, context.pending_cost_micros - ticket.reserved_cost_micros)
        context.events.append(event)
        context.call_count += 1
        context.input_tokens += input_tokens
        context.output_tokens += output_tokens
        context.estimated_cost_micros += cost_micros
    report_usage_event(event)


def estimate_text_tokens(value: str) -> int:
    if not value:
        return 0
    cjk = sum(1 for character in value if "\u3400" <= character <= "\u9fff")
    non_cjk = max(0, len(value) - cjk)
    return max(1, cjk + (non_cjk + 3) // 4)
