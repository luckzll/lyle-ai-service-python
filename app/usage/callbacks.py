from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from time import perf_counter
from typing import Any
import uuid

from langchain_core.callbacks.base import BaseCallbackHandler

from app.core.config import settings
from app.usage.context import ModelCallContext, current_model_call_context
from app.usage.reporter import report_usage_event


class UsageBudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class _RunReservation:
    context: ModelCallContext | None
    started: float
    estimated_input_tokens: int
    reserved_cost_micros: int


class UsageCallbackHandler(BaseCallbackHandler):
    raise_error = True

    def __init__(self, task_type: str, provider: str, model: str) -> None:
        self.task_type = task_type.upper()
        self.provider = provider
        self.model = model
        self._runs: dict[str, _RunReservation] = {}
        self._lock = Lock()

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        self._start(run_id, sum(_estimate_tokens(prompt) for prompt in prompts))

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        prompt_tokens = sum(
            _estimate_tokens(str(getattr(message, "content", "")))
            for batch in messages
            for message in batch
        )
        self._start(run_id, prompt_tokens)

    def on_llm_end(self, response: Any, *, run_id: uuid.UUID, **kwargs: Any) -> None:
        reservation = self._pop_run(run_id)
        input_tokens, output_tokens = _extract_usage(response)
        usage_source = "PROVIDER_REPORTED"
        if input_tokens <= 0 and output_tokens <= 0:
            input_tokens = reservation.estimated_input_tokens
            output_tokens = _estimate_output_tokens(response)
            usage_source = "ESTIMATED"
        self._record(
            run_id=run_id,
            reservation=reservation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usage_source=usage_source,
            status="SUCCEEDED",
            error_category=None,
        )

    def on_llm_error(self, error: BaseException, *, run_id: uuid.UUID, **kwargs: Any) -> None:
        reservation = self._pop_run(run_id)
        self._record(
            run_id=run_id,
            reservation=reservation,
            input_tokens=reservation.estimated_input_tokens,
            output_tokens=0,
            usage_source="ESTIMATED",
            status="FAILED",
            error_category=type(error).__name__[:64].upper(),
        )

    def _start(self, run_id: uuid.UUID, estimated_input_tokens: int) -> None:
        context = current_model_call_context()
        projected_input_cost = max(
            0,
            round(estimated_input_tokens * settings.ai_usage_input_price_cny_per_million),
        )
        if context is not None:
            with context.lock:
                if (
                    context.max_calls is not None
                    and context.call_count + context.pending_calls >= context.max_calls
                ):
                    raise UsageBudgetExceeded("单轮 AI 调用次数已达到上限。")
                if (
                    context.max_input_tokens is not None
                    and context.input_tokens + context.pending_input_tokens + estimated_input_tokens
                    > context.max_input_tokens
                ):
                    raise UsageBudgetExceeded("本轮输入上下文超过上限。")
                if context.max_output_tokens is not None and context.output_tokens >= context.max_output_tokens:
                    raise UsageBudgetExceeded("本轮输出用量已达到上限。")
                if (
                    context.max_cost_micros is not None
                    and context.estimated_cost_micros + context.pending_cost_micros + projected_input_cost
                    > context.max_cost_micros
                ):
                    raise UsageBudgetExceeded("本轮 AI 估算费用已达到上限。")
                # Reserve before leaving the lock. Parallel model callbacks for
                # the same Turn must observe this in-flight request.
                context.pending_calls += 1
                context.pending_input_tokens += estimated_input_tokens
                context.pending_cost_micros += projected_input_cost
        with self._lock:
            self._runs[str(run_id)] = _RunReservation(
                context=context,
                started=perf_counter(),
                estimated_input_tokens=estimated_input_tokens,
                reserved_cost_micros=projected_input_cost,
            )

    def _pop_run(self, run_id: uuid.UUID) -> _RunReservation:
        with self._lock:
            return self._runs.pop(
                str(run_id),
                _RunReservation(
                    context=current_model_call_context(),
                    started=perf_counter(),
                    estimated_input_tokens=0,
                    reserved_cost_micros=0,
                ),
            )

    def _record(
        self,
        *,
        run_id: uuid.UUID,
        reservation: _RunReservation,
        input_tokens: int,
        output_tokens: int,
        usage_source: str,
        status: str,
        error_category: str | None,
    ) -> None:
        context = reservation.context
        if context is None:
            return
        if not context.user_id:
            with context.lock:
                context.pending_calls = max(0, context.pending_calls - 1)
                context.pending_input_tokens = max(
                    0,
                    context.pending_input_tokens - reservation.estimated_input_tokens,
                )
                context.pending_cost_micros = max(
                    0,
                    context.pending_cost_micros - reservation.reserved_cost_micros,
                )
            return
        estimated_cost_micros = max(
            0,
            round(
                input_tokens * settings.ai_usage_input_price_cny_per_million
                + output_tokens * settings.ai_usage_output_price_cny_per_million
            ),
        )
        event = {
            "eventId": str(uuid.uuid4()),
            "userId": context.user_id,
            "conversationId": context.conversation_id,
            "turnId": context.turn_id,
            "traceId": context.trace_id,
            "reservationId": context.reservation_id,
            "callType": "LLM_TEXT",
            "taskType": self.task_type,
            "provider": self.provider,
            "model": self.model,
            "routeVersion": context.route_version or settings.ai_usage_route_version,
            "inputTokens": max(0, input_tokens),
            "outputTokens": max(0, output_tokens),
            "cacheReadTokens": 0,
            "cacheWriteTokens": 0,
            "usageSource": usage_source,
            "estimatedCostMicros": estimated_cost_micros,
            "pricingVersion": settings.ai_usage_pricing_version,
            "currency": "CNY",
            "latencyMillis": max(0, round((perf_counter() - reservation.started) * 1000)),
            "status": status,
            "errorCategory": error_category,
            "occurredAt": datetime.now(timezone.utc).isoformat(),
        }
        with context.lock:
            context.pending_calls = max(0, context.pending_calls - 1)
            context.pending_input_tokens = max(
                0,
                context.pending_input_tokens - reservation.estimated_input_tokens,
            )
            context.pending_cost_micros = max(
                0,
                context.pending_cost_micros - reservation.reserved_cost_micros,
            )
            context.events.append(event)
            context.call_count += 1
            context.input_tokens += max(0, input_tokens)
            context.output_tokens += max(0, output_tokens)
            context.estimated_cost_micros += estimated_cost_micros
            if context.max_output_tokens is not None and context.output_tokens > context.max_output_tokens:
                # 已发出的供应商请求正常结算；下一次调用会被 max_calls/预算策略阻止。
                pass
        report_usage_event(event)


def _extract_usage(response: Any) -> tuple[int, int]:
    llm_output = getattr(response, "llm_output", None) or {}
    usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    input_tokens = _int(usage.get("prompt_tokens") or usage.get("input_tokens"))
    output_tokens = _int(usage.get("completion_tokens") or usage.get("output_tokens"))
    if input_tokens or output_tokens:
        return input_tokens, output_tokens

    for generation_batch in getattr(response, "generations", None) or []:
        for generation in generation_batch:
            message = getattr(generation, "message", None)
            metadata = getattr(message, "usage_metadata", None) or {}
            input_tokens += _int(metadata.get("input_tokens"))
            output_tokens += _int(metadata.get("output_tokens"))
    return input_tokens, output_tokens


def _estimate_output_tokens(response: Any) -> int:
    parts: list[str] = []
    for generation_batch in getattr(response, "generations", None) or []:
        for generation in generation_batch:
            text = getattr(generation, "text", None)
            if text:
                parts.append(str(text))
                continue
            message = getattr(generation, "message", None)
            content = getattr(message, "content", "")
            if content:
                parts.append(str(content))
    return sum(_estimate_tokens(part) for part in parts)


def _estimate_tokens(value: str) -> int:
    if not value:
        return 0
    # 无供应商 usage 时只作为明确标记的治理估算。中文按字符、英文按近似子词折算。
    cjk = sum(1 for character in value if "\u3400" <= character <= "\u9fff")
    non_cjk = max(0, len(value) - cjk)
    return max(1, cjk + (non_cjk + 3) // 4)


def _int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0
