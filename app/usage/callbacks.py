from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from datetime import datetime, timezone
from math import isfinite
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
        input_tokens, output_tokens, cache_read_tokens, cache_write_tokens = _extract_usage_details(response)
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
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
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
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
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
            "cacheReadTokens": _int(cache_read_tokens),
            "cacheWriteTokens": _int(cache_write_tokens),
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
    return _extract_usage_details(response)[:2]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_count(*values: Any) -> int | None:
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            if isinstance(value, float) and not isfinite(value):
                continue
            return max(0, int(value))
        except (TypeError, ValueError, OverflowError):
            continue
    return None


def _input_output(usage: Mapping[str, Any]) -> tuple[int, int]:
    return (_first_count(usage.get("prompt_tokens"), usage.get("input_tokens")) or 0,
            _first_count(usage.get("completion_tokens"), usage.get("output_tokens")) or 0)


def _cache_counts(usage: Mapping[str, Any]) -> tuple[int | None, int | None]:
    details = _mapping(usage.get("input_token_details"))
    prompt_details = _mapping(usage.get("prompt_tokens_details"))
    response_details = _mapping(usage.get("input_tokens_details"))
    # These are alternative representations of the same measurements, not
    # independent token buckets to add together. Explicit zero is authoritative.
    return (
        _first_count(usage.get("cache_read_input_tokens"), details.get("cache_read"),
                     details.get("priority_cache_read"), details.get("flex_cache_read"),
                     prompt_details.get("cached_tokens"), response_details.get("cached_tokens")),
        _first_count(usage.get("cache_creation_input_tokens"), details.get("cache_creation")),
    )


def _usage_payload(value: Any) -> Mapping[str, Any]:
    data = _mapping(value)
    return _mapping(data.get("token_usage")) or _mapping(data.get("usage")) or data


def _extract_usage_details(response: Any) -> tuple[int, int, int, int]:
    usage = _usage_payload(getattr(response, "llm_output", None))
    input_tokens, output_tokens = _input_output(usage)
    cache_read, cache_write = _cache_counts(usage)
    message_input = message_output = message_read = message_write = 0
    seen_messages: set[int] = set()

    for generation_batch in getattr(response, "generations", None) or []:
        for generation in generation_batch:
            message = getattr(generation, "message", None)
            if message is None or id(message) in seen_messages:
                continue
            seen_messages.add(id(message))
            metadata = _mapping(getattr(message, "usage_metadata", None))
            provider_metadata = _usage_payload(getattr(message, "response_metadata", None))
            tokens_in, tokens_out = _input_output(metadata)
            if not (tokens_in or tokens_out):
                tokens_in, tokens_out = _input_output(provider_metadata)
            read, write = _cache_counts(metadata)
            raw_read, raw_write = _cache_counts(provider_metadata)
            message_input += tokens_in
            message_output += tokens_out
            message_read += (read if read is not None else raw_read) or 0
            message_write += (write if write is not None else raw_write) or 0
    # llm_output reports the request aggregate; message usage often repeats it.
    # Use messages only for fields absent from that aggregate, never add both.
    if not (input_tokens or output_tokens):
        input_tokens, output_tokens = message_input, message_output
    return (input_tokens, output_tokens,
            cache_read if cache_read is not None else message_read,
            cache_write if cache_write is not None else message_write)


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
    return _first_count(value) or 0
