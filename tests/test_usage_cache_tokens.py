import json
from types import SimpleNamespace
import uuid

import pytest

from app.core.config import settings
from app.usage.callbacks import UsageCallbackHandler, _extract_usage_details
from app.usage.context import ModelCallContext, set_model_call_context


def response(*, aggregate=None, messages=()):
    return SimpleNamespace(llm_output=aggregate or {}, generations=[[
        SimpleNamespace(message=SimpleNamespace(usage_metadata=usage, response_metadata=raw))
        for usage, raw in messages
    ]])


@pytest.mark.parametrize(("result", "expected"), [
    (response(aggregate={"token_usage": {"prompt_tokens": 100, "completion_tokens": 10,
                                        "prompt_tokens_details": {"cached_tokens": 64}}}), (100, 10, 64, 0)),
    (response(aggregate={"usage": {"input_tokens": 23, "output_tokens": 12,
                                   "cache_read_input_tokens": 17, "cache_creation_input_tokens": 8}}), (23, 12, 17, 8)),
    (response(messages=[({"input_tokens": 40, "output_tokens": 9,
                          "input_token_details": {"cache_read": 12, "cache_creation": 5}}, {})]), (40, 9, 12, 5)),
    (response(messages=[({}, {"token_usage": {"prompt_tokens": 30, "completion_tokens": 4,
                                              "prompt_tokens_details": {"cached_tokens": 20}}})]), (30, 4, 20, 0)),
    (response(aggregate={"usage": {"input_tokens": 80, "output_tokens": 5,
                                   "input_tokens_details": {"cached_tokens": 72}}}), (80, 5, 72, 0)),
])
def test_preserves_actual_provider_cache_usage_formats(result, expected):
    assert _extract_usage_details(result) == expected


def test_aggregate_and_message_usage_are_not_added_twice():
    payload = {"input_tokens": 100, "output_tokens": 10, "input_token_details": {"cache_read": 80, "cache_creation": 5}}
    assert _extract_usage_details(response(aggregate={"usage": payload}, messages=[(payload, {"usage": payload})])) == (100, 10, 80, 5)


def test_missing_aggregate_cache_details_can_use_message_details_without_adding_input_twice():
    result = response(aggregate={"usage": {"input_tokens": 200, "output_tokens": 20}}, messages=[
        ({"input_tokens": 100, "output_tokens": 10, "input_token_details": {"cache_read": 80}}, {}),
        ({"input_tokens": 100, "output_tokens": 10, "input_token_details": {"cache_read": 60, "cache_creation": 7}}, {}),
    ])
    assert _extract_usage_details(result) == (200, 20, 140, 7)


def test_explicit_aggregate_cache_zero_is_authoritative():
    result = response(aggregate={"usage": {"input_tokens": 100, "output_tokens": 10,
                                          "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}},
                      messages=[({"input_tokens": 100, "output_tokens": 10,
                                  "input_token_details": {"cache_read": 80, "cache_creation": 7}}, {})])
    assert _extract_usage_details(result) == (100, 10, 0, 0)


def test_standard_message_and_raw_message_cache_aliases_are_not_summed():
    result = response(messages=[({"input_tokens": 100, "output_tokens": 10,
                                  "input_token_details": {"cache_read": 80, "cache_creation": 5}},
                                 {"usage": {"cache_read_input_tokens": 80, "cache_creation_input_tokens": 5}})])
    assert _extract_usage_details(result) == (100, 10, 80, 5)


@pytest.mark.parametrize("tier", ["priority", "flex"])
def test_langchain_service_tier_cache_read_is_preserved(tier):
    result = response(messages=[({"input_tokens": 100, "output_tokens": 10,
                                  "input_token_details": {f"{tier}_cache_read": 80, tier: 20}}, {})])
    assert _extract_usage_details(result) == (100, 10, 80, 0)


def test_langchain_service_tier_aliases_do_not_duplicate_cache_read():
    result = response(messages=[({"input_tokens": 100, "output_tokens": 10,
                                  "input_token_details": {"cache_read": 80, "priority_cache_read": 80,
                                                          "flex_cache_read": 80}}, {})])
    assert _extract_usage_details(result) == (100, 10, 80, 0)


@pytest.mark.parametrize("invalid", [-5, float("nan"), float("inf"), float("-inf"), "not-a-number", True, [], {}])
def test_invalid_cache_counts_are_finite_non_negative_and_never_crash(invalid):
    result = response(aggregate={"usage": {"input_tokens": 100, "output_tokens": 10,
                                          "cache_read_input_tokens": invalid, "cache_creation_input_tokens": invalid}})
    counts = _extract_usage_details(result)
    assert counts == (100, 10, 0, 0)
    json.dumps(counts, allow_nan=False)


def test_missing_cache_usage_does_not_invent_a_hit():
    assert _extract_usage_details(response(aggregate={"usage": {"input_tokens": 100, "output_tokens": 10}})) == (100, 10, 0, 0)


def test_repeated_reference_to_same_message_does_not_duplicate_usage():
    message = SimpleNamespace(usage_metadata={"input_tokens": 50, "output_tokens": 4, "input_token_details": {"cache_read": 30}}, response_metadata={})
    generation = SimpleNamespace(message=message)
    result = SimpleNamespace(llm_output={}, generations=[[generation, generation]])
    assert _extract_usage_details(result) == (50, 4, 30, 0)


def test_cache_metrics_are_recorded_without_changing_input_budgets_or_price_formula(monkeypatch):
    events = []
    monkeypatch.setattr("app.usage.callbacks.report_usage_event", events.append)
    monkeypatch.setattr(settings, "ai_usage_input_price_cny_per_million", 10)
    monkeypatch.setattr(settings, "ai_usage_output_price_cny_per_million", 20)
    context = ModelCallContext(user_id="u", conversation_id="c", turn_id="t", trace_id="trace", reservation_id=None, route_version="test")
    set_model_call_context(context)
    try:
        handler = UsageCallbackHandler("chat", "provider", "model")
        run_id = uuid.uuid4()
        handler.on_llm_start({}, ["private prompt"], run_id=run_id)
        handler.on_llm_end(response(aggregate={"usage": {"input_tokens": 100, "output_tokens": 10,
                                                         "cache_read_input_tokens": 80, "cache_creation_input_tokens": 5}}), run_id=run_id)
        assert events[0]["cacheReadTokens"] == 80 and events[0]["cacheWriteTokens"] == 5
        assert events[0]["inputTokens"] == context.input_tokens == 100
        assert events[0]["outputTokens"] == context.output_tokens == 10
        assert events[0]["estimatedCostMicros"] == context.estimated_cost_micros == 1200
        assert events[0]["usageSource"] == "PROVIDER_REPORTED"
        assert "private prompt" not in str(events)
        assert context.pending_calls == context.pending_input_tokens == context.pending_cost_micros == 0
    finally:
        set_model_call_context(None)
