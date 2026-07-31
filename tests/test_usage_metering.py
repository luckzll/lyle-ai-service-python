from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import uuid

import pytest

from app.usage.callbacks import UsageBudgetExceeded, UsageCallbackHandler
from app.usage.context import ModelCallContext, set_model_call_context, usage_totals
from app.usage.manual import begin_external_call, finish_external_call
from app.usage import reporter


@pytest.fixture(autouse=True)
def reset_usage_context():
    set_model_call_context(None)
    yield
    set_model_call_context(None)


def test_provider_usage_is_recorded_without_prompt_content(monkeypatch):
    delivered: list[dict] = []
    monkeypatch.setattr("app.usage.callbacks.report_usage_event", delivered.append)
    context = ModelCallContext(
        user_id="42",
        conversation_id="100",
        turn_id="turn-1",
        trace_id="trace-1",
        reservation_id="reservation-1",
        route_version="v6.4-1",
    )
    set_model_call_context(context)
    handler = UsageCallbackHandler("chat", "provider-a", "model-a")
    run_id = uuid.uuid4()

    handler.on_llm_start({}, ["private prompt text"], run_id=run_id)
    handler.on_llm_end(
        SimpleNamespace(
            llm_output={"token_usage": {"prompt_tokens": 120, "completion_tokens": 35}},
            generations=[],
        ),
        run_id=run_id,
    )

    assert usage_totals(context)["prompt_tokens"] == 120
    assert usage_totals(context)["completion_tokens"] == 35
    assert usage_totals(context)["measurement_source"] == "PROVIDER_REPORTED"
    assert delivered[0]["reservationId"] == "reservation-1"
    assert delivered[0]["usageSource"] == "PROVIDER_REPORTED"
    assert "private prompt text" not in str(delivered[0])


def test_estimated_usage_is_explicit_and_cumulative_budget_is_enforced(monkeypatch):
    monkeypatch.setattr("app.usage.callbacks.report_usage_event", lambda event: None)
    context = ModelCallContext(
        user_id="42",
        conversation_id=None,
        turn_id="turn-2",
        trace_id="trace-2",
        reservation_id=None,
        route_version="v6.4-1",
        max_calls=2,
        max_input_tokens=5,
    )
    set_model_call_context(context)
    handler = UsageCallbackHandler("summary", "provider-a", "model-a")
    first_run = uuid.uuid4()

    handler.on_llm_start({}, ["四个字"], run_id=first_run)
    handler.on_llm_end(
        SimpleNamespace(
            llm_output={},
            generations=[[SimpleNamespace(text="回答", message=None)]],
        ),
        run_id=first_run,
    )

    assert usage_totals(context)["measurement_source"] == "ESTIMATED"
    with pytest.raises(UsageBudgetExceeded):
        handler.on_llm_start({}, ["再来四个"], run_id=uuid.uuid4())


def test_durable_outbox_defers_network_delivery(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(reporter.settings, "ai_usage_reporting_enabled", True)
    monkeypatch.setattr(reporter, "_persist", lambda event: True)
    monkeypatch.setattr(reporter, "_deliver", lambda events: calls.append("deliver") or True)

    reporter.report_usage_event({"eventId": "event-1"})

    assert calls == []


def test_development_without_outbox_delivers_synchronously(monkeypatch):
    calls: list[list[dict]] = []
    monkeypatch.setattr(reporter.settings, "ai_usage_reporting_enabled", True)
    monkeypatch.setattr(reporter, "_persist", lambda event: False)
    monkeypatch.setattr(reporter, "_deliver", lambda events: calls.append(events) or True)

    reporter.report_usage_event({"eventId": "event-2"})

    assert calls == [[{"eventId": "event-2"}]]


def test_parallel_external_calls_reserve_budget_before_delivery(monkeypatch):
    monkeypatch.setattr("app.usage.manual.report_usage_event", lambda event: None)
    context = ModelCallContext(
        user_id="42",
        conversation_id=None,
        turn_id="turn-3",
        trace_id="trace-3",
        reservation_id="reservation-3",
        route_version="v6.4-1",
        max_calls=1,
        max_input_tokens=10,
    )
    set_model_call_context(context)

    ticket = begin_external_call(estimated_input_tokens=5)
    with pytest.raises(UsageBudgetExceeded):
        begin_external_call(estimated_input_tokens=1)

    finish_external_call(
        ticket,
        call_type="IMAGE_GENERATION",
        task_type="IMAGE_GENERATE",
        provider="provider-a",
        model="model-a",
        input_tokens=5,
        estimated_cost_micros=10,
    )
    assert usage_totals(context)["call_count"] == 1
    assert context.pending_calls == 0


def test_parallel_llm_callbacks_reserve_budget_before_provider_call(monkeypatch):
    monkeypatch.setattr("app.usage.callbacks.report_usage_event", lambda event: None)
    context = ModelCallContext(
        user_id="42",
        conversation_id=None,
        turn_id="turn-llm-parallel",
        trace_id="trace-llm-parallel",
        reservation_id="reservation-llm-parallel",
        route_version="v6.4-1",
        max_calls=1,
        max_input_tokens=10,
    )
    set_model_call_context(context)
    handler = UsageCallbackHandler("chat", "provider-a", "model-a")
    first_run = uuid.uuid4()

    handler.on_llm_start({}, ["first"], run_id=first_run)

    def start_second_call() -> str:
        set_model_call_context(context)
        with pytest.raises(UsageBudgetExceeded):
            handler.on_llm_start({}, ["second"], run_id=uuid.uuid4())
        return "blocked"

    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(start_second_call).result() == "blocked"

    assert context.pending_calls == 1
    assert context.pending_input_tokens > 0
    handler.on_llm_end(
        SimpleNamespace(
            llm_output={"token_usage": {"prompt_tokens": 2, "completion_tokens": 1}},
            generations=[],
        ),
        run_id=first_run,
    )
    assert context.pending_calls == 0
    assert context.pending_input_tokens == 0
    assert context.pending_cost_micros == 0
    assert usage_totals(context)["call_count"] == 1


def test_outbox_partial_rejection_settles_each_event_once(monkeypatch):
    monkeypatch.setattr(reporter.settings, "postgres_enabled", True)
    monkeypatch.setattr(reporter.settings, "ai_usage_reporting_enabled", True)
    monkeypatch.setattr(
        reporter,
        "postgres_connection",
        _fake_outbox_connection(
            [
                ("accepted", {"eventId": "accepted"}),
                ("duplicate", {"eventId": "duplicate"}),
                ("rejected", {"eventId": "rejected"}),
                ("omitted", {"eventId": "omitted"}),
            ]
        ),
    )
    monkeypatch.setattr(
        reporter,
        "_deliver",
        lambda events: reporter.DeliveryOutcome(
            delivered_ids=frozenset({"accepted", "duplicate"}),
            rejected={"rejected": "INVALID_EVENT_ID"},
        ),
    )
    delivered: list[list[str]] = []
    terminal: list[dict[str, str]] = []
    retried: list[list[str]] = []
    monkeypatch.setattr(reporter, "_mark_delivered", delivered.append)
    monkeypatch.setattr(reporter, "_mark_terminal_rejected", terminal.append)
    monkeypatch.setattr(reporter, "_mark_retry", retried.append)

    assert reporter.deliver_pending_events() == 2
    assert delivered == [["accepted", "duplicate"]]
    assert terminal == [{"rejected": "INVALID_EVENT_ID"}]
    assert retried == [["omitted"]]


class _FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def execute(self, query, params):
        return None

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class _FakeConnection:
    def __init__(self, rows):
        self.rows = rows

    def cursor(self):
        return _FakeCursor(self.rows)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def _fake_outbox_connection(rows):
    return lambda: _FakeConnection(rows)
