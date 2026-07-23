from contextlib import contextmanager

import pytest
from pydantic import ValidationError

from app.api.experience import ProductEventRequest, product_event
from app.experience import analytics as analytics_module
from app.experience.events import EFFECTIVE_PROGRESS_ACTIONS


@pytest.mark.parametrize(("page", "action_type", "project_id"), [
    ("tasks", "task_complete", None),
    ("tasks", "task_reschedule", None),
    ("execution-center", "confirmation", None),
    ("automations", "automation_enable", None),
    ("automations", "automation_run", None),
    ("execution-center", "result_action", None),
    ("chat", "project_chat", 7),
])
def test_effective_progress_event_action_contract(page, action_type, project_id):
    request = ProductEventRequest(
        eventType="effective_progress_action",
        page=page,
        actionType=action_type,
        projectId=project_id,
    )
    assert request.actionType in EFFECTIVE_PROGRESS_ACTIONS


def test_effective_progress_event_rejects_metric_poisoning_and_drops_text(monkeypatch):
    with pytest.raises(ValidationError):
        ProductEventRequest(
            eventType="effective_progress_action",
            page="tasks",
            actionType="open",
        )
    with pytest.raises(ValidationError):
        ProductEventRequest(
            eventType="effective_progress_action",
            page="dashboard",
            actionType="task_complete",
        )
    with pytest.raises(ValidationError):
        ProductEventRequest(
            eventType="effective_progress_action",
            page="chat",
            actionType="project_chat",
        )

    captured = {}
    monkeypatch.setattr(
        "app.api.experience.record_event",
        lambda user_id, event_type, **kwargs: captured.update(
            user_id=user_id,
            event_type=event_type,
            **kwargs,
        ),
    )
    request = ProductEventRequest(**{
        "eventType": "effective_progress_action",
        "page": "chat",
        "actionType": "project_chat",
        "projectId": 7,
        "text": "this body must never be recorded",
    })

    assert product_event("42", request) == {"accepted": True}
    assert captured["metadata"] == {
        "page": "chat",
        "actionType": "project_chat",
        "projectId": 7,
        "errorCategory": None,
        "itemCount": None,
    }


class _AnalyticsCursor:
    def __init__(self, *, rows=None, row=None):
        self.rows = rows or []
        self.row = row

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.row


class _AnalyticsConnection:
    def __init__(self):
        self.queries = []

    def execute(self, query, params=None):
        self.queries.append((query, params))
        if "effective_actions >= 2" in query:
            return _AnalyticsCursor(row=(3, 10, 27))
        if "SELECT event_type, COUNT(*)" in query:
            return _AnalyticsCursor(rows=[("effective_progress_action", 27)])
        if "SELECT template_id, COUNT(*)" in query:
            return _AnalyticsCursor(rows=[])
        if "FROM execution_result_summary" in query and "status IN" in query:
            return _AnalyticsCursor(rows=[])
        if "FROM execution_result_summary" in query:
            return _AnalyticsCursor(rows=[("completed", 4)])
        raise AssertionError(f"Unexpected analytics query: {query}")


def test_analytics_returns_aggregate_seven_day_north_star_without_user_ids(monkeypatch):
    connection = _AnalyticsConnection()

    @contextmanager
    def connect():
        yield connection

    monkeypatch.setattr(analytics_module.settings, "postgres_enabled", True)
    monkeypatch.setattr(analytics_module, "postgres_connection", connect)

    result = analytics_module.experience_analytics(days=30)

    assert result["periodDays"] == 30
    assert result["northStar"] == {
        "windowDays": 7,
        "twoPlusEffectiveProgressUsers": 3,
        "eventActiveUsers": 10,
        "shareOfEventActiveUsers": 0.3,
        "effectiveProgressActions": 27,
    }
    assert "userId" not in str(result)
    north_star_query = next(query for query, _ in connection.queries if "effective_actions >= 2" in query)
    assert "interval '7 days'" in north_star_query
    assert "GROUP BY anonymous_user_id" in north_star_query
