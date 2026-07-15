from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.automation.schedule import next_run_at
from app.automation.automation_store import _summary_row
from app.experience.draft_service import _schedule_from_text, _validate_schedule
from app.experience.connection_diagnostics import recommended_connections
from app.experience.template_catalog import get_template, list_templates, workflow_payload
from app.api.experience import ProductEventRequest, product_event
from app.security.secret_masker import mask_secrets


def test_template_catalog_contains_five_result_templates_and_only_read_only_steps():
    templates = list_templates()
    assert len(templates) == 5
    assert all(item["writes"] is False for item in templates)
    workflow = workflow_payload(get_template("weekly_project_review"), title="周报", project_id=1)
    assert workflow["steps"]
    assert {step["stepType"] for step in workflow["steps"]} == {"summarize"}
    assert all(step["requiresConfirmation"] is False for step in workflow["steps"])
    assert all(step["inputTemplate"]["readOnly"] is True for step in workflow["steps"])


def test_natural_language_schedule_uses_product_facing_weekly_format():
    assert _schedule_from_text("每周五下午五点生成周报", "weekly:fri:17:00") == "weekly:fri:17:00"
    assert _schedule_from_text("每天 09:30 生成简报", "daily:09:00") == "daily:09:30"


def test_schedule_formats_are_supported_by_automation_scheduler():
    now = datetime(2026, 7, 10, 8, 0, tzinfo=UTC)
    assert next_run_at("daily:17:00", "Asia/Shanghai", now) > now
    assert next_run_at("weekly:fri:17:00", "Asia/Shanghai", now) > now
    with pytest.raises(Exception):
        _validate_schedule("*/5 * * * *")


def test_result_summary_masks_sensitive_values():
    masked = mask_secrets("token=abc123 password:letmein api_key=xyz")
    assert "abc123" not in masked
    assert "letmein" not in masked
    assert "xyz" not in masked


def test_recommended_connection_placeholders_are_navigation_actions(monkeypatch):
    monkeypatch.setattr("app.experience.connection_diagnostics.list_workspaces", lambda user_id: [])
    monkeypatch.setattr("app.experience.connection_diagnostics.list_mcp_servers", lambda user_id: [])

    items = recommended_connections("42")["items"]

    local_project = next(item for item in items if item["id"] == "local-project")
    github = next(item for item in items if item["id"] == "github")
    assert local_project["diagnosable"] is False
    assert local_project["actionTarget"] == "/advanced-settings?tab=cli"
    assert github["diagnosable"] is False
    assert github["actionTarget"] == "/advanced-settings?tab=mcp"


def test_v42_product_events_accept_only_allowlisted_metadata(monkeypatch):
    captured = {}

    def capture(user_id, event_type, **kwargs):
        captured.update(user_id=user_id, event_type=event_type, **kwargs)

    monkeypatch.setattr("app.api.experience.record_event", capture)
    result = product_event("42", ProductEventRequest(**{
        "eventType": "quick_intent_submitted",
        "page": "dashboard",
        "actionType": "inspect",
        "projectId": 7,
        "text": "must never be recorded",
    }))

    assert result == {"accepted": True}
    assert captured["user_id"] == "42"
    assert captured["event_type"] == "quick_intent_submitted"
    assert captured["metadata"]["page"] == "dashboard"
    assert captured["metadata"]["actionType"] == "inspect"
    assert captured["metadata"]["projectId"] == 7
    assert "text" not in captured["metadata"]
    with pytest.raises(ValidationError):
        ProductEventRequest(eventType="chat_body_uploaded")


@pytest.mark.parametrize(("field", "value"), [
    ("page", "sk-live-secret"),
    ("actionType", "password=secret"),
    ("templateId", "not-a-catalog-template"),
    ("status", "token-secret"),
    ("errorCategory", "private chat body"),
])
def test_v42_product_event_rejects_values_that_could_smuggle_sensitive_text(field, value):
    payload = {"eventType": "dashboard_viewed", field: value}
    with pytest.raises(ValidationError):
        ProductEventRequest(**payload)


def test_v42_product_event_enforces_database_bounds():
    with pytest.raises(ValidationError):
        ProductEventRequest(eventType="dashboard_viewed", templateId="x" * 65)
    with pytest.raises(ValidationError):
        ProductEventRequest(eventType="dashboard_viewed", durationMs=9_223_372_036_854_775_808)


@pytest.mark.parametrize("payload", [
    {"eventType": "dashboard_viewed", "page": "dashboard"},
    {"eventType": "quick_intent_submitted", "page": "dashboard", "actionType": "freeform"},
    {"eventType": "quick_intent_submitted", "page": "dashboard", "actionType": "organize-project"},
    {"eventType": "dashboard_action_clicked", "page": "dashboard", "actionType": "start_chat"},
    {"eventType": "context_panel_opened", "page": "chat", "projectId": 7},
    {"eventType": "automation_card_actioned", "page": "automations", "actionType": "run", "status": "enabled"},
    {"eventType": "execution_filter_applied", "page": "execution-center", "actionType": "view_results"},
    {"eventType": "execution_filter_applied", "page": "execution-center", "actionType": "filter_change"},
    {"eventType": "result_source_opened", "page": "execution-center", "actionType": "view", "status": "completed"},
    {"eventType": "confirmation_completed", "page": "execution-center", "actionType": "confirm", "status": "confirmed"},
])
def test_v42_current_frontend_product_events_match_the_strict_contract(payload):
    assert ProductEventRequest(**payload).eventType == payload["eventType"]


def test_v42_automation_summary_includes_project_and_latest_result():
    now = datetime(2026, 7, 11, tzinfo=UTC)
    row = (
        17, 8, "每日巡检", "schedule", "daily:09:00", "Asia/Shanghai", "enabled",
        now, now, {"projectId": 7}, 2, 60, 1, True, None, None, None, None, {}, now, now,
        7, "V4.2", "failed", 301, "巡检需要处理", "failed", True,
    )

    summary = _summary_row(row)

    assert summary["projectId"] == 7
    assert summary["projectName"] == "V4.2"
    assert summary["lastRunStatus"] == "failed"
    assert summary["lastResultId"] == 301
    assert summary["lastResultTitle"] == "巡检需要处理"
    assert summary["needsAttention"] is True
