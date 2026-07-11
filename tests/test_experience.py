from datetime import UTC, datetime

import pytest

from app.automation.schedule import next_run_at
from app.experience.draft_service import _schedule_from_text, _validate_schedule
from app.experience.connection_diagnostics import recommended_connections
from app.experience.result_summary_service import _mask
from app.experience.template_catalog import get_template, list_templates, workflow_payload


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
    masked = _mask("token=abc123 password:letmein api_key=xyz")
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
