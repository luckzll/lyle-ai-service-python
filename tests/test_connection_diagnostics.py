from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from app.experience import connection_diagnostics as diagnostics


@pytest.fixture(autouse=True)
def isolated_connections(monkeypatch):
    monkeypatch.setattr(diagnostics.settings, "postgres_enabled", True)
    monkeypatch.setattr(diagnostics, "list_workspaces", lambda user_id: [])
    monkeypatch.setattr(diagnostics, "list_mcp_servers", lambda user_id: [])
    monkeypatch.setattr(diagnostics, "record_event", lambda *args, **kwargs: None)


def server(**overrides):
    return {
        "id": 7, "name": "本地 MCP", "status": "enabled", "serverType": "builtin",
        "lastHealthStatus": None, "lastHealthMessage": None, "lastCheckedAt": None,
        **overrides,
    }


def test_unused_connections_are_optional_setup_not_attention():
    items = diagnostics.recommended_connections("42")["items"]
    assert {item["type"] for item in items} == {"local_project", "github"}
    assert all(item["status"] == "not_configured" for item in items)
    assert all(item["configured"] is False and item["needsAttention"] is False for item in items)
    assert all(item["diagnosable"] is False for item in items)


def test_configured_builtin_github_is_not_reported_as_missing(monkeypatch):
    monkeypatch.setattr(diagnostics.settings, "connector_github_token", "private-token")
    item = next(item for item in diagnostics.recommended_connections("42")["items"] if item["type"] == "github")
    assert item["status"] == "unchecked" and item["configured"] is True
    assert item["needsAttention"] is False and item["diagnosable"] is False
    assert item["actionTarget"] == "/advanced-settings?tab=connectors"
    assert "private-token" not in json.dumps(item)


@pytest.mark.parametrize(("values", "status", "attention"), [
    ({}, "unchecked", False),
    ({"status": "disabled", "lastHealthStatus": "error", "lastHealthMessage": "401 unauthorized"}, "disabled", False),
    ({"lastHealthStatus": "ok", "lastHealthMessage": "authentication succeeded"}, "healthy", False),
    ({"lastHealthStatus": "error", "lastHealthMessage": "connection timed out"}, "unavailable", True),
    ({"lastHealthStatus": "error", "lastHealthMessage": "401 unauthorized"}, "reauthorize_required", True),
    ({"lastHealthStatus": "error", "lastHealthMessage": "403 forbidden"}, "restricted", True),
])
def test_only_enabled_connections_with_real_failed_checks_need_attention(monkeypatch, values, status, attention):
    monkeypatch.setattr(diagnostics, "list_mcp_servers", lambda user_id: [server(**values)])
    item = diagnostics.connection_diagnostics("42")["items"][0]
    assert item["status"] == status
    assert item["needsAttention"] is attention
    assert item["configured"] is True
    assert item["diagnosable"] is (values.get("status") != "disabled")


def test_snapshot_preserves_check_date_without_echoing_private_error_details(monkeypatch):
    private = "https://private.example/mcp?token=secret Authorization: Bearer secret"
    monkeypatch.setattr(diagnostics, "list_mcp_servers", lambda user_id: [server(
        lastHealthStatus="error", lastHealthMessage=private, lastCheckedAt="2026-10-16T08:00:00Z",
    )])
    item = diagnostics.connection_diagnostics("42")["items"][0]
    assert item["lastCheckedAt"] == "2026-10-16T08:00:00Z"
    assert item["status"] == "unavailable"
    assert "private.example" not in json.dumps(item)
    assert "secret" not in json.dumps(item)


def test_workspace_authorization_does_not_pretend_to_check_the_local_host(monkeypatch):
    monkeypatch.setattr(diagnostics, "list_workspaces", lambda user_id: [
        {"id": 1, "name": "已授权", "status": "active", "updatedAt": "2026-10-16T08:00:00Z", "rootPath": "/private"},
        {"id": 2, "name": "已停用", "status": "disabled"},
    ])
    items = diagnostics.connection_diagnostics("42")["items"]
    assert [item["status"] for item in items] == ["unchecked", "disabled"]
    assert all(item["lastCheckedAt"] is None and item["diagnosable"] is False for item in items)
    assert all(item["needsAttention"] is False for item in items)
    assert "/private" not in json.dumps(items)


def test_diagnosis_runs_real_check_once_with_user_scoped_config_and_persists_health(monkeypatch):
    calls = []
    config = server(configRef=None)

    def get_config(user_id, server_id):
        calls.append(("lookup", user_id, server_id))
        return config

    def check(value):
        assert value is config
        calls.append(("check",))
        return {"ok": True, "message": "internal endpoint must not escape"}

    def save(user_id, server_id, status, message):
        calls.append(("save", user_id, server_id, status))
        assert "internal endpoint" not in message
        return server(lastHealthStatus=status, lastHealthMessage=message, lastCheckedAt="2026-10-16T09:00:00Z")

    monkeypatch.setattr(diagnostics, "get_mcp_server_config", get_config)
    monkeypatch.setattr(diagnostics, "run_mcp_health_check", check)
    monkeypatch.setattr(diagnostics, "mark_health", save)
    item = diagnostics.diagnose_connection("42", "mcp-7")
    assert calls == [("lookup", "42", 7), ("check",), ("save", "42", 7, "ok")]
    assert item["status"] == "healthy" and item["needsAttention"] is False
    assert item["lastCheckedAt"] == "2026-10-16T09:00:00Z"


@pytest.mark.parametrize(("failure", "status", "saved_status"), [
    ({"ok": False, "message": "401 Unauthorized https://private.example?token=secret"}, "reauthorize_required", "auth_required"),
    ({"ok": False, "message": "403 Forbidden https://private.example?token=secret"}, "restricted", "restricted"),
    (HTTPException(503, "连接凭据当前不可用。token=secret"), "reauthorize_required", "auth_required"),
    (RuntimeError("https://private.example?token=secret"), "unavailable", "error"),
])
def test_diagnosis_persists_safe_failure_categories(monkeypatch, failure, status, saved_status):
    monkeypatch.setattr(diagnostics, "get_mcp_server_config", lambda user_id, server_id: server())

    def check(value):
        if isinstance(failure, Exception):
            raise failure
        return failure

    def save(user_id, server_id, health, message):
        assert (user_id, server_id, health) == ("42", 7, saved_status)
        assert "secret" not in message and "private.example" not in message
        return server(lastHealthStatus=health, lastHealthMessage=message, lastCheckedAt="2026-10-16T09:00:00Z")

    monkeypatch.setattr(diagnostics, "run_mcp_health_check", check)
    monkeypatch.setattr(diagnostics, "mark_health", save)
    item = diagnostics.diagnose_connection("42", "mcp-7")
    assert item["status"] == status and item["needsAttention"] is True
    assert "secret" not in json.dumps(item) and "private.example" not in json.dumps(item)


@pytest.mark.parametrize(("connection_id", "lookup", "expected"), [
    ("mcp-7", "missing", 404),
    ("mcp-7", "disabled", 409),
    ("workspace-7", "workspace", 409),
    ("github", "none", 404),
    ("mcp-7/other", "none", 404),
])
def test_unowned_disabled_and_nondiagnosable_connections_never_execute(monkeypatch, connection_id, lookup, expected):
    calls = []

    def get_config(user_id, server_id):
        calls.append(("mcp", user_id, server_id))
        if lookup == "missing":
            raise HTTPException(404, "MCP Server 不存在。")
        return server(status="disabled")

    def workspace(user_id, workspace_id):
        calls.append(("workspace", user_id, workspace_id))
        return {"id": workspace_id}

    def forbidden(*args, **kwargs):
        pytest.fail("Unauthorized or non-diagnosable connections must not execute or save health")

    monkeypatch.setattr(diagnostics, "get_mcp_server_config", get_config)
    monkeypatch.setattr(diagnostics, "get_workspace", workspace)
    monkeypatch.setattr(diagnostics, "run_mcp_health_check", forbidden)
    monkeypatch.setattr(diagnostics, "mark_health", forbidden)
    with pytest.raises(HTTPException) as exc:
        diagnostics.diagnose_connection("current-user", connection_id)
    assert exc.value.status_code == expected
    assert calls == ([("mcp", "current-user", 7)] if lookup in {"missing", "disabled"}
                     else [("workspace", "current-user", 7)] if lookup == "workspace" else [])
