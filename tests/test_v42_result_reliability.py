from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import HTTPException

from app.automation import automation_runner, automation_store
from app.experience import result_summary_service
from app.security.secret_masker import mask_secrets


class _RowsCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _RowsConnection:
    def __init__(self, rows):
        self.rows = rows
        self.queries = []

    def execute(self, query, params):
        self.queries.append((query, params))
        if "FROM workflow_run run" in query:
            return _RowsCursor([])
        return _RowsCursor(self.rows)

    def commit(self):
        return None


class _SummaryConnection(_RowsConnection):
    def execute(self, query, params):
        self.queries.append((query, params))
        if "SELECT id FROM execution_result_summary" in query:
            return _RowsCursor([])
        return _RowsCursor([(301,)])


def _connection_factory(connection):
    @contextmanager
    def connect():
        yield connection

    return connect


def test_failure_messages_mask_headers_bearer_tokens_and_url_credentials():
    raw = (
        "Authorization: Bearer top-secret-token "
        "fallback Bearer abcdefghijklmnop "
        "key sk-abcdef1234567890 "
        "dsn postgresql://alice:super-secret@db.internal/app"
    )

    masked = mask_secrets(raw)

    assert "top-secret-token" not in masked
    assert "abcdefghijklmnop" not in masked
    assert "sk-abcdef" not in masked
    assert "alice" not in masked
    assert "super-secret" not in masked
    assert "Authorization=***" in masked
    assert "postgresql://***:***@" in masked


def test_failure_summary_uses_workflow_project_as_authority(monkeypatch):
    monkeypatch.setattr(
        automation_runner,
        "get_workflow",
        lambda user_id, workflow_id: {"id": workflow_id, "projectId": 73},
    )
    automation = {"workflowId": 9, "input": {"projectId": 999}}

    assert automation_runner._workflow_project_id("42", automation) == 73


def test_automation_failure_summary_persists_only_stable_user_copy(monkeypatch):
    connection = _SummaryConnection([])
    monkeypatch.setattr(
        result_summary_service,
        "postgres_connection",
        _connection_factory(connection),
    )
    monkeypatch.setattr(
        result_summary_service,
        "get_result",
        lambda *_args, **_kwargs: {"id": 301},
    )

    result_summary_service.ensure_automation_failure_summary(
        "42",
        12,
        automation_name="巡检",
        project_id=9,
    )

    insert_params = connection.queries[0][1]
    persisted = " ".join(str(value) for value in insert_params)
    assert "自动化运行失败，请检查连接、权限或输入配置后重试。" in persisted
    assert "Authorization" not in persisted
    assert "error_message" not in connection.queries[0][0]


def test_workflow_failure_summary_does_not_persist_raw_step_errors(monkeypatch):
    connection = _SummaryConnection([])
    monkeypatch.setattr(
        result_summary_service,
        "postgres_connection",
        _connection_factory(connection),
    )
    monkeypatch.setattr(
        result_summary_service,
        "get_workflow_run",
        lambda *_args: {
            "status": "failed",
            "projectId": 7,
            "input": {},
            "errorMessage": "Authorization: Bearer raw-workflow-secret",
            "startedAt": None,
            "completedAt": None,
            "steps": [{
                "title": "连接检查",
                "status": "failed",
                "errorMessage": "postgresql://alice:password@db/app",
            }],
        },
    )
    monkeypatch.setattr(
        result_summary_service,
        "get_result",
        lambda *_args, **_kwargs: {"id": 301},
    )

    result_summary_service.ensure_workflow_summary("42", 101)

    insert_query, insert_params = connection.queries[0]
    persisted = " ".join(str(value) for value in insert_params)
    assert "执行未完成，请检查连接、权限或输入配置后重试。" in persisted
    assert "连接检查尚未完成，请查看技术详情。" in persisted
    assert "raw-workflow-secret" not in persisted
    assert "alice" not in persisted
    assert "password@" not in persisted
    assert "error_message" not in insert_query


def test_paused_workflow_summary_uses_confirmation_action_instead_of_retry(monkeypatch):
    connection = _SummaryConnection([])
    monkeypatch.setattr(
        result_summary_service,
        "postgres_connection",
        _connection_factory(connection),
    )
    monkeypatch.setattr(
        result_summary_service,
        "get_workflow_run",
        lambda *_args: {
            "status": "paused",
            "projectId": 7,
            "input": {},
            "startedAt": None,
            "completedAt": None,
            "steps": [{"title": "确认验收口径", "status": "awaiting_confirmation"}],
        },
    )
    monkeypatch.setattr(
        result_summary_service,
        "get_result",
        lambda *_args, **_kwargs: {"id": 301},
    )

    result_summary_service.ensure_workflow_summary("42", 93)

    persisted = " ".join(str(value) for value in connection.queries[0][1])
    assert "resume_workflow" in persisted
    assert "确认并继续" in persisted
    assert "重试本次运行" not in persisted


def test_automation_summary_backfill_covers_workflow_and_no_workflow_failures(monkeypatch):
    rows = [
        (11, 101, "completed", "每日汇总", 7),
        (12, None, "failed", "巡检", 9),
    ]
    connection = _RowsConnection(rows)
    monkeypatch.setattr(
        automation_store,
        "postgres_connection",
        _connection_factory(connection),
    )
    workflow_calls = []
    failure_calls = []
    monkeypatch.setattr(
        result_summary_service,
        "ensure_workflow_summary",
        lambda user_id, run_id: workflow_calls.append((user_id, run_id)),
    )
    monkeypatch.setattr(
        result_summary_service,
        "ensure_automation_failure_summary",
        lambda user_id, run_id, **kwargs: failure_calls.append((user_id, run_id, kwargs)),
    )

    automation_store._ensure_latest_result_summaries("42")

    assert workflow_calls == [("42", 101)]
    assert failure_calls == [(
        "42",
        12,
        {
            "automation_name": "巡检",
            "project_id": 9,
        },
    )]
    query = connection.queries[0][0]
    assert "workflow_run_id IS NOT NULL OR status = 'failed'" in query
    assert "NOT EXISTS" in query


def test_result_list_batches_automation_association_without_per_item_detail_queries(monkeypatch):
    now = datetime(2026, 7, 11, tzinfo=UTC)
    row = (
        301, "workflow_run", "101", 7, None, "failed", "巡检需要处理",
        "运行失败。", [], ["运行失败。"], [{"type": "retry", "label": "重试"}],
        ["所选项目"], 1200, "workflow_run:101", 17, "每日巡检", now,
    )
    connection = _RowsConnection([row])
    monkeypatch.setattr(result_summary_service.settings, "postgres_enabled", True)
    monkeypatch.setattr(
        result_summary_service,
        "postgres_connection",
        _connection_factory(connection),
    )
    monkeypatch.setattr(
        result_summary_service,
        "get_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("N+1 detail lookup")),
    )

    result = result_summary_service.list_results("42", limit=10)

    assert result["items"][0]["id"] == 301
    assert result["items"][0]["automationId"] == 17
    detail_queries = [query for query, _ in connection.queries if "FROM execution_result_summary result" in query]
    assert len(detail_queries) == 1
    assert "run.workflow_run_id = CASE" in detail_queries[0]
    assert "run.workflow_run_id::text" not in detail_queries[0]


def test_result_list_uses_keyset_cursor_and_server_side_automation_filter(monkeypatch):
    now = datetime(2026, 7, 11, 12, 0, tzinfo=UTC)

    def result_row(result_id, created_at):
        return (
            result_id, "workflow_run", str(result_id), 7, None, "completed", "已完成",
            "检查完成。", [], [], [], ["所选项目"], 100, f"workflow_run:{result_id}",
            17, "每日巡检", created_at,
        )

    rows = [
        result_row(303, now - timedelta(minutes=1)),
        result_row(302, now - timedelta(minutes=2)),
        result_row(301, now - timedelta(minutes=3)),
    ]
    connection = _RowsConnection(rows)
    monkeypatch.setattr(result_summary_service.settings, "postgres_enabled", True)
    monkeypatch.setattr(
        result_summary_service,
        "postgres_connection",
        _connection_factory(connection),
    )
    initial_cursor = result_summary_service._encode_cursor(now, 400)

    result = result_summary_service.list_results(
        "42",
        limit=2,
        cursor=initial_cursor,
        automation_id=17,
    )

    assert [item["id"] for item in result["items"]] == [303, 302]
    assert result["hasMore"] is True
    assert result_summary_service._decode_cursor(result["nextCursor"]) == (
        now - timedelta(minutes=2),
        302,
    )
    result_query, params = connection.queries[-1]
    assert "automation_run.automation_id=%s" in result_query
    assert "(result.created_at, result.id) < (%s, %s)" in result_query
    assert params == ["42", 17, now, 400, 3]


def test_result_cursor_rejects_malformed_or_naive_time_values():
    with pytest.raises(HTTPException) as error:
        result_summary_service._decode_cursor("not-a-valid-cursor")
    assert error.value.status_code == 400

    naive = result_summary_service._encode_cursor(datetime(2026, 7, 11), 10)
    with pytest.raises(HTTPException):
        result_summary_service._decode_cursor(naive)
