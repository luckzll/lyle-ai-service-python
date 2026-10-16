from contextlib import contextmanager
from datetime import UTC, datetime

import pytest
from fastapi import HTTPException

from app.briefings import briefing_builder as briefing
from app.experience import confirmation_inbox, result_summary_service as results


class Cursor:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


def connection_factory(connection):
    @contextmanager
    def connect():
        yield connection
    return connect


class FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 10, 15, 16, 30, tzinfo=UTC)


def test_shanghai_day_boundaries_cover_next_local_day_before_utc_midnight():
    now = datetime(2026, 10, 15, 16, 30, tzinfo=UTC)
    start, end = briefing._day_bounds(now)
    assert start == datetime(2026, 10, 15, 16, tzinfo=UTC)
    assert end == datetime(2026, 10, 16, 16, tzinfo=UTC)


def test_daily_briefing_counts_full_scope_independently_of_limited_lists(monkeypatch):
    now = FixedDateTime.now(UTC)

    class Connection:
        queries = []

        def execute(self, query, params):
            self.queries.append((query, params))
            if "COUNT(*) FILTER" in query:
                return Cursor([(37, 19)])
            if "COUNT(*)" in query:
                return Cursor([(62,)])
            if "FROM agent_pending_operation" in query:
                return Cursor([(i, "todo", "write", "low", f"确认{i}", now, now) for i in range(5)])
            if "FROM agent_todo" in query:
                due = datetime(2026, 10, 15, 17, tzinfo=UTC) if "due_at >=" in query else datetime(2026, 10, 15, 15, tzinfo=UTC)
                return Cursor([(i, f"任务{i}", "open", due, "c", now, now, 7) for i in range(5)])
            raise AssertionError(query)

    connection = Connection()
    monkeypatch.setattr(briefing.settings, "postgres_enabled", True)
    monkeypatch.setattr(briefing, "datetime", FixedDateTime)
    monkeypatch.setattr(briefing, "postgres_connection", connection_factory(connection))
    monkeypatch.setattr(briefing, "get_project", lambda user, project_id: {
        "id": project_id, "name": "发布计划", "status": "active", "currentStage": "验收",
    })
    goal_scopes = []
    monkeypatch.setattr(briefing, "list_goals", lambda user, **kwargs: goal_scopes.append(kwargs["project_id"]) or [])

    result = briefing.generate_daily_briefing("owner", project_id=7, persist=False)

    assert result["counts"] == {"dueToday": 37, "overdue": 19, "pending": 62}
    assert len(result["dueTasks"]) == len(result["overdueTasks"]) == len(result["pendingOperations"]) == 5
    assert "今日到期任务：37 个" in result["content"]
    assert goal_scopes == [7]
    assert all(item["projectId"] == 7 for item in result["dueTasks"])
    for query, params in connection.queries:
        assert "user_id = %s" in query
        assert "owner" in params
        if "FROM agent_todo" in query:
            assert "project_id = %s" in query and 7 in params
        else:
            assert "payload->>'projectId' = %s" in query and "7" in params
        if "COUNT(*)" in query:
            assert "LIMIT" not in query


def test_todo_today_flag_uses_shanghai_inclusive_start_exclusive_end(monkeypatch):
    now = FixedDateTime.now(UTC)
    due_dates = [datetime(2026, 10, 15, 15, 59, tzinfo=UTC),
                 datetime(2026, 10, 15, 16, tzinfo=UTC),
                 datetime(2026, 10, 16, 15, 59, tzinfo=UTC),
                 datetime(2026, 10, 16, 16, tzinfo=UTC)]

    class Connection:
        def execute(self, query, params):
            return Cursor([(i, "边界任务", "open", due, "c", now, now, None) for i, due in enumerate(due_dates)])

    monkeypatch.setattr(briefing, "postgres_connection", connection_factory(Connection()))
    items = briefing._list_todos("owner", 10, now=now)
    assert [item["isToday"] for item in items] == [False, True, True, False]
    assert [item["isOverdue"] for item in items] == [True, True, False, False]


def test_unscoped_homepage_does_not_assign_global_tasks_to_first_recent_project(monkeypatch):
    monkeypatch.setattr(briefing.settings, "postgres_enabled", True)
    monkeypatch.setattr(briefing, "_list_briefing_projects", lambda *_: [{"id": 99, "name": "另一个计划"}])
    scopes = []
    monkeypatch.setattr(briefing, "list_goals", lambda user, **kwargs: scopes.append(kwargs["project_id"]) or [])
    monkeypatch.setattr(briefing, "_list_pending_operations", lambda *args, **kwargs: [])
    monkeypatch.setattr(briefing, "_list_todos", lambda *args, **kwargs: [])
    monkeypatch.setattr(briefing, "_briefing_counts", lambda *args, **kwargs: {"dueToday": 0, "overdue": 0, "pending": 0})
    result = briefing.generate_daily_briefing("owner", persist=False)
    assert result["projectId"] is None
    assert scopes == [None]
    assert result["projects"][0]["id"] == 99


def test_recent_project_selection_orders_before_limiting_and_is_user_scoped(monkeypatch):
    now = FixedDateTime.now(UTC)
    captured = []

    class Connection:
        def execute(self, query, params):
            captured.append((query, params))
            return Cursor([(991, "最近活动计划", "recent", "", "active", "验收", 0, now, now, now)])

    monkeypatch.setattr(briefing, "postgres_connection", connection_factory(Connection()))
    project = briefing._list_briefing_projects("owner", None)[0]
    query, params = captured[0]
    assert "ORDER BY COALESCE(last_activity_at, updated_at) DESC, priority DESC" in query
    assert query.index("ORDER BY") < query.index("LIMIT 5")
    assert "user_id = %s" in query and params == ("owner",)
    assert project["id"] == 991 and "nextStep" not in project


def test_explicit_project_is_authorized_directly_not_lost_outside_first_page(monkeypatch):
    monkeypatch.setattr(briefing, "get_project", lambda user, project: {
        "id": project, "name": "分页外的计划", "status": "active",
    })
    assert briefing._list_briefing_projects("owner", 999)[0]["id"] == 999

    def unauthorized(user, project):
        raise HTTPException(status_code=404, detail="项目不存在")
    monkeypatch.setattr(briefing, "get_project", unauthorized)
    with pytest.raises(HTTPException) as error:
        briefing._list_briefing_projects("other-owner", 999)
    assert error.value.status_code == 404


def test_confirmation_total_matches_pending_scope_without_limit(monkeypatch):
    queries = []

    class Connection:
        def execute(self, query, params):
            queries.append((query, params))
            return Cursor([(230,)]) if "COUNT(*)" in query else Cursor([])

    monkeypatch.setattr(confirmation_inbox.settings, "postgres_enabled", True)
    monkeypatch.setattr(confirmation_inbox, "postgres_connection", connection_factory(Connection()))
    result = confirmation_inbox.list_confirmations("owner", limit=5)
    assert result == {"items": [], "total": 230, "hasMore": True}
    assert "user_id=%s AND status='pending'" in queries[1][0]
    assert "LIMIT" not in queries[1][0] and queries[1][1] == ("owner",)


def result_row(*, title="执行已完成", conclusion="已完成只读检查，可以查看建议并决定下一步。", output=None):
    return (301, "workflow_run", "101", 7, None, "completed", title, conclusion,
            ["导出月报", "核对文件"], [], [], ["已授权工作区"], 100, "workflow_run:101",
            17, "每月报表", None, FixedDateTime.now(UTC), "导出月度报表", output)


@pytest.mark.parametrize("status, groups", [(None, [("completed", 140), ("failed", 29), ("partial", 7)]),
                                           ("failed", [("failed", 29)])])
def test_status_counts_share_filters_but_ignore_cursor_and_page_limit(monkeypatch, status, groups):
    queries = []

    class Connection:
        def execute(self, query, params):
            queries.append((query, params))
            if "GROUP BY result.status" in query:
                return Cursor(groups)
            if "FROM workflow_run run" in query:
                return Cursor([])
            return Cursor([result_row()])

    monkeypatch.setattr(results.settings, "postgres_enabled", True)
    monkeypatch.setattr(results, "postgres_connection", connection_factory(Connection()))
    cursor = results._encode_cursor(FixedDateTime.now(UTC), 400)
    response = results.list_results("owner", status=status, project_id=7, automation_id=17,
                                    attention_only=True, cursor=cursor, limit=1)
    count_query, count_params = next(item for item in queries if "GROUP BY" in item[0])
    page_query, page_params = queries[-1]
    for clause in ("result.user_id=%s", "result.project_id=%s", "automation_run.automation_id=%s",
                   "result.dismissed_at IS NULL", "result.status IN ('failed', 'partial', 'awaiting_confirmation')"):
        assert clause in count_query and clause in page_query
    assert "(result.created_at, result.id) <" not in count_query
    assert "LIMIT %s" not in count_query
    assert count_params == ["owner", *([status] if status else []), 7, 17]
    assert len(page_params) == len(count_params) + 3
    assert len(response["items"]) == 1
    assert response["statusCounts"]["failed"] == 29
    assert response["statusCounts"]["partial"] == (7 if status is None else 0)


def test_legacy_summary_uses_source_name_and_conclusion_without_mutating_history():
    source = {"steps": [{"title": "生成摘要", "result": {"summary": "已导出 2 份月报，发现 1 份数据尚未齐全。"}}]}
    row = result_row(output=source)
    result = results._result_row(row)
    assert result["title"] == "导出月度报表 · 已完成"
    assert result["conclusion"] == "已导出 2 份月报，发现 1 份数据尚未齐全。"
    assert result["recordedTitle"] == row[6]
    assert result["recordedConclusion"] == row[7]
    assert row[6] == "执行已完成" and row[7].startswith("已完成只读检查")


def test_existing_specific_summary_is_preserved_and_missing_evidence_is_explicit():
    specific = results._result_row(result_row(title="客户清单归档完成", conclusion="已归档 8 条记录。"))
    assert specific["title"] == "客户清单归档完成" and specific["conclusion"] == "已归档 8 条记录。"
    assert "recordedTitle" not in specific
    legacy = list(result_row())
    legacy[8] = []
    empty = results._result_row(tuple(legacy))
    assert "历史摘要未保存具体结论" in empty["conclusion"]
    assert "只读检查" not in empty["conclusion"]


def test_new_workflow_summary_names_source_and_never_counts_skipped_steps_as_completed(monkeypatch):
    inserts = []

    class Connection:
        def execute(self, query, params):
            if query.startswith("SELECT name"):
                assert params == ("owner", 19)
                return Cursor([("月度报表导出",)])
            inserts.append(params)
            return Cursor([(301,)])
        def commit(self):
            pass

    monkeypatch.setattr(results, "postgres_connection", connection_factory(Connection()))
    monkeypatch.setattr(results, "get_result", lambda *args, **kwargs: {"id": 301})
    monkeypatch.setattr(results, "get_workflow_run", lambda *args: {
        "workflowId": 19, "status": "completed", "input": {},
        "steps": [{"title": "导出文件", "status": "completed"}, {"title": "通知客户", "status": "skipped"}],
    })
    results.ensure_workflow_summary("owner", 101)
    assert inserts[0][5] == "月度报表导出 · 已完成"
    assert "运行报告完成 1 个步骤：导出文件" in inserts[0][6]
    assert "通知客户" not in inserts[0][7]
    assert "只读" not in inserts[0][6]
