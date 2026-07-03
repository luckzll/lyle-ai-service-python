from datetime import UTC, datetime

from app.tools import task_tool
from app.tools.task_tool import _extract_cn_datetime, extract_candidate_tasks, preview_todo_from_question


def test_extract_cn_datetime_understands_tomorrow_afternoon():
    now = datetime(2026, 6, 30, 8, 0, tzinfo=UTC)

    result = _extract_cn_datetime("明天下午三点开会", now)

    assert result == datetime(2026, 7, 1, 7, 0, tzinfo=UTC)


def test_extract_cn_datetime_understands_next_monday_morning():
    now = datetime(2026, 6, 30, 8, 0, tzinfo=UTC)

    result = _extract_cn_datetime("下周一早上复盘", now)

    assert result == datetime(2026, 7, 6, 1, 0, tzinfo=UTC)


def test_extract_candidate_tasks_from_commitment():
    result = extract_candidate_tasks("这周我要完成产品定位文档")

    assert result == [{
        "type": "todo",
        "title": "产品定位文档",
        "dueAt": None,
        "source": "这周我要完成产品定位文档",
        "confidence": 0.82,
        "status": "pending",
    }]


def test_preview_todo_from_question_requires_confirmation(monkeypatch):
    captured = {}

    def fake_create_pending_operation(**kwargs):
        captured.update(kwargs)
        return {
            "ok": True,
            "requiresConfirmation": True,
            "operation": {"id": 1, "summary": kwargs["summary"], "status": "pending"},
        }

    monkeypatch.setattr(task_tool, "create_pending_operation", fake_create_pending_operation)

    result = preview_todo_from_question("u1", "c1", "帮我创建待办 明天下午三点提交周报")

    assert result["requiresConfirmation"] is True
    assert captured["tool_name"] == "todo"
    assert captured["operation_type"] == "todo.create"
    assert captured["payload"]["title"] == "提交周报"
    assert captured["payload"]["dueAt"].endswith("07:00:00+00:00")
