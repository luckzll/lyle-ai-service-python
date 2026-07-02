from datetime import UTC, datetime

from app.tools.task_tool import _extract_cn_datetime, extract_candidate_tasks


def test_extract_cn_datetime_understands_tomorrow_afternoon():
    now = datetime(2026, 6, 30, 8, 0, tzinfo=UTC)

    result = _extract_cn_datetime("明天下午三点开会", now)

    assert result == datetime(2026, 7, 1, 15, 0, tzinfo=UTC)


def test_extract_cn_datetime_understands_next_monday_morning():
    now = datetime(2026, 6, 30, 8, 0, tzinfo=UTC)

    result = _extract_cn_datetime("下周一早上复盘", now)

    assert result == datetime(2026, 7, 6, 9, 0, tzinfo=UTC)


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
