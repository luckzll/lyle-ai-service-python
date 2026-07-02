import re
from datetime import UTC, datetime, timedelta

from app.core.config import settings
from app.core.database import postgres_connection


def extract_candidate_tasks(question: str) -> list[dict[str, object]]:
    title = _extract_candidate_todo_title(question)
    if not title:
        return []
    due_at = _extract_datetime(question)
    return [{
        "type": "todo",
        "title": title,
        "dueAt": due_at.isoformat() if due_at else None,
        "source": question[:240],
        "confidence": 0.82,
        "status": "pending",
    }]


def create_todo(
    user_id: str,
    conversation_id: str | None,
    title: str,
    due_at: datetime | None = None,
) -> dict[str, object]:
    if not settings.postgres_enabled:
        return {"ok": False, "content": "PostgreSQL 未启用，无法创建待办。"}
    title = title.strip()[:200]
    if not title:
        return {"ok": False, "content": "没有识别到待办内容。"}

    now = datetime.now(UTC)
    with postgres_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO agent_todo (user_id, conversation_id, title, status, due_at, created_at, updated_at)
            VALUES (%s, %s, %s, 'open', %s, %s, %s)
            RETURNING id, created_at, updated_at
            """,
            (user_id, conversation_id or "", title, due_at, now, now),
        ).fetchone()
        conn.commit()

    due_text = due_at.isoformat() if due_at else None
    return {
        "ok": True,
        "content": f"已创建待办：{title}，时间：{due_text or '未设置时间'}",
        "todo": {
            "id": row[0],
            "title": title,
            "status": "open",
            "dueAt": due_text,
            "conversationId": conversation_id or "",
            "createdAt": row[1].isoformat(),
            "updatedAt": row[2].isoformat(),
        },
    }


def create_todo_from_question(user_id: str, conversation_id: str | None, question: str) -> dict[str, object]:
    title = _cleanup_title(question, ["帮我创建待办", "创建待办", "加个待办", "待办", "todo"])
    return create_todo(user_id, conversation_id, title, _extract_datetime(question))


def create_calendar_event_from_question(user_id: str, conversation_id: str | None, question: str) -> dict[str, object]:
    if not settings.postgres_enabled:
        return {"ok": False, "content": "PostgreSQL 未启用，无法创建日历事件。"}

    title = _cleanup_title(question, ["帮我创建日历", "创建日历", "加个日程", "日程", "日历", "calendar"])
    if not title:
        return {"ok": False, "content": "没有识别到日历事件内容。"}

    starts_at = _extract_datetime(question)
    with postgres_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO agent_calendar_event
              (user_id, conversation_id, title, starts_at, status, created_at, updated_at)
            VALUES (%s, %s, %s, %s, 'scheduled', %s, %s)
            RETURNING id
            """,
            (user_id, conversation_id or "", title, starts_at, datetime.now(UTC), datetime.now(UTC)),
        ).fetchone()
        conn.commit()

    time_text = starts_at.isoformat() if starts_at else "未设置具体时间"
    return {
        "ok": True,
        "content": f"已创建日历事件：{title}，时间：{time_text}",
        "event": {"id": row[0], "title": title, "startsAt": time_text, "status": "scheduled"},
    }


def _cleanup_title(question: str, prefixes: list[str]) -> str:
    title = question.strip()
    for prefix in prefixes:
        title = title.replace(prefix, "")
    title = re.sub(r"^(：|:|，|,|\s)+", "", title).strip()
    return title[:200]


def _extract_candidate_todo_title(question: str) -> str:
    if any(keyword in question for keyword in ["创建待办", "加个待办", "待办", "todo"]):
        return _cleanup_title(question, ["帮我创建待办", "创建待办", "加个待办", "待办", "todo"])

    patterns = [
        r"(?:这周|本周|今天|明天|后天|下周[一二三四五六日天]?|周[一二三四五六日天]?)?(?:我)?(?:要|想要|需要|打算|计划)(?:完成|做完|搞定|处理|推进|写完|整理|提交)([^，。,.!\n]{2,80})",
        r"(?:帮我)?(?:提醒我|记得提醒我)([^，。,.!\n]{2,80})",
    ]
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            return match.group(1).strip(" ：:，,。.")
    return ""


def _extract_datetime(question: str) -> datetime | None:
    return _extract_iso_datetime(question) or _extract_cn_datetime(question)


def _extract_iso_datetime(question: str) -> datetime | None:
    match = re.search(r"(\d{4}-\d{2}-\d{2})(?:[ T](\d{2}:\d{2})(?::\d{2})?)?", question)
    if not match:
        return None
    date_part = match.group(1)
    time_part = match.group(2) or "09:00"
    return datetime.fromisoformat(f"{date_part}T{time_part}:00").replace(tzinfo=UTC)


def _extract_cn_datetime(question: str, now: datetime | None = None) -> datetime | None:
    now = now or datetime.now(UTC)
    base_date = _extract_cn_date(question, now)
    if base_date is None:
        return None

    hour, minute = _extract_cn_time(question)
    return base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _extract_cn_date(question: str, now: datetime) -> datetime | None:
    if "今天" in question:
        return now
    if "明天" in question:
        return now + timedelta(days=1)
    if "后天" in question:
        return now + timedelta(days=2)

    weekdays = {
        "一": 0,
        "二": 1,
        "三": 2,
        "四": 3,
        "五": 4,
        "六": 5,
        "日": 6,
        "天": 6,
    }
    match = re.search(r"下周([一二三四五六日天])", question)
    if match:
        target = weekdays[match.group(1)]
        days_until_next_week = 7 - now.weekday()
        return now + timedelta(days=days_until_next_week + target)

    match = re.search(r"周([一二三四五六日天])", question)
    if match:
        target = weekdays[match.group(1)]
        delta = (target - now.weekday()) % 7
        if delta == 0:
            delta = 7
        return now + timedelta(days=delta)

    return None


def _extract_cn_time(question: str) -> tuple[int, int]:
    match = re.search(r"(\d{1,2})[:：](\d{2})", question)
    if match:
        return int(match.group(1)), int(match.group(2))

    match = re.search(r"([零一二两三四五六七八九十\d]{1,3})点(半)?", question)
    if match:
        hour = _cn_number_to_int(match.group(1))
        minute = 30 if match.group(2) else 0
    elif "早上" in question or "上午" in question:
        hour, minute = 9, 0
    elif "中午" in question:
        hour, minute = 12, 0
    elif "下午" in question:
        hour, minute = 15, 0
    elif "晚上" in question:
        hour, minute = 20, 0
    else:
        hour, minute = 9, 0

    if any(word in question for word in ["下午", "晚上"]) and 1 <= hour <= 11:
        hour += 12
    return hour, minute


def _cn_number_to_int(value: str) -> int:
    if value.isdigit():
        return int(value)
    digits = {
        "零": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if value == "十":
        return 10
    if value.startswith("十"):
        return 10 + digits.get(value[-1], 0)
    if "十" in value:
        left, right = value.split("十", 1)
        return digits.get(left, 1) * 10 + (digits.get(right, 0) if right else 0)
    return digits.get(value, 9)
