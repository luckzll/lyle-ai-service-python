import re
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from app.core.config import settings
from app.core.database import postgres_connection
from app.memory.profile_store import list_user_profiles
from app.tools.operation_confirmation import create_pending_operation

LOCAL_TIMEZONE = ZoneInfo("Asia/Shanghai")


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
        duplicate = _find_open_duplicate_todo(conn, user_id, title)
        if duplicate:
            duplicate["goalLinks"] = infer_task_goal_links(user_id, title)
            return {
                "ok": True,
                "content": f"已存在相同待办：{title}，本次没有重复创建。",
                "duplicate": True,
                "todo": duplicate,
            }
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
            "sourceRef": f"conversation:{conversation_id}" if conversation_id else "",
            "goalLinks": infer_task_goal_links(user_id, title),
            "createdAt": row[1].isoformat(),
            "updatedAt": row[2].isoformat(),
        },
    }


def create_todo_from_question(user_id: str, conversation_id: str | None, question: str) -> dict[str, object]:
    title = _cleanup_title(question, ["帮我创建待办", "创建待办", "加个待办", "待办", "todo"])
    return create_todo(user_id, conversation_id, title, _extract_datetime(question))


def preview_todo_from_question(user_id: str, conversation_id: str | None, question: str) -> dict[str, object]:
    title = _cleanup_title(question, ["帮我创建待办", "创建待办", "加个待办", "待办", "todo"])
    due_at = _extract_datetime(question)
    title = title.strip()[:200]
    if not title:
        return {"ok": False, "content": "没有识别到待办内容。"}
    due_text = due_at.isoformat() if due_at else None
    due_summary = _format_local_datetime(due_at) if due_at else None
    summary = f"创建待办「{title}」" + (f"，截止时间 {due_summary}" if due_summary else "，不设置截止时间")
    return create_pending_operation(
        user_id=user_id,
        conversation_id=conversation_id,
        tool_name="todo",
        operation_type="todo.create",
        risk_level="write",
        summary=summary,
        payload={"title": title, "dueAt": due_text},
        source_text=question,
    )


def create_calendar_event(
    user_id: str,
    conversation_id: str | None,
    title: str,
    starts_at: datetime | None = None,
) -> dict[str, object]:
    if not settings.postgres_enabled:
        return {"ok": False, "content": "PostgreSQL 未启用，无法创建日历事件。"}

    title = title.strip()[:200]
    if not title:
        return {"ok": False, "content": "没有识别到日历事件内容。"}

    now = datetime.now(UTC)
    with postgres_connection() as conn:
        duplicate = _find_duplicate_calendar_event(conn, user_id, title, starts_at)
        if duplicate:
            return {
                "ok": True,
                "content": f"已存在相同日程：{title}，本次没有重复创建。",
                "duplicate": True,
                "event": duplicate,
            }
        row = conn.execute(
            """
            INSERT INTO agent_calendar_event
              (user_id, conversation_id, title, starts_at, status, created_at, updated_at)
            VALUES (%s, %s, %s, %s, 'scheduled', %s, %s)
            RETURNING id, created_at, updated_at
            """,
            (user_id, conversation_id or "", title, starts_at, now, now),
        ).fetchone()
        conn.commit()

    time_text = starts_at.isoformat() if starts_at else None
    return {
        "ok": True,
        "content": f"已创建日历事件：{title}，时间：{time_text or '未设置具体时间'}",
        "event": {
            "id": row[0],
            "title": title,
            "startsAt": time_text,
            "status": "scheduled",
            "conversationId": conversation_id or "",
            "sourceRef": f"conversation:{conversation_id}" if conversation_id else "",
            "createdAt": row[1].isoformat(),
            "updatedAt": row[2].isoformat(),
        },
    }


def create_calendar_event_from_question(user_id: str, conversation_id: str | None, question: str) -> dict[str, object]:
    title = _cleanup_title(question, ["帮我创建日历", "创建日历", "加个日程", "日程", "日历", "calendar"])
    return create_calendar_event(user_id, conversation_id, title, _extract_datetime(question))


def preview_calendar_event_from_question(user_id: str, conversation_id: str | None, question: str) -> dict[str, object]:
    title = _cleanup_title(question, ["帮我创建日历", "创建日历", "加个日程", "日程", "日历", "calendar"])
    starts_at = _extract_datetime(question)
    title = title.strip()[:200]
    if not title:
        return {"ok": False, "content": "没有识别到日历事件内容。"}
    starts_text = starts_at.isoformat() if starts_at else None
    starts_summary = _format_local_datetime(starts_at) if starts_at else None
    summary = f"创建日程「{title}」" + (f"，开始时间 {starts_summary}" if starts_summary else "，不设置开始时间")
    return create_pending_operation(
        user_id=user_id,
        conversation_id=conversation_id,
        tool_name="calendar",
        operation_type="calendar.create",
        risk_level="write",
        summary=summary,
        payload={"title": title, "startsAt": starts_text},
        source_text=question,
    )


def _cleanup_title(question: str, prefixes: list[str]) -> str:
    title = question.strip()
    for prefix in prefixes:
        title = title.replace(prefix, "")
    title = re.sub(r"^(：|:|，|,|\s)+", "", title).strip()
    return _strip_leading_datetime_phrase(title)[:200]


def _format_local_datetime(value: datetime) -> str:
    local_value = value.astimezone(LOCAL_TIMEZONE)
    return (
        f"{local_value.year}/{local_value.month}/{local_value.day} "
        f"{local_value.hour:02d}:{local_value.minute:02d}:{local_value.second:02d}（北京时间）"
    )


def _find_open_duplicate_todo(conn, user_id: str, title: str) -> dict[str, object] | None:
    row = conn.execute(
        """
        SELECT id, title, status, due_at, conversation_id, created_at, updated_at
        FROM agent_todo
        WHERE user_id = %s AND status = 'open' AND lower(title) = lower(%s)
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        (user_id, title),
    ).fetchone()
    if row is None:
        return None
    return {
        "id": row[0],
        "title": row[1],
        "status": row[2],
        "dueAt": row[3].isoformat() if row[3] else None,
        "conversationId": row[4],
        "sourceRef": f"conversation:{row[4]}" if row[4] else "",
        "createdAt": row[5].isoformat(),
        "updatedAt": row[6].isoformat(),
    }


def _find_duplicate_calendar_event(
    conn,
    user_id: str,
    title: str,
    starts_at: datetime | None,
) -> dict[str, object] | None:
    row = conn.execute(
        """
        SELECT id, title, starts_at, status, conversation_id, created_at, updated_at
        FROM agent_calendar_event
        WHERE user_id = %s
          AND status = 'scheduled'
          AND lower(title) = lower(%s)
          AND ((starts_at IS NULL AND %s IS NULL) OR starts_at = %s)
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        (user_id, title, starts_at, starts_at),
    ).fetchone()
    if row is None:
        return None
    return {
        "id": row[0],
        "title": row[1],
        "startsAt": row[2].isoformat() if row[2] else None,
        "status": row[3],
        "conversationId": row[4],
        "sourceRef": f"conversation:{row[4]}" if row[4] else "",
        "createdAt": row[5].isoformat(),
        "updatedAt": row[6].isoformat(),
    }


def infer_task_goal_links(user_id: str, title: str) -> list[dict[str, object]]:
    try:
        profiles = list_user_profiles(user_id, status="active", limit=200)
    except Exception:
        return []
    title_terms = _link_terms(title)
    links: list[dict[str, object]] = []
    for profile in profiles:
        schema_path = str(profile.get("schemaPath") or "")
        if not schema_path.startswith("goals.") and schema_path != "projects.current":
            continue
        value = str(profile.get("summary") or profile.get("value") or "")
        overlap = title_terms & _link_terms(value)
        if not overlap and title and value and (title in value or value in title):
            overlap = {title[:12]}
        if not overlap:
            continue
        links.append({
            "schemaPath": schema_path,
            "label": profile.get("label") or schema_path,
            "value": value,
            "confidence": profile.get("confidence"),
            "reason": "任务标题与当前目标/项目存在关键词重合。",
        })
    return links[:3]


def _link_terms(text: str) -> set[str]:
    chinese_terms = {
        token
        for token in re.split(r"[\s，。！？、,.!?；;：:（）()\[\]【】\"'“”]+", text)
        if len(token) >= 2
    }
    cjk_bigrams = {
        text[index:index + 2]
        for index in range(max(len(text) - 1, 0))
        if re.fullmatch(r"[\u4e00-\u9fff]{2}", text[index:index + 2])
    }
    latin_terms = set(re.findall(r"[a-zA-Z0-9][a-zA-Z0-9._-]{1,}", text.lower()))
    return chinese_terms | cjk_bigrams | latin_terms


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
    local_time = datetime.fromisoformat(f"{date_part}T{time_part}:00").replace(tzinfo=LOCAL_TIMEZONE)
    return local_time.astimezone(UTC)


def _extract_cn_datetime(question: str, now: datetime | None = None) -> datetime | None:
    now = now or datetime.now(LOCAL_TIMEZONE)
    local_now = now.astimezone(LOCAL_TIMEZONE) if now.tzinfo else now.replace(tzinfo=LOCAL_TIMEZONE)
    base_date = _extract_cn_date(question, local_now)
    if base_date is None:
        return None

    hour, minute = _extract_cn_time(question)
    return base_date.replace(hour=hour, minute=minute, second=0, microsecond=0).astimezone(UTC)


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


def _strip_leading_datetime_phrase(title: str) -> str:
    cleaned = re.sub(
        r"^(?:"
        r"(?:今天|明天|后天|今晚|明早|本周|这周|下周[一二三四五六日天]?|周[一二三四五六日天]?)"
        r")?(?:早上|上午|中午|下午|晚上)?"
        r"(?:(?:[零一二两三四五六七八九十\d]{1,3})点(?:半)?|\d{1,2}[:：]\d{2})?"
        r"(?:前|之前)?",
        "",
        title,
    ).strip(" ：:，,。.")
    return cleaned or title


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
