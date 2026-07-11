from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

WEEKDAYS = {
    "mon": 0,
    "tue": 1,
    "wed": 2,
    "thu": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}


def next_run_at(schedule_expr: str, timezone_name: str = "Asia/Shanghai", now: datetime | None = None) -> datetime:
    now_utc = now.astimezone(UTC) if now else datetime.now(UTC)
    schedule_expr = (schedule_expr or "daily").strip().lower()
    timezone = _timezone(timezone_name)
    local_now = now_utc.astimezone(timezone)

    if schedule_expr == "hourly":
        return (now_utc + timedelta(hours=1)).replace(microsecond=0)

    if schedule_expr.startswith("daily@") or schedule_expr.startswith("daily:"):
        separator = "@" if "@" in schedule_expr else ":"
        target_time = _parse_hhmm(schedule_expr.split(separator, 1)[1])
        target = local_now.replace(
            hour=target_time.hour,
            minute=target_time.minute,
            second=0,
            microsecond=0,
        )
        if target <= local_now:
            target += timedelta(days=1)
        return target.astimezone(UTC)

    if schedule_expr.startswith("weekly:"):
        day_and_time = schedule_expr.split(":", 1)[1]
        if "@" in day_and_time:
            day, raw_time = day_and_time.split("@", 1)
        elif ":" in day_and_time:
            day, raw_time = day_and_time.split(":", 1)
        else:
            day, raw_time = day_and_time, "09:00"
        if day not in WEEKDAYS:
            raise ValueError("weekly schedule day must be one of mon/tue/wed/thu/fri/sat/sun")
        target_time = _parse_hhmm(raw_time)
        days_ahead = (WEEKDAYS[day] - local_now.weekday()) % 7
        target = local_now.replace(
            hour=target_time.hour,
            minute=target_time.minute,
            second=0,
            microsecond=0,
        ) + timedelta(days=days_ahead)
        if target <= local_now:
            target += timedelta(days=7)
        return target.astimezone(UTC)

    if schedule_expr == "weekly":
        return (now_utc + timedelta(days=7)).replace(microsecond=0)

    if schedule_expr == "daily":
        return (now_utc + timedelta(days=1)).replace(microsecond=0)

    raise ValueError("scheduleExpr must be hourly, daily, daily@HH:MM, daily:HH:MM, weekly, weekly:day@HH:MM, or weekly:day:HH:MM")


def _parse_hhmm(raw: str) -> time:
    try:
        hour_text, minute_text = raw.split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
    except ValueError as exc:
        raise ValueError("schedule time must use HH:MM") from exc
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError("schedule time must use a valid 24-hour HH:MM value")
    return time(hour=hour, minute=minute)


def _timezone(timezone_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(timezone_name or "Asia/Shanghai")
    except ZoneInfoNotFoundError as exc:
        raise ValueError("timezone is not valid") from exc
