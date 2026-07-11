from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException

from app.automation.automation_store import create_automation
from app.core.config import settings
from app.core.database import postgres_connection
from app.experience.template_catalog import get_template, workflow_payload
from app.experience.events import record_event
from app.workflows.workflow_store import create_workflow


SUPPORTED_INTENTS = {
    "weekly_project_review": ("周报", "每周", "复盘", "weekly"),
    "project_daily_briefing": ("日报", "每日", "每天", "简报", "daily"),
    "workspace_health_check": ("代码", "健康", "git", "todo", "测试"),
    "overdue_and_stale_alert": ("逾期", "到期", "停滞", "提醒"),
}


def create_from_template(user_id: str, request: dict[str, Any], source: str = "template") -> dict[str, Any]:
    _ensure_postgres()
    template_id = str(request.get("templateId") or "")
    template = get_template(template_id)
    if not template or template.get("availability") == "coming_soon":
        raise HTTPException(status_code=400, detail="该模板当前不可用。")
    project_id = _optional_int(request.get("projectId"))
    workspace_id = _optional_int(request.get("workspaceId"))
    _validate_scope(user_id, project_id, workspace_id)
    title = str(request.get("title") or template["name"]).strip()[:256]
    schedule = _validate_schedule(str(request.get("schedule") or template["defaultSchedule"]))
    timezone = _validate_timezone(str(request.get("timezone") or "Asia/Shanghai"))
    workflow = workflow_payload(template, title=title, project_id=project_id)
    with postgres_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO automation_draft
              (user_id, source, template_id, template_version, title, user_goal, project_id,
               workspace_id, schedule_expr, timezone, allowed_sources_json,
               proposed_workflow_json, risk_summary)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb)
            RETURNING id
            """,
            (
                user_id, source, template_id, template["version"], title, request.get("userGoal"),
                project_id, workspace_id, schedule, timezone,
                json.dumps(template["sources"], ensure_ascii=False),
                json.dumps(workflow, ensure_ascii=False),
                json.dumps({"level": "read_only", "writes": False, "message": "只读取已授权范围，不修改文件或外部系统。"}, ensure_ascii=False),
            ),
        ).fetchone()
        conn.commit()
    draft = get_draft(user_id, int(row[0]))
    record_event(user_id, "draft_created", template_id=template_id, status="draft", metadata={"source": source})
    return draft


def create_from_intent(user_id: str, request: dict[str, Any]) -> dict[str, Any]:
    text = str(request.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="请描述希望定期获得的结果。")
    lowered = text.lower()
    if any(word in lowered for word in ("删除", "发送", "发布", "修改", "写入", "delete", "send", "write")):
        return _unsupported("当前只支持只读结果型自动化；写操作需要在结果产生后单独确认。")
    scored = [(sum(1 for word in words if word in lowered), template_id) for template_id, words in SUPPORTED_INTENTS.items()]
    score, template_id = max(scored)
    if score == 0:
        return _unsupported("暂时无法将这个目标安全映射到首批模板。")
    schedule = _schedule_from_text(lowered, get_template(template_id)["defaultSchedule"])
    payload = {**request, "templateId": template_id, "schedule": schedule, "userGoal": text}
    return {"kind": "draft", "draft": create_from_template(user_id, payload, source="chat")}


def get_draft(user_id: str, draft_id: int) -> dict[str, Any]:
    _ensure_postgres()
    with postgres_connection() as conn:
        row = conn.execute(
            """
            SELECT id, source, template_id, template_version, title, user_goal, project_id,
                   workspace_id, schedule_expr, timezone, allowed_sources_json,
                   proposed_workflow_json, risk_summary, status, workflow_id, automation_id,
                   created_at, updated_at, expires_at
            FROM automation_draft WHERE user_id = %s AND id = %s
            """, (user_id, draft_id),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="自动化草稿不存在。")
    keys = ("id", "source", "templateId", "templateVersion", "title", "userGoal", "projectId", "workspaceId", "schedule", "timezone", "allowedSources", "proposedWorkflow", "riskSummary", "status", "workflowId", "automationId")
    result = dict(zip(keys, row[:16]))
    result.update({"createdAt": row[16].isoformat(), "updatedAt": row[17].isoformat(), "expiresAt": row[18].isoformat()})
    return result


def enable_draft(user_id: str, draft_id: int, idempotency_key: str | None) -> dict[str, Any]:
    draft = get_draft(user_id, draft_id)
    if draft["status"] == "enabled":
        return {"draft": draft, "idempotentReplay": True}
    if draft["status"] == "enabling":
        return {"draft": draft, "idempotentReplay": True}
    if draft["status"] != "draft":
        raise HTTPException(status_code=409, detail="该草稿已不可启用。")
    if datetime.fromisoformat(draft["expiresAt"]) <= datetime.now(UTC):
        raise HTTPException(status_code=409, detail="该草稿已过期，请复制后重新确认。")
    key = (idempotency_key or hashlib.sha256(f"{user_id}:{draft_id}:enable".encode()).hexdigest())[:128]
    with postgres_connection() as conn:
        claimed = conn.execute(
            """UPDATE automation_draft SET status='enabling', idempotency_key=%s, updated_at=now()
               WHERE user_id=%s AND id=%s AND status='draft'
               RETURNING id""",
            (key, user_id, draft_id),
        ).fetchone()
        conn.commit()
    if claimed is None:
        return {"draft": get_draft(user_id, draft_id), "idempotentReplay": True}
    try:
        workflow = create_workflow(user_id, draft["proposedWorkflow"])
        automation = create_automation(user_id, {
            "workflowId": workflow["id"], "name": draft["title"], "triggerType": "schedule",
            "scheduleExpr": draft["schedule"], "timezone": draft["timezone"], "status": "enabled",
            "input": {"projectId": draft["projectId"], "workspaceId": draft["workspaceId"], "templateId": draft["templateId"]},
        })
    except Exception:
        with postgres_connection() as conn:
            conn.execute(
                """UPDATE automation_draft SET status='draft', idempotency_key=NULL, updated_at=now()
                   WHERE user_id=%s AND id=%s AND status='enabling'""",
                (user_id, draft_id),
            )
            conn.commit()
        raise
    with postgres_connection() as conn:
        conn.execute(
            """UPDATE automation_draft SET status='enabled', workflow_id=%s, automation_id=%s, updated_at=now()
               WHERE user_id=%s AND id=%s AND status='enabling'""",
            (workflow["id"], automation["id"], user_id, draft_id),
        )
        conn.commit()
    enabled = get_draft(user_id, draft_id)
    record_event(user_id, "draft_enabled", template_id=str(draft["templateId"]), status="enabled")
    return {"draft": enabled, "workflow": workflow, "automation": automation, "idempotentReplay": False}


def discard_draft(user_id: str, draft_id: int) -> dict[str, Any]:
    get_draft(user_id, draft_id)
    with postgres_connection() as conn:
        conn.execute("UPDATE automation_draft SET status='discarded', updated_at=now() WHERE user_id=%s AND id=%s AND status='draft'", (user_id, draft_id))
        conn.commit()
    discarded = get_draft(user_id, draft_id)
    record_event(user_id, "draft_discarded", template_id=str(discarded["templateId"]), status="discarded")
    return discarded


def _unsupported(reason: str) -> dict[str, Any]:
    return {"kind": "unsupported_intent", "unsupportedReason": reason, "recommendedTemplateIds": ["weekly_project_review", "project_daily_briefing", "workspace_health_check", "overdue_and_stale_alert"]}


def _schedule_from_text(text: str, fallback: str) -> str:
    time_match = re.search(r"(?:下午|晚上)?\s*(\d{1,2})\s*(?:点|:)(\d{1,2})?", text)
    hour = int(time_match.group(1)) if time_match else None
    minute = int(time_match.group(2) or 0) if time_match else 0
    if hour is not None and ("下午" in text or "晚上" in text) and hour < 12:
        hour += 12
    fallback_parts = fallback.split(":")
    fallback_time = ":".join(fallback_parts[-2:]) if len(fallback_parts) >= 3 else fallback_parts[-1]
    time_part = f"{hour:02d}:{minute:02d}" if hour is not None and hour < 24 and minute < 60 else fallback_time
    weekdays = {"周一": "mon", "周二": "tue", "周三": "wed", "周四": "thu", "周五": "fri", "周六": "sat", "周日": "sun"}
    for label, value in weekdays.items():
        if label in text:
            return f"weekly:{value}:{time_part}"
    if any(word in text for word in ("每天", "每日", "日报")):
        return f"daily:{time_part}"
    return fallback


def _validate_scope(user_id: str, project_id: int | None, workspace_id: int | None) -> None:
    with postgres_connection() as conn:
        if project_id is not None and conn.execute("SELECT 1 FROM project WHERE user_id=%s AND id=%s", (user_id, project_id)).fetchone() is None:
            raise HTTPException(status_code=403, detail="无权访问所选项目。")
        if workspace_id is not None and conn.execute("SELECT 1 FROM workspace WHERE user_id=%s AND id=%s", (user_id, workspace_id)).fetchone() is None:
            raise HTTPException(status_code=403, detail="无权访问所选工作区。")


def _validate_schedule(value: str) -> str:
    if not re.fullmatch(r"(?:daily:\d{2}:\d{2}|weekly:(?:mon|tue|wed|thu|fri|sat|sun):\d{2}:\d{2})", value):
        raise HTTPException(status_code=400, detail="仅支持每日或每周固定时间。")
    return value


def _validate_timezone(value: str) -> str:
    if value not in {"Asia/Shanghai", "UTC"}:
        raise HTTPException(status_code=400, detail="暂时仅支持 Asia/Shanghai 或 UTC 时区。")
    return value


def _optional_int(value: object) -> int | None:
    return int(value) if value not in (None, "") else None


def _ensure_postgres() -> None:
    if not settings.postgres_enabled:
        raise HTTPException(status_code=503, detail="PostgreSQL 未启用。")
