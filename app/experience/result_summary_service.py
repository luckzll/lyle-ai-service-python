from __future__ import annotations

import json
import re
from typing import Any

from fastapi import HTTPException

from app.core.config import settings
from app.core.database import postgres_connection
from app.workflows.workflow_runner import get_workflow_run
from app.experience.events import record_event


SECRET_PATTERN = re.compile(r"(?i)(token|secret|password|api[_-]?key)\s*[:=]\s*[^\s,}]+")


def ensure_workflow_summary(user_id: str, run_id: int) -> dict[str, Any]:
    run = get_workflow_run(user_id, run_id)
    with postgres_connection() as conn:
        existing = conn.execute("SELECT id FROM execution_result_summary WHERE source_type='workflow_run' AND source_id=%s", (str(run_id),)).fetchone()
        if existing:
            return get_result(user_id, int(existing[0]), track_view=False)
        status = {"completed": "completed", "failed": "failed", "paused": "awaiting_confirmation"}.get(run["status"], "partial")
        completed = [step["title"] for step in run["steps"] if step["status"] in {"completed", "skipped"}]
        risks = [step.get("errorMessage") for step in run["steps"] if step.get("errorMessage")]
        title = "执行已完成" if status == "completed" else "执行需要处理"
        conclusion = "已完成只读检查，可以查看建议并决定下一步。" if status == "completed" else _mask(run.get("errorMessage") or "部分步骤尚未完成。")
        actions = ([{"type": "view", "label": "查看详情"}] if status == "completed" else [{"type": "retry", "label": "重试本次运行"}, {"type": "view", "label": "查看技术详情"}])
        duration = None
        if run.get("startedAt") and run.get("completedAt"):
            from datetime import datetime
            duration = int((datetime.fromisoformat(run["completedAt"]) - datetime.fromisoformat(run["startedAt"])).total_seconds() * 1000)
        row = conn.execute(
            """INSERT INTO execution_result_summary(user_id, source_type, source_id, project_id,
               template_id, status, title, conclusion, completed_items_json, risks_json,
               recommended_actions_json, data_scope_json, duration_ms, trace_ref)
               VALUES (%s,'workflow_run',%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s)
               ON CONFLICT(source_type, source_id) DO UPDATE SET status=EXCLUDED.status, conclusion=EXCLUDED.conclusion
               RETURNING id""",
            (user_id, str(run_id), run.get("projectId"), (run.get("input") or {}).get("templateId"), status, title, conclusion,
             json.dumps(completed, ensure_ascii=False), json.dumps([_mask(str(x)) for x in risks], ensure_ascii=False),
             json.dumps(actions, ensure_ascii=False), json.dumps(["所选项目", "已授权工作区"], ensure_ascii=False), duration, f"workflow_run:{run_id}"),
        ).fetchone()
        conn.commit()
    return get_result(user_id, int(row[0]), track_view=False)


def list_results(user_id: str, status: str | None = None, project_id: int | None = None, limit: int = 50) -> dict[str, Any]:
    _ensure()
    with postgres_connection() as conn:
        runs = conn.execute("SELECT id FROM workflow_run WHERE user_id=%s ORDER BY created_at DESC LIMIT %s", (user_id, min(limit, 100))).fetchall()
    for run in runs:
        ensure_workflow_summary(user_id, int(run[0]))
    clauses, params = ["user_id=%s"], [user_id]
    if status:
        clauses.append("status=%s"); params.append(status)
    if project_id is not None:
        clauses.append("project_id=%s"); params.append(project_id)
    params.append(max(1, min(limit, 100)))
    with postgres_connection() as conn:
        rows = conn.execute(f"SELECT id FROM execution_result_summary WHERE {' AND '.join(clauses)} ORDER BY created_at DESC LIMIT %s", params).fetchall()
    return {"items": [get_result(user_id, int(row[0])) for row in rows], "nextCursor": None}


def get_result(user_id: str, result_id: int, *, track_view: bool = True) -> dict[str, Any]:
    _ensure()
    with postgres_connection() as conn:
        row = conn.execute("""SELECT id, source_type, source_id, project_id, template_id, status, title,
          conclusion, completed_items_json, risks_json, recommended_actions_json, data_scope_json,
          duration_ms, trace_ref, created_at FROM execution_result_summary WHERE user_id=%s AND id=%s""", (user_id, result_id)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="执行结果不存在。")
    keys = ("id", "sourceType", "sourceId", "projectId", "templateId", "status", "title", "conclusion", "completedItems", "risks", "recommendedActions", "dataScope", "durationMs", "traceRef")
    result = {**dict(zip(keys, row[:14])), "createdAt": row[14].isoformat()}
    if track_view:
        record_event(user_id, "result_viewed", template_id=result.get("templateId"), status=str(result["status"]))
    return result


def needs_attention(user_id: str) -> dict[str, Any]:
    result = list_results(user_id, limit=100)
    return {"items": [item for item in result["items"] if item["status"] in {"failed", "partial", "awaiting_confirmation"}]}


def _mask(value: str) -> str:
    return SECRET_PATTERN.sub(lambda match: f"{match.group(1)}=[已隐藏]", value)[:2000]


def _ensure() -> None:
    if not settings.postgres_enabled:
        raise HTTPException(status_code=503, detail="PostgreSQL 未启用。")
