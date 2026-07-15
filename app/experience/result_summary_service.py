from __future__ import annotations

import base64
import binascii
import json
from datetime import datetime
from typing import Any

from fastapi import HTTPException

from app.core.config import settings
from app.core.database import postgres_connection
from app.workflows.workflow_runner import get_workflow_run
from app.experience.events import record_event


RESULT_SELECT = """
  SELECT result.id, result.source_type, result.source_id, result.project_id,
    result.template_id, result.status, result.title, result.conclusion,
    result.completed_items_json, result.risks_json, result.recommended_actions_json,
    result.data_scope_json, result.duration_ms, result.trace_ref,
    automation_run.automation_id, automation.name, result.created_at
  FROM execution_result_summary result
  LEFT JOIN LATERAL (
    SELECT run.automation_id
    FROM automation_run run
    WHERE run.user_id = result.user_id
      AND (
        (
          result.source_type = 'workflow_run'
          AND run.workflow_run_id = CASE
            WHEN result.source_id ~ '^[0-9]+$' THEN result.source_id::bigint
          END
        )
        OR (
          result.source_type = 'automation'
          AND run.id = CASE
            WHEN result.source_id ~ '^[0-9]+$' THEN result.source_id::bigint
          END
        )
      )
    ORDER BY run.created_at DESC, run.id DESC
    LIMIT 1
  ) automation_run ON true
  LEFT JOIN automation_rule automation
    ON automation.id = automation_run.automation_id AND automation.user_id = result.user_id
"""


def ensure_workflow_summary(user_id: str, run_id: int) -> dict[str, Any]:
    run = get_workflow_run(user_id, run_id)
    with postgres_connection() as conn:
        status = {"completed": "completed", "failed": "failed", "paused": "awaiting_confirmation"}.get(run["status"], "partial")
        completed = [step["title"] for step in run["steps"] if step["status"] in {"completed", "skipped"}]
        incomplete_steps = [
            step["title"]
            for step in run["steps"]
            if step["status"] in {"failed", "awaiting_confirmation", "cancelled"}
        ]
        risks = [f"{title}尚未完成，请查看技术详情。" for title in incomplete_steps]
        title = "执行已完成" if status == "completed" else "执行需要处理"
        conclusion = {
            "completed": "已完成只读检查，可以查看建议并决定下一步。",
            "failed": "执行未完成，请检查连接、权限或输入配置后重试。",
            "awaiting_confirmation": "执行已暂停，等待确认后可继续。",
            "partial": "部分步骤尚未完成，请查看状态后继续。",
        }[status]
        actions = (
            [{"type": "view", "label": "查看详情"}]
            if status == "completed"
            else [{"type": "resume_workflow", "label": "确认并继续"}]
            if status == "awaiting_confirmation"
            else [{"type": "retry", "label": "重试本次运行"}, {"type": "view", "label": "查看技术详情"}]
        )
        duration = None
        if run.get("startedAt") and run.get("completedAt"):
            duration = int((datetime.fromisoformat(run["completedAt"]) - datetime.fromisoformat(run["startedAt"])).total_seconds() * 1000)
        row = conn.execute(
            """INSERT INTO execution_result_summary(user_id, source_type, source_id, project_id,
               template_id, status, title, conclusion, completed_items_json, risks_json,
               recommended_actions_json, data_scope_json, duration_ms, trace_ref)
               VALUES (%s,'workflow_run',%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s)
               ON CONFLICT(source_type, source_id) DO UPDATE SET
                 project_id=EXCLUDED.project_id, template_id=EXCLUDED.template_id,
                 status=EXCLUDED.status, title=EXCLUDED.title, conclusion=EXCLUDED.conclusion,
                 completed_items_json=EXCLUDED.completed_items_json,
                 risks_json=EXCLUDED.risks_json,
                 recommended_actions_json=EXCLUDED.recommended_actions_json,
                 data_scope_json=EXCLUDED.data_scope_json,
                 duration_ms=EXCLUDED.duration_ms, trace_ref=EXCLUDED.trace_ref
               RETURNING id""",
            (user_id, str(run_id), run.get("projectId"), (run.get("input") or {}).get("templateId"), status, title, conclusion,
             json.dumps(completed, ensure_ascii=False), json.dumps(risks, ensure_ascii=False),
             json.dumps(actions, ensure_ascii=False), json.dumps(["所选项目", "已授权工作区"], ensure_ascii=False), duration, f"workflow_run:{run_id}"),
        ).fetchone()
        conn.commit()
    return get_result(user_id, int(row[0]), track_view=False)


def ensure_automation_failure_summary(
    user_id: str,
    automation_run_id: int,
    *,
    automation_name: str,
    project_id: int | None,
) -> dict[str, Any]:
    """Create a user-readable failure even when no workflow run was created."""
    title = f"{automation_name[:180] or '自动化'}需要处理"
    conclusion = "自动化运行失败，请检查连接、权限或输入配置后重试。"
    with postgres_connection() as conn:
        row = conn.execute(
            """INSERT INTO execution_result_summary(
                 user_id, source_type, source_id, project_id, status, title, conclusion,
                 completed_items_json, risks_json, recommended_actions_json, data_scope_json,
                 trace_ref)
               VALUES (%s, 'automation', %s, %s, 'failed', %s, %s, '[]'::jsonb, %s::jsonb,
                 %s::jsonb, '[]'::jsonb, %s)
               ON CONFLICT(source_type, source_id) DO UPDATE SET
                 status = EXCLUDED.status, title = EXCLUDED.title, conclusion = EXCLUDED.conclusion,
                 project_id = COALESCE(EXCLUDED.project_id, execution_result_summary.project_id),
                 risks_json = EXCLUDED.risks_json,
                 recommended_actions_json = EXCLUDED.recommended_actions_json,
                 trace_ref = EXCLUDED.trace_ref
               RETURNING id""",
            (
                user_id,
                str(automation_run_id),
                project_id,
                title,
                conclusion,
                json.dumps([conclusion], ensure_ascii=False),
                json.dumps([
                    {"type": "retry_automation", "label": "重新运行自动化"},
                    {"type": "view", "label": "查看失败详情"},
                ], ensure_ascii=False),
                f"automation_run:{automation_run_id}",
            ),
        ).fetchone()
        conn.commit()
    return get_result(user_id, int(row[0]), track_view=False)


def list_results(
    user_id: str,
    status: str | None = None,
    project_id: int | None = None,
    limit: int = 50,
    *,
    cursor: str | None = None,
    automation_id: int | None = None,
) -> dict[str, Any]:
    _ensure()
    with postgres_connection() as conn:
        runs = conn.execute(
            """
            SELECT run.id
            FROM workflow_run run
            LEFT JOIN execution_result_summary result
              ON result.user_id = run.user_id
             AND result.source_type = 'workflow_run'
             AND result.source_id = run.id::text
            WHERE run.user_id = %s
              AND (
                result.id IS NULL
                OR result.status <> CASE run.status
                  WHEN 'completed' THEN 'completed'
                  WHEN 'failed' THEN 'failed'
                  WHEN 'paused' THEN 'awaiting_confirmation'
                  ELSE 'partial'
                END
              )
            ORDER BY run.created_at DESC
            LIMIT %s
            """,
            (user_id, min(limit, 100)),
        ).fetchall()
    for run in runs:
        ensure_workflow_summary(user_id, int(run[0]))
    clauses, params = ["result.user_id=%s"], [user_id]
    if status:
        clauses.append("result.status=%s"); params.append(status)
    if project_id is not None:
        clauses.append("result.project_id=%s"); params.append(project_id)
    if automation_id is not None:
        clauses.append("automation_run.automation_id=%s"); params.append(automation_id)
    if cursor:
        cursor_created_at, cursor_id = _decode_cursor(cursor)
        clauses.append("(result.created_at, result.id) < (%s, %s)")
        params.extend((cursor_created_at, cursor_id))
    page_limit = max(1, min(limit, 100))
    params.append(page_limit + 1)
    with postgres_connection() as conn:
        rows = conn.execute(
            f"{RESULT_SELECT} WHERE {' AND '.join(clauses)} "
            "ORDER BY result.created_at DESC, result.id DESC LIMIT %s",
            params,
        ).fetchall()
    has_more = len(rows) > page_limit
    page_rows = rows[:page_limit]
    next_cursor = (
        _encode_cursor(page_rows[-1][16], int(page_rows[-1][0]))
        if has_more and page_rows
        else None
    )
    return {
        "items": [_result_row(row) for row in page_rows],
        "nextCursor": next_cursor,
        "hasMore": has_more,
    }


def get_result(user_id: str, result_id: int, *, track_view: bool = True) -> dict[str, Any]:
    _ensure()
    with postgres_connection() as conn:
        row = conn.execute(
            f"{RESULT_SELECT} WHERE result.user_id=%s AND result.id=%s",
            (user_id, result_id),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="执行结果不存在。")
    result = _result_row(row)
    if track_view:
        record_event(user_id, "result_viewed", template_id=result.get("templateId"), status=str(result["status"]))
    return result


def _result_row(row: tuple[Any, ...]) -> dict[str, Any]:
    keys = (
        "id", "sourceType", "sourceId", "projectId", "templateId", "status", "title",
        "conclusion", "completedItems", "risks", "recommendedActions", "dataScope",
        "durationMs", "traceRef", "automationId", "automationName",
    )
    return {**dict(zip(keys, row[:16])), "createdAt": row[16].isoformat()}


def _encode_cursor(created_at: datetime, result_id: int) -> str:
    payload = json.dumps(
        {"v": 1, "createdAt": created_at.isoformat(), "id": result_id},
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_cursor(cursor: str) -> tuple[datetime, int]:
    try:
        if not cursor or len(cursor) > 256:
            raise ValueError("invalid cursor length")
        padded = cursor + "=" * (-len(cursor) % 4)
        raw = base64.b64decode(padded, altchars=b"-_", validate=True)
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict) or payload.get("v") != 1:
            raise ValueError("unsupported cursor version")
        created_at = datetime.fromisoformat(str(payload["createdAt"]))
        result_id = payload["id"]
        if created_at.tzinfo is None:
            raise ValueError("cursor timestamp must include timezone")
        if isinstance(result_id, bool) or not isinstance(result_id, int) or result_id <= 0:
            raise ValueError("invalid cursor result id")
        return created_at, result_id
    except (binascii.Error, KeyError, TypeError, UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="分页游标无效。") from exc


def needs_attention(user_id: str) -> dict[str, Any]:
    result = list_results(user_id, limit=100)
    return {"items": [item for item in result["items"] if item["status"] in {"failed", "partial", "awaiting_confirmation"}]}


def _ensure() -> None:
    if not settings.postgres_enabled:
        raise HTTPException(status_code=503, detail="PostgreSQL 未启用。")
