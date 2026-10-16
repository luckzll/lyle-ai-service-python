from __future__ import annotations

import base64
import binascii
import json
import re
from datetime import datetime
from typing import Any

from fastapi import HTTPException

from app.core.config import settings
from app.core.database import postgres_connection
from app.workflows.workflow_runner import get_workflow_run
from app.experience.events import record_event
from app.security.secret_masker import mask_secrets


RESULT_SELECT = """
  SELECT result.id, result.source_type, result.source_id, result.project_id,
    result.template_id, result.status, result.title, result.conclusion,
    result.completed_items_json, result.risks_json, result.recommended_actions_json,
    result.data_scope_json, result.duration_ms, result.trace_ref,
    automation_run.automation_id, automation.name, result.dismissed_at, result.created_at,
    source_workflow.name, source_run.output
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
  LEFT JOIN workflow_run source_run
    ON result.source_type = 'workflow_run' AND source_run.user_id = result.user_id
    AND source_run.id = CASE WHEN result.source_id ~ '^[0-9]+$' THEN result.source_id::bigint END
  LEFT JOIN workflow_template source_workflow
    ON source_workflow.id = source_run.workflow_id AND source_workflow.user_id = result.user_id
"""

RESULT_FROM = RESULT_SELECT[RESULT_SELECT.index("  FROM execution_result_summary result"):]
RESULT_STATUSES = ("completed", "partial", "failed", "awaiting_confirmation")
GENERIC_TITLES = {"执行已完成", "执行需要处理", "已完成", "检查完成"}
GENERIC_CONCLUSIONS = {
    "已完成只读检查，可以查看建议并决定下一步。",
    "执行未完成，请检查连接、权限或输入配置后重试。",
    "执行已暂停，等待确认后可继续。",
    "部分步骤尚未完成，请查看状态后继续。",
}


def ensure_workflow_summary(user_id: str, run_id: int) -> dict[str, Any]:
    run = get_workflow_run(user_id, run_id)
    with postgres_connection() as conn:
        status = {"completed": "completed", "failed": "failed", "paused": "awaiting_confirmation"}.get(run["status"], "partial")
        completed = [_safe_line(step["title"]) for step in run["steps"] if step["status"] == "completed"]
        incomplete_steps = [
            step["title"]
            for step in run["steps"]
            if step["status"] in {"failed", "awaiting_confirmation", "cancelled"}
        ]
        risks = [f"{title}尚未完成，请查看技术详情。" for title in incomplete_steps]
        workflow_name = ""
        if run.get("workflowId"):
            named = conn.execute(
                "SELECT name FROM workflow_template WHERE user_id = %s AND id = %s",
                (user_id, run["workflowId"]),
            ).fetchone()
            workflow_name = str(named[0]) if named else ""
        title = _result_title(workflow_name or f"工作流运行 #{run_id}", status)
        conclusion = _specific_conclusion(status, completed, run.get("output"), run.get("steps"))
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
                 duration_ms=EXCLUDED.duration_ms, trace_ref=EXCLUDED.trace_ref,
                 dismissed_at=NULL
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
                 trace_ref = EXCLUDED.trace_ref,
                 dismissed_at = NULL
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
    attention_only: bool = False,
) -> dict[str, Any]:
    _ensure()
    page_limit = max(1, min(limit, 100))
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
            (user_id, page_limit),
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
    if attention_only:
        clauses.append("result.dismissed_at IS NULL")
        clauses.append("result.status IN ('failed', 'partial', 'awaiting_confirmation')")
    count_clauses, count_params = list(clauses), list(params)
    if cursor:
        cursor_created_at, cursor_id = _decode_cursor(cursor)
        clauses.append("(result.created_at, result.id) < (%s, %s)")
        params.extend((cursor_created_at, cursor_id))
    params.append(page_limit + 1)
    with postgres_connection() as conn:
        counts = conn.execute(
            f"SELECT result.status, COUNT(*) {RESULT_FROM} WHERE {' AND '.join(count_clauses)} GROUP BY result.status",
            count_params,
        ).fetchall()
        rows = conn.execute(
            f"{RESULT_SELECT} WHERE {' AND '.join(clauses)} "
            "ORDER BY result.created_at DESC, result.id DESC LIMIT %s",
            params,
        ).fetchall()
    has_more = len(rows) > page_limit
    page_rows = rows[:page_limit]
    next_cursor = (
        _encode_cursor(page_rows[-1][17], int(page_rows[-1][0]))
        if has_more and page_rows
        else None
    )
    return {
        "items": [_result_row(row) for row in page_rows],
        "nextCursor": next_cursor,
        "hasMore": has_more,
        "statusCounts": {status: int(dict(counts).get(status, 0)) for status in RESULT_STATUSES},
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
        "dismissedAt",
    )
    result = {**dict(zip(keys, row[:17])), "createdAt": row[17].isoformat()}
    # Improve presentation of legacy generic copies without rewriting the stored
    # historical summary or doing per-item source lookups.
    source_name = (row[18] if len(row) > 18 else None) or result.get("automationName")
    output = row[19] if len(row) > 19 else None
    if result["title"] in GENERIC_TITLES:
        result["recordedTitle"] = result["title"]
        fallback = f"工作流运行 #{result['sourceId']}" if result["sourceType"] == "workflow_run" else f"执行记录 #{result['id']}"
        result["title"] = _result_title(source_name or fallback, str(result["status"]))
    if result["conclusion"] in GENERIC_CONCLUSIONS:
        result["recordedConclusion"] = result["conclusion"]
        result["conclusion"] = _specific_conclusion(str(result["status"]), result.get("completedItems") or [], output)
    return result


def _safe_line(value: object, limit: int = 200) -> str:
    return re.sub(r"\s+", " ", mask_secrets(str(value or ""))).strip()[:limit]


def _result_title(name: object, status: str) -> str:
    suffix = {"completed": "已完成", "failed": "未完成", "partial": "部分完成", "awaiting_confirmation": "等待确认"}[status]
    return f"{_safe_line(name, 180)} · {suffix}"


def _specific_conclusion(status: str, completed: list, output: object = None, steps: list | None = None) -> str:
    # Only labelled textual summaries are suitable here. Raw input, errors and
    # arbitrary tool payloads can contain credentials and are never serialized.
    if status == "completed" and isinstance(output, dict):
        candidates = [output]
        candidates.extend(item.get("result") for item in reversed(output.get("steps") or []) if isinstance(item, dict))
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            for key in ("conclusion", "summary"):
                value = candidate.get(key)
                if (isinstance(value, str) and value.strip() and value.strip() not in GENERIC_CONCLUSIONS
                        and not re.fullmatch(r".+?\s*已完成[。.!！]?", value.strip())):
                    return _safe_line(value, 500)
    labels = [_safe_line(title) for title in completed if isinstance(title, str) and title.strip()]
    if status == "completed":
        if labels:
            return f"运行报告完成 {len(labels)} 个步骤：{'；'.join(labels[:3])}" + ("等。" if len(labels) > 3 else "。")
        return "该运行记录为已完成，历史摘要未保存具体结论；可打开来源查看原始记录。"
    base = {
        "failed": "执行未完成，请检查连接、权限或输入配置后重试。",
        "awaiting_confirmation": "执行已暂停，等待确认后可继续。",
        "partial": "部分步骤尚未完成，请查看状态后继续。",
    }[status]
    blocked = [_safe_line(step.get("title")) for step in steps or []
               if step.get("status") in {"failed", "awaiting_confirmation", "cancelled"}]
    if blocked:
        base += "未完成步骤：" + "；".join(blocked[:3]) + "。"
    if labels:
        base += "已完成：" + "；".join(labels[:3]) + "。"
    return base


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
    result = list_results(user_id, limit=100, attention_only=True)
    return {"items": result["items"]}


def dismiss_result(user_id: str, result_id: int) -> dict[str, Any]:
    """Remove a resolved failure/partial result from attention without deleting history."""
    _ensure()
    with postgres_connection() as conn:
        updated = conn.execute(
            """UPDATE execution_result_summary
               SET dismissed_at = COALESCE(dismissed_at, now())
               WHERE user_id = %s AND id = %s AND status IN ('failed', 'partial')
               RETURNING id""",
            (user_id, result_id),
        ).fetchone()
        if updated is None:
            existing = conn.execute(
                "SELECT status FROM execution_result_summary WHERE user_id = %s AND id = %s",
                (user_id, result_id),
            ).fetchone()
            if existing is None:
                raise HTTPException(status_code=404, detail="执行结果不存在。")
            raise HTTPException(status_code=409, detail="只有失败或部分完成的结果可以移出待处理。")
        conn.commit()
    result = get_result(user_id, result_id, track_view=False)
    record_event(user_id, "result_dismissed", status=str(result["status"]), metadata={"entityId": result_id})
    return result


def restore_result(user_id: str, result_id: int) -> dict[str, Any]:
    """Return a dismissed failure/partial result to attention without changing history."""
    _ensure()
    with postgres_connection() as conn:
        updated = conn.execute(
            """UPDATE execution_result_summary
               SET dismissed_at = NULL
               WHERE user_id = %s AND id = %s AND status IN ('failed', 'partial')
               RETURNING id""",
            (user_id, result_id),
        ).fetchone()
        if updated is None:
            existing = conn.execute(
                "SELECT status FROM execution_result_summary WHERE user_id = %s AND id = %s",
                (user_id, result_id),
            ).fetchone()
            if existing is None:
                raise HTTPException(status_code=404, detail="执行结果不存在。")
            raise HTTPException(status_code=409, detail="只有失败或部分完成的结果可以恢复到待处理。")
        conn.commit()
    result = get_result(user_id, result_id, track_view=False)
    record_event(user_id, "result_restored", status=str(result["status"]), metadata={"entityId": result_id})
    return result


def _ensure() -> None:
    if not settings.postgres_enabled:
        raise HTTPException(status_code=503, detail="PostgreSQL 未启用。")
