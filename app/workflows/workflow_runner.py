from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import HTTPException

from app.cli.command_executor import run_command
from app.core.config import settings
from app.core.database import postgres_connection
from app.mcp.tool_executor import invoke_mcp_tool_by_name
from app.security.execution_audit import write_execution_audit
from app.workflows.workflow_context import TemplateResolutionError, condition_matches, resolve_template
from app.workflows.workflow_store import get_workflow


logger = logging.getLogger(__name__)


def run_workflow(user_id: str, workflow_id: int, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    if not settings.postgres_enabled:
        raise HTTPException(status_code=503, detail="PostgreSQL 未启用。")
    payload = payload or {}
    workflow = get_workflow(user_id, workflow_id)
    run_input = _workflow_input(payload)
    with postgres_connection() as conn:
        run_row = conn.execute(
            """
            INSERT INTO workflow_run
              (user_id, workflow_id, project_id, trigger_type, status, input, started_at)
            VALUES (%s, %s, %s, %s, 'running', %s::jsonb, now())
            RETURNING id
            """,
            (
                user_id,
                workflow_id,
                workflow.get("projectId"),
                str(payload.get("triggerType") or "manual"),
                json.dumps(run_input, ensure_ascii=False),
            ),
        ).fetchone()
        run_id = int(run_row[0])
        for step in workflow.get("steps", []):
            conn.execute(
                """
                INSERT INTO workflow_step_run
                  (user_id, run_id, step_template_id, sort_order, title, step_type, status, input)
                VALUES (%s, %s, %s, %s, %s, %s, 'pending', %s::jsonb)
                """,
                (
                    user_id,
                    run_id,
                    step["id"],
                    step["sortOrder"],
                    step["title"],
                    step["stepType"],
                    json.dumps(step.get("inputTemplate") or {}, ensure_ascii=False),
                ),
            )
        conn.commit()
    _advance_workflow(user_id, run_id, payload)
    return get_workflow_run(user_id, run_id)


def resume_workflow_run(user_id: str, run_id: int, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or {}
    if not bool(payload.get("confirmed", False)):
        raise HTTPException(status_code=400, detail="继续工作流前必须明确确认当前步骤。")
    run = get_workflow_run(user_id, run_id)
    if run["status"] != "paused":
        raise HTTPException(status_code=409, detail="只有暂停中的工作流可以继续。")
    with postgres_connection() as conn:
        step = conn.execute(
            """
            SELECT id
            FROM workflow_step_run
            WHERE user_id = %s AND run_id = %s AND status = 'awaiting_confirmation'
            ORDER BY sort_order ASC, id ASC
            LIMIT 1
            """,
            (user_id, run_id),
        ).fetchone()
        if step is None:
            raise HTTPException(status_code=409, detail="工作流没有等待确认的步骤。")
        step_id = int(step[0])
        conn.execute(
            """
            UPDATE workflow_step_run
            SET status = 'pending', error_code = NULL, error_message = NULL,
                started_at = NULL, completed_at = NULL
            WHERE user_id = %s AND id = %s
            """,
            (user_id, step_id),
        )
        conn.execute(
            """
            UPDATE workflow_run
            SET status = 'running', error_code = NULL, error_message = NULL, completed_at = NULL
            WHERE user_id = %s AND id = %s
            """,
            (user_id, run_id),
        )
        conn.commit()
    context = _resume_context(run, payload)
    _advance_workflow(
        user_id,
        run_id,
        context,
        confirmed_step_id=step_id,
        second_confirmed=bool(payload.get("secondConfirmed", False)),
    )
    return get_workflow_run(user_id, run_id)


def retry_workflow_run(user_id: str, run_id: int, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or {}
    run = get_workflow_run(user_id, run_id)
    if run["status"] != "failed":
        raise HTTPException(status_code=409, detail="只有失败的工作流可以重试。")
    with postgres_connection() as conn:
        failed_step = conn.execute(
            """
            SELECT sort_order
            FROM workflow_step_run
            WHERE user_id = %s AND run_id = %s AND status = 'failed'
            ORDER BY sort_order ASC, id ASC
            LIMIT 1
            """,
            (user_id, run_id),
        ).fetchone()
        if failed_step is None:
            raise HTTPException(status_code=409, detail="工作流没有可重试的失败步骤。")
        conn.execute(
            """
            UPDATE workflow_step_run
            SET status = 'pending', output = NULL, tool_call_id = NULL, cli_run_id = NULL,
                error_code = NULL, error_message = NULL, started_at = NULL, completed_at = NULL
            WHERE user_id = %s AND run_id = %s AND sort_order >= %s
            """,
            (user_id, run_id, int(failed_step[0])),
        )
        conn.execute(
            """
            UPDATE workflow_run
            SET status = 'running', output = NULL, error_code = NULL,
                error_message = NULL, completed_at = NULL
            WHERE user_id = %s AND id = %s
            """,
            (user_id, run_id),
        )
        conn.commit()
    _advance_workflow(user_id, run_id, _resume_context(run, payload))
    return get_workflow_run(user_id, run_id)


def cancel_workflow_run(user_id: str, run_id: int) -> dict[str, Any]:
    run = get_workflow_run(user_id, run_id)
    if run["status"] not in {"running", "paused"}:
        raise HTTPException(status_code=409, detail="只有运行中或暂停中的工作流可以取消。")
    with postgres_connection() as conn:
        conn.execute(
            """
            UPDATE workflow_step_run
            SET status = 'cancelled', error_code = 'RUN_CANCELLED',
                error_message = '工作流已由用户取消。', completed_at = now()
            WHERE user_id = %s AND run_id = %s
              AND status IN ('pending', 'running', 'awaiting_confirmation')
            """,
            (user_id, run_id),
        )
        conn.execute(
            """
            UPDATE workflow_run
            SET status = 'cancelled', error_code = 'RUN_CANCELLED',
                error_message = '工作流已由用户取消。', completed_at = now()
            WHERE user_id = %s AND id = %s
            """,
            (user_id, run_id),
        )
        conn.commit()
    write_execution_audit(
        user_id=user_id,
        source_type="workflow",
        source_id=str(run_id),
        executor_type="workflow",
        executor_name="runner",
        operation="cancel",
        risk_level="read_only",
        status="cancelled",
        output_summary="工作流已由用户取消。",
    )
    return get_workflow_run(user_id, run_id)


def get_workflow_run(user_id: str, run_id: int) -> dict[str, Any]:
    if not settings.postgres_enabled:
        raise HTTPException(status_code=503, detail="PostgreSQL 未启用。")
    with postgres_connection() as conn:
        row = conn.execute(
            """
            SELECT id, workflow_id, project_id, trigger_type, status, input, output,
                   error_code, error_message, started_at, completed_at, created_at
            FROM workflow_run
            WHERE user_id = %s AND id = %s
            """,
            (user_id, run_id),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="工作流运行不存在。")
        steps = conn.execute(
            """
            SELECT wsr.id, wsr.step_template_id, wsr.sort_order, wsr.title, wsr.step_type,
                   wsr.status, wsr.input, wsr.output, wsr.tool_call_id, wsr.cli_run_id,
                   wsr.error_code, wsr.error_message, wsr.started_at, wsr.completed_at,
                   wsr.created_at, wst.output_key, wst.condition_expr
            FROM workflow_step_run wsr
            LEFT JOIN workflow_step_template wst ON wst.id = wsr.step_template_id
            WHERE wsr.user_id = %s AND wsr.run_id = %s
            ORDER BY wsr.sort_order ASC, wsr.id ASC
            """,
            (user_id, run_id),
        ).fetchall()
    return {
        "id": row[0],
        "workflowId": row[1],
        "projectId": row[2],
        "triggerType": row[3],
        "status": row[4],
        "input": row[5] or {},
        "output": row[6] or {},
        "errorCode": row[7],
        "errorMessage": row[8],
        "startedAt": row[9].isoformat() if row[9] else None,
        "completedAt": row[10].isoformat() if row[10] else None,
        "createdAt": row[11].isoformat(),
        "steps": [_step_run_row(step) for step in steps],
    }


def list_workflow_runs(user_id: str, workflow_id: int | None = None, limit: int = 50) -> list[dict[str, Any]]:
    if not settings.postgres_enabled:
        raise HTTPException(status_code=503, detail="PostgreSQL 未启用。")
    limit = max(1, min(limit, 200))
    clauses = ["user_id = %s"]
    params: list[Any] = [user_id]
    if workflow_id is not None:
        clauses.append("workflow_id = %s")
        params.append(workflow_id)
    params.append(limit)
    with postgres_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT id, workflow_id, project_id, trigger_type, status, error_code,
                   error_message, started_at, completed_at, created_at
            FROM workflow_run
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at DESC
            LIMIT %s
            """,
            tuple(params),
        ).fetchall()
        run_ids = [int(row[0]) for row in rows]
        if not run_ids:
            return []
        step_rows = conn.execute(
            """
            SELECT run_id, id, step_template_id, sort_order, title, step_type, status,
                   error_code, error_message, started_at, completed_at, created_at
            FROM workflow_step_run
            WHERE user_id = %s AND run_id = ANY(%s::bigint[])
            ORDER BY sort_order ASC, id ASC
            """,
            (user_id, run_ids),
        ).fetchall()
    steps_by_run: dict[int, list[dict[str, Any]]] = {}
    for step in step_rows:
        steps_by_run.setdefault(int(step[0]), []).append(
            {
                "id": step[1],
                "stepTemplateId": step[2],
                "sortOrder": step[3],
                "title": step[4],
                "stepType": step[5],
                "status": step[6],
                "errorCode": step[7],
                "errorMessage": step[8],
                "startedAt": step[9].isoformat() if step[9] else None,
                "completedAt": step[10].isoformat() if step[10] else None,
                "createdAt": step[11].isoformat(),
            }
        )
    return [
        {
            "id": row[0],
            "workflowId": row[1],
            "projectId": row[2],
            "triggerType": row[3],
            "status": row[4],
            "errorCode": row[5],
            "errorMessage": row[6],
            "startedAt": row[7].isoformat() if row[7] else None,
            "completedAt": row[8].isoformat() if row[8] else None,
            "createdAt": row[9].isoformat(),
            "steps": steps_by_run.get(int(row[0]), []),
        }
        for row in rows
    ]


def _advance_workflow(
    user_id: str,
    run_id: int,
    payload: dict[str, Any],
    *,
    confirmed_step_id: int | None = None,
    second_confirmed: bool = False,
) -> None:
    context = _execution_context(user_id, run_id, payload)
    workspace_id = context["input"].get("workspaceId")
    with postgres_connection() as conn:
        steps = conn.execute(
            """
            SELECT wsr.id, wsr.step_template_id, wsr.sort_order, wsr.title, wsr.step_type,
                   wst.command_template, wst.requires_confirmation, wst.tool_name, wsr.input,
                   wst.output_key, wst.condition_expr
            FROM workflow_step_run wsr
            LEFT JOIN workflow_step_template wst ON wst.id = wsr.step_template_id
            WHERE wsr.user_id = %s AND wsr.run_id = %s AND wsr.status = 'pending'
            ORDER BY wsr.sort_order ASC, wsr.id ASC
            """,
            (user_id, run_id),
        ).fetchall()
    for (
        step_id,
        _template_id,
        sort_order,
        title,
        step_type,
        command_template,
        requires_confirmation,
        tool_name,
        step_input,
        output_key,
        condition,
    ) in steps:
        step_id = int(step_id)
        output_key = str(output_key or f"step_{int(sort_order) + 1}")
        try:
            if not condition_matches(condition, context):
                skipped = {"ok": True, "skipped": True, "reason": "步骤条件未满足。"}
                _skip_step(user_id, step_id, skipped)
                context["steps"][output_key] = skipped
                continue
            resolved_input = resolve_template(step_input if isinstance(step_input, dict) else {}, context)
            resolved_command = resolve_template(str(command_template), context) if command_template else None
        except (TemplateResolutionError, ValueError) as exc:
            _fail_step(user_id, run_id, step_id, "INPUT_REFERENCE_ERROR", str(exc))
            return
        _set_step_input(user_id, step_id, resolved_input)
        step_confirmed = confirmed_step_id == step_id
        if requires_confirmation and not step_confirmed:
            _pause_step(
                user_id,
                run_id,
                step_id,
                {
                    "ok": False,
                    "requiresConfirmation": True,
                    "requiresSecondConfirmation": False,
                    "reason": "工作流模板要求执行前确认。",
                },
            )
            return
        with postgres_connection() as conn:
            conn.execute(
                "UPDATE workflow_step_run SET status = 'running', started_at = now() WHERE user_id = %s AND id = %s",
                (user_id, step_id),
            )
            conn.commit()
        if step_type == "cli":
            if workspace_id is None or not resolved_command:
                _fail_step(user_id, run_id, int(step_id), "WORKSPACE_REQUIRED", "CLI 步骤需要 workspaceId 和 commandTemplate。")
                return
            result = run_command(
                user_id,
                int(workspace_id),
                str(resolved_command),
                confirmed=step_confirmed,
                second_confirmed=step_confirmed and second_confirmed,
            )
            if result.get("requiresConfirmation"):
                _pause_step(user_id, run_id, int(step_id), result)
                return
            if not result.get("ok"):
                _fail_step_with_result(user_id, run_id, int(step_id), "STEP_FAILED", f"步骤失败：{title}", result)
                _fail_run(user_id, run_id, "STEP_FAILED", f"步骤失败：{title}")
                return
            _complete_step(user_id, int(step_id), result, cli_run_id=result.get("id"))
            context["steps"][output_key] = result
        elif step_type == "mcp_tool":
            if not tool_name:
                _fail_step(user_id, run_id, int(step_id), "TOOL_REQUIRED", "MCP 步骤需要 toolName。")
                return
            arguments = resolved_input if isinstance(resolved_input, dict) else {}
            if workspace_id is not None:
                arguments = {**arguments, "workspaceId": workspace_id}
            result = invoke_mcp_tool_by_name(
                user_id,
                str(tool_name),
                {
                    "arguments": arguments,
                    "confirmed": step_confirmed,
                    "secondConfirmed": step_confirmed and second_confirmed,
                    "triggerType": str(payload.get("triggerType") or "manual"),
                },
            )
            if result.get("requiresConfirmation"):
                _pause_step(user_id, run_id, int(step_id), result)
                return
            if not result.get("ok"):
                _fail_step_with_result(user_id, run_id, int(step_id), "STEP_FAILED", f"步骤失败：{title}", result)
                _fail_run(user_id, run_id, "STEP_FAILED", f"步骤失败：{title}")
                return
            _complete_step(user_id, int(step_id), result)
            context["steps"][output_key] = result
        else:
            result = {"ok": True, "summary": f"{title} 已完成。", "input": resolved_input}
            _complete_step(user_id, int(step_id), result)
            context["steps"][output_key] = result
    with postgres_connection() as conn:
        completed_steps = conn.execute(
            """
            SELECT id, title, output
            FROM workflow_step_run
            WHERE user_id = %s AND run_id = %s AND status IN ('completed', 'skipped')
            ORDER BY sort_order ASC, id ASC
            """,
            (user_id, run_id),
        ).fetchall()
        combined_outputs = [
            {"stepId": row[0], "title": row[1], "result": row[2] or {}}
            for row in completed_steps
        ]
        conn.execute(
            """
            UPDATE workflow_run
            SET status = 'completed', output = %s::jsonb, completed_at = now()
            WHERE user_id = %s AND id = %s
            """,
            (json.dumps({"steps": combined_outputs}, ensure_ascii=False), user_id, run_id),
        )
        conn.commit()
    write_execution_audit(
        user_id=user_id,
        source_type="workflow",
        source_id=str(run_id),
        executor_type="workflow",
        executor_name="runner",
        operation="run",
        risk_level="read_only",
        status="completed",
        output_summary=f"工作流运行完成，共 {len(combined_outputs)} 步。",
    )
    _write_result_summary_safely(user_id, run_id)


def _workflow_input(payload: dict[str, Any]) -> dict[str, Any]:
    nested = payload.get("input")
    run_input = dict(nested) if isinstance(nested, dict) else {}
    for key, value in payload.items():
        if key not in {"input", "triggerType", "confirmed", "secondConfirmed"}:
            run_input[key] = value
    return run_input


def _resume_context(run: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    context = dict(run.get("input") or {})
    nested = payload.get("input")
    if isinstance(nested, dict):
        context.update(nested)
    for key, value in payload.items():
        if key not in {"input", "confirmed", "secondConfirmed", "triggerType"}:
            context[key] = value
    context["triggerType"] = str(payload.get("triggerType") or run.get("triggerType") or "manual")
    return context


def _execution_context(user_id: str, run_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    with postgres_connection() as conn:
        completed = conn.execute(
            """
            SELECT wsr.sort_order, wsr.output, wst.output_key
            FROM workflow_step_run wsr
            LEFT JOIN workflow_step_template wst ON wst.id = wsr.step_template_id
            WHERE wsr.user_id = %s AND wsr.run_id = %s AND wsr.status IN ('completed', 'skipped')
            ORDER BY wsr.sort_order ASC, wsr.id ASC
            """,
            (user_id, run_id),
        ).fetchall()
    step_outputs = {
        str(output_key or f"step_{int(sort_order) + 1}"): output or {}
        for sort_order, output, output_key in completed
    }
    return {"input": _workflow_input(payload), "steps": step_outputs}


def _set_step_input(user_id: str, step_id: int, resolved_input: Any) -> None:
    with postgres_connection() as conn:
        conn.execute(
            "UPDATE workflow_step_run SET input = %s::jsonb WHERE user_id = %s AND id = %s",
            (json.dumps(resolved_input, ensure_ascii=False), user_id, step_id),
        )
        conn.commit()


def _skip_step(user_id: str, step_id: int, result: dict[str, Any]) -> None:
    with postgres_connection() as conn:
        conn.execute(
            """
            UPDATE workflow_step_run
            SET status = 'skipped', output = %s::jsonb, completed_at = now()
            WHERE user_id = %s AND id = %s
            """,
            (json.dumps(result, ensure_ascii=False), user_id, step_id),
        )
        conn.commit()


def _complete_step(user_id: str, step_id: int, result: dict[str, Any], cli_run_id: object | None = None) -> None:
    with postgres_connection() as conn:
        conn.execute(
            """
            UPDATE workflow_step_run
            SET status = 'completed', output = %s::jsonb, cli_run_id = %s, completed_at = now()
            WHERE user_id = %s AND id = %s
            """,
            (json.dumps(result, ensure_ascii=False), cli_run_id, user_id, step_id),
        )
        conn.commit()


def _pause_step(user_id: str, run_id: int, step_id: int, result: dict[str, Any]) -> None:
    with postgres_connection() as conn:
        conn.execute(
            """
            UPDATE workflow_step_run
            SET status = 'awaiting_confirmation', output = %s::jsonb,
                error_code = 'CONFIRMATION_REQUIRED',
                error_message = '步骤需要用户确认。'
            WHERE user_id = %s AND id = %s
            """,
            (json.dumps(result, ensure_ascii=False), user_id, step_id),
        )
        conn.execute("UPDATE workflow_run SET status = 'paused' WHERE user_id = %s AND id = %s", (user_id, run_id))
        conn.commit()
    _write_result_summary_safely(user_id, run_id)


def _fail_step(user_id: str, run_id: int, step_id: int, code: str, message: str) -> None:
    with postgres_connection() as conn:
        conn.execute(
            """
            UPDATE workflow_step_run
            SET status = 'failed', error_code = %s, error_message = %s, completed_at = now()
            WHERE user_id = %s AND id = %s
            """,
            (code, message, user_id, step_id),
        )
        conn.commit()
    _fail_run(user_id, run_id, code, message)


def _fail_step_with_result(
    user_id: str,
    run_id: int,
    step_id: int,
    code: str,
    message: str,
    result: dict[str, Any],
) -> None:
    with postgres_connection() as conn:
        conn.execute(
            """
            UPDATE workflow_step_run
            SET status = 'failed', output = %s::jsonb,
                error_code = %s, error_message = %s, completed_at = now()
            WHERE user_id = %s AND id = %s
            """,
            (json.dumps(result, ensure_ascii=False), code, message, user_id, step_id),
        )
        conn.commit()


def _fail_run(user_id: str, run_id: int, code: str, message: str) -> None:
    with postgres_connection() as conn:
        conn.execute(
            """
            UPDATE workflow_run
            SET status = 'failed', error_code = %s, error_message = %s, completed_at = now()
            WHERE user_id = %s AND id = %s
            """,
            (code, message, user_id, run_id),
        )
        conn.commit()
    _write_result_summary_safely(user_id, run_id)


def _write_result_summary_safely(user_id: str, run_id: int) -> None:
    """A summary failure must never change the underlying execution outcome."""
    try:
        from app.experience.result_summary_service import ensure_workflow_summary

        ensure_workflow_summary(user_id, run_id)
    except Exception:
        logger.exception("Unable to derive result summary for workflow run %s", run_id)


def _step_run_row(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "id": row[0],
        "stepTemplateId": row[1],
        "sortOrder": row[2],
        "title": row[3],
        "stepType": row[4],
        "status": row[5],
        "input": row[6] or {},
        "output": row[7] or {},
        "toolCallId": row[8],
        "cliRunId": row[9],
        "errorCode": row[10],
        "errorMessage": row[11],
        "startedAt": row[12].isoformat() if row[12] else None,
        "completedAt": row[13].isoformat() if row[13] else None,
        "createdAt": row[14].isoformat(),
        "outputKey": row[15],
        "condition": row[16] or None,
    }
