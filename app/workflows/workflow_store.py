from __future__ import annotations

import json
import re
from typing import Any

from fastapi import HTTPException

from app.core.config import settings
from app.core.database import postgres_connection
from app.plans.plan_store import get_plan
from app.workflows.workflow_context import validate_condition
from app.workflows.workflow_plan_mapper import plan_to_workflow_payload


def create_workflow(user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_postgres()
    name = str(payload.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="工作流名称不能为空。")
    steps = payload.get("steps") if isinstance(payload.get("steps"), list) else []
    with postgres_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO workflow_template
              (user_id, project_id, name, description, status, trigger_type, created_from_plan_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                user_id,
                _optional_int(payload.get("projectId")),
                name[:256],
                payload.get("description"),
                str(payload.get("status") or "active"),
                str(payload.get("triggerType") or "manual"),
                _optional_int(payload.get("createdFromPlanId")),
            ),
        ).fetchone()
        workflow_id = int(row[0])
        _insert_steps(conn, user_id, workflow_id, steps)
        conn.commit()
    return get_workflow(user_id, workflow_id)


def update_workflow(user_id: str, workflow_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_postgres()
    current = get_workflow(user_id, workflow_id)
    name = str(payload.get("name") or current["name"]).strip()
    if not name:
        raise HTTPException(status_code=400, detail="工作流名称不能为空。")
    steps = payload.get("steps") if isinstance(payload.get("steps"), list) else current.get("steps", [])
    with postgres_connection() as conn:
        row = conn.execute(
            """
            UPDATE workflow_template
            SET project_id = %s, name = %s, description = %s, status = %s,
                trigger_type = %s, updated_at = now()
            WHERE user_id = %s AND id = %s
            RETURNING id
            """,
            (
                _optional_int(payload.get("projectId") if "projectId" in payload else current.get("projectId")),
                name[:256],
                payload.get("description") if "description" in payload else current.get("description"),
                str(payload.get("status") or current.get("status") or "active"),
                str(payload.get("triggerType") or current.get("triggerType") or "manual"),
                user_id,
                workflow_id,
            ),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="工作流不存在。")
        conn.execute(
            "DELETE FROM workflow_step_template WHERE user_id = %s AND workflow_id = %s",
            (user_id, workflow_id),
        )
        _insert_steps(conn, user_id, workflow_id, steps)
        conn.commit()
    return get_workflow(user_id, workflow_id)


def create_workflow_from_plan(user_id: str, plan_id: int) -> dict[str, Any]:
    _ensure_postgres()
    plan = get_plan(user_id, plan_id)
    with postgres_connection() as conn:
        tool_rows = conn.execute(
            """
            SELECT tool.tool_name
            FROM mcp_tool tool
            JOIN mcp_server server ON server.id = tool.server_id
            WHERE tool.user_id = %s AND tool.enabled = true AND server.status = 'enabled'
            """,
            (user_id,),
        ).fetchall()
        payload = plan_to_workflow_payload(plan, {str(row[0]) for row in tool_rows})
        if not payload["steps"]:
            raise HTTPException(status_code=400, detail="计划没有可转换的步骤。")
        existing = conn.execute(
            "SELECT id FROM workflow_template WHERE user_id = %s AND created_from_plan_id = %s",
            (user_id, plan_id),
        ).fetchone()
        if existing is not None:
            active_run = conn.execute(
                """
                SELECT id FROM workflow_run
                WHERE user_id = %s AND workflow_id = %s AND status IN ('running', 'paused')
                LIMIT 1
                """,
                (user_id, int(existing[0])),
            ).fetchone()
            if active_run is not None:
                raise HTTPException(status_code=409, detail="来源工作流正在运行或等待确认，暂时不能覆盖模板。")
        row = conn.execute(
            """
            INSERT INTO workflow_template
              (user_id, project_id, name, description, status, trigger_type, created_from_plan_id)
            VALUES (%s, %s, %s, %s, 'active', 'manual', %s)
            ON CONFLICT (user_id, created_from_plan_id) WHERE created_from_plan_id IS NOT NULL
            DO UPDATE SET project_id = EXCLUDED.project_id, name = EXCLUDED.name,
                          description = EXCLUDED.description, status = 'active',
                          trigger_type = 'manual', updated_at = now()
            RETURNING id
            """,
            (
                user_id,
                payload.get("projectId"),
                str(payload["name"])[:256],
                payload.get("description"),
                plan_id,
            ),
        ).fetchone()
        workflow_id = int(row[0])
        conn.execute(
            "DELETE FROM workflow_step_template WHERE user_id = %s AND workflow_id = %s",
            (user_id, workflow_id),
        )
        _insert_steps(conn, user_id, workflow_id, payload["steps"])
        conn.commit()
    return get_workflow(user_id, workflow_id)


def list_workflows(user_id: str, limit: int = 100) -> list[dict[str, Any]]:
    _ensure_postgres()
    with postgres_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, project_id, name, description, status, trigger_type,
                   created_from_plan_id, created_at, updated_at
            FROM workflow_template
            WHERE user_id = %s
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            (user_id, max(1, min(limit, 200))),
        ).fetchall()
    return [_workflow_row(row) for row in rows]


def get_workflow(user_id: str, workflow_id: int) -> dict[str, Any]:
    _ensure_postgres()
    with postgres_connection() as conn:
        row = conn.execute(
            """
            SELECT id, project_id, name, description, status, trigger_type,
                   created_from_plan_id, created_at, updated_at
            FROM workflow_template
            WHERE user_id = %s AND id = %s
            """,
            (user_id, workflow_id),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="工作流不存在。")
        steps = conn.execute(
            """
            SELECT id, workflow_id, sort_order, title, step_type, tool_name,
                   command_template, input_template, requires_confirmation, output_key,
                   condition_expr, created_at
            FROM workflow_step_template
            WHERE user_id = %s AND workflow_id = %s
            ORDER BY sort_order ASC, id ASC
            """,
            (user_id, workflow_id),
        ).fetchall()
    return {**_workflow_row(row), "steps": [_step_row(step) for step in steps]}


def _workflow_row(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "id": row[0],
        "projectId": row[1],
        "name": row[2],
        "description": row[3],
        "status": row[4],
        "triggerType": row[5],
        "createdFromPlanId": row[6],
        "createdAt": row[7].isoformat(),
        "updatedAt": row[8].isoformat(),
    }


def _step_row(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "id": row[0],
        "workflowId": row[1],
        "sortOrder": row[2],
        "title": row[3],
        "stepType": row[4],
        "toolName": row[5],
        "commandTemplate": row[6],
        "inputTemplate": row[7] or {},
        "requiresConfirmation": row[8],
        "outputKey": row[9],
        "condition": row[10] or None,
        "createdAt": row[11].isoformat(),
    }


def _insert_steps(conn: object, user_id: str, workflow_id: int, steps: list[object]) -> None:
    output_keys: set[str] = set()
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        title = str(step.get("title") or "").strip()
        if not title:
            continue
        output_key = _output_key(step.get("outputKey") or step.get("output_key"), index)
        if output_key in output_keys:
            raise HTTPException(status_code=400, detail=f"步骤 outputKey 不能重复：{output_key}")
        output_keys.add(output_key)
        try:
            condition = validate_condition(step.get("condition") or step.get("conditionExpr"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        conn.execute(
            """
            INSERT INTO workflow_step_template
              (user_id, workflow_id, sort_order, title, step_type, tool_name,
               command_template, input_template, requires_confirmation, output_key, condition_expr)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s::jsonb)
            """,
            (
                user_id,
                workflow_id,
                int(step.get("sortOrder") if step.get("sortOrder") is not None else index),
                title[:256],
                str(step.get("stepType") or step.get("step_type") or "summarize"),
                step.get("toolName") or step.get("tool_name"),
                step.get("commandTemplate") or step.get("command_template"),
                json.dumps(step.get("inputTemplate") or step.get("input_template") or {}, ensure_ascii=False),
                bool(step.get("requiresConfirmation", False)),
                output_key,
                json.dumps(condition, ensure_ascii=False) if condition else None,
            ),
        )


def _output_key(value: object, index: int) -> str:
    raw = str(value or f"step_{index + 1}").strip()
    if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]{0,127}", raw):
        raise HTTPException(status_code=400, detail="步骤 outputKey 必须以字母开头，且只能包含字母、数字、下划线或短横线。")
    return raw


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _ensure_postgres() -> None:
    if not settings.postgres_enabled:
        raise HTTPException(status_code=503, detail="PostgreSQL 未启用。")
