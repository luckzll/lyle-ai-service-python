from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from app.core.config import settings
from app.core.database import postgres_connection
from app.experience.template_catalog import get_template, workflow_payload
from app.experience.events import record_event
from app.workflows.workflow_runner import run_workflow
from app.workflows.workflow_store import create_workflow


def get_onboarding(user_id: str) -> dict[str, Any]:
    _ensure()
    with postgres_connection() as conn:
        inserted = conn.execute(
            "INSERT INTO experience_onboarding_state(user_id) VALUES (%s) ON CONFLICT DO NOTHING RETURNING user_id",
            (user_id,),
        ).fetchone()
        conn.commit()
        row = conn.execute(
            """SELECT first_opened_at, completed_at, skipped_at, selected_project_id,
               selected_workspace_id, first_run_id, first_success_at, last_step, updated_at
               FROM experience_onboarding_state WHERE user_id=%s""", (user_id,),
        ).fetchone()
        projects = conn.execute("SELECT id, name FROM project WHERE user_id=%s AND status='active' ORDER BY updated_at DESC LIMIT 20", (user_id,)).fetchall()
        workspaces = conn.execute("SELECT id, name, root_path FROM workspace WHERE user_id=%s AND status='active' ORDER BY updated_at DESC LIMIT 20", (user_id,)).fetchall()
    response = {
        "firstOpenedAt": row[0].isoformat(), "completedAt": row[1].isoformat() if row[1] else None,
        "skippedAt": row[2].isoformat() if row[2] else None, "selectedProjectId": row[3],
        "selectedWorkspaceId": row[4], "firstRunId": row[5], "firstSuccessAt": row[6].isoformat() if row[6] else None,
        "lastStep": row[7], "updatedAt": row[8].isoformat(),
        "candidates": {"projects": [{"id": x[0], "name": x[1]} for x in projects], "workspaces": [{"id": x[0], "name": x[1], "scope": "已授权本地工作区"} for x in workspaces]},
    }
    if inserted is not None:
        record_event(user_id, "onboarding_started")
    return response


def update_onboarding(user_id: str, request: dict[str, Any]) -> dict[str, Any]:
    get_onboarding(user_id)
    action = str(request.get("action") or "progress")
    if action not in {"progress", "complete", "skip", "reopen"}:
        raise HTTPException(status_code=400, detail="引导操作不合法。")
    with postgres_connection() as conn:
        conn.execute(
            """UPDATE experience_onboarding_state SET selected_project_id=COALESCE(%s, selected_project_id),
               selected_workspace_id=COALESCE(%s, selected_workspace_id), last_step=%s,
               completed_at=CASE WHEN %s='complete' THEN now() WHEN %s='reopen' THEN NULL ELSE completed_at END,
               skipped_at=CASE WHEN %s='skip' THEN now() WHEN %s='reopen' THEN NULL ELSE skipped_at END,
               updated_at=now() WHERE user_id=%s""",
            (request.get("projectId"), request.get("workspaceId"), str(request.get("lastStep") or action), action, action, action, action, user_id),
        )
        conn.commit()
    if action == "skip":
        record_event(user_id, "onboarding_skipped")
    return get_onboarding(user_id)


def run_first_inspection(user_id: str, request: dict[str, Any]) -> dict[str, Any]:
    state = update_onboarding(user_id, {**request, "action": "progress", "lastStep": "running"})
    template = get_template("weekly_project_review")
    record_event(user_id, "first_run_started", template_id=template["id"], metadata={"projectSelected": state["selectedProjectId"] is not None, "workspaceSelected": state["selectedWorkspaceId"] is not None})
    workflow = create_workflow(user_id, workflow_payload(template, title="首次项目巡检", project_id=state["selectedProjectId"]))
    run = run_workflow(user_id, workflow["id"], {"workspaceId": state["selectedWorkspaceId"], "triggerType": "manual", "templateId": template["id"]})
    with postgres_connection() as conn:
        conn.execute(
            """UPDATE experience_onboarding_state SET first_run_id=%s,
               first_success_at=CASE WHEN %s='completed' THEN now() ELSE first_success_at END,
               completed_at=CASE WHEN %s='completed' THEN now() ELSE completed_at END,
               last_step=%s, updated_at=now() WHERE user_id=%s""",
            (run["id"], run["status"], run["status"], "completed" if run["status"] == "completed" else "result", user_id),
        )
        conn.commit()
    record_event(user_id, "first_run_completed", template_id=template["id"], status=str(run["status"]))
    return {"run": run, "onboarding": get_onboarding(user_id), "permissionSummary": "只读取所选项目与已授权工作区，不会修改文件或外部系统。"}


def _ensure() -> None:
    if not settings.postgres_enabled:
        raise HTTPException(status_code=503, detail="PostgreSQL 未启用。")
