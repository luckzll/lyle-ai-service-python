from __future__ import annotations

import hashlib
import shlex
import subprocess
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from app.cli.command_risk import classify_command
from app.cli.command_policy import evaluate_command_policy, stricter_risk_level
from app.cli.output_summarizer import summarize_output
from app.cli.workspace_store import get_workspace, list_workspace_policies
from app.core.config import settings
from app.core.database import postgres_connection
from app.security.execution_audit import write_execution_audit
from app.security.permission_engine import decide_permission


def preview_command(user_id: str, workspace_id: int, command: str) -> dict[str, object]:
    workspace = get_workspace(user_id, workspace_id)
    risk = classify_command(command)
    command_policy = evaluate_command_policy(command, list_workspace_policies(user_id, workspace_id))
    effective_risk = stricter_risk_level(risk.risk_level, command_policy.risk_level)
    decision = decide_permission(
        executor_type="cli",
        executor_name=shlex.split(command)[0] if command.strip() else "",
        operation="run",
        risk_level=effective_risk,
    )
    permission = decision.to_dict()
    if command_policy.requires_confirmation:
        permission["requiresConfirmation"] = True
        permission["reason"] = command_policy.reason
    if not command_policy.allowed or workspace["status"] != "active":
        permission.update({"allowed": False, "requiresConfirmation": False, "requiresSecondConfirmation": False})
        permission["reason"] = command_policy.reason if not command_policy.allowed else "工作区未启用。"
    return {
        "workspace": workspace,
        "command": command,
        "risk": risk.to_dict(),
        "commandPolicy": command_policy.to_dict(),
        "permission": permission,
    }


def run_command(
    user_id: str,
    workspace_id: int,
    command: str,
    *,
    confirmed: bool = False,
    second_confirmed: bool = False,
    timeout_seconds: int = 60,
) -> dict[str, object]:
    if not settings.postgres_enabled:
        raise HTTPException(status_code=503, detail="PostgreSQL 未启用。")
    workspace = get_workspace(user_id, workspace_id)
    if workspace["status"] != "active":
        raise HTTPException(status_code=400, detail="工作区未启用。")
    root = Path(str(workspace["rootPath"])).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail="工作区路径不存在。")
    risk = classify_command(command)
    command_policy = evaluate_command_policy(command, list_workspace_policies(user_id, workspace_id))
    effective_risk = stricter_risk_level(risk.risk_level, command_policy.risk_level)
    decision = decide_permission(
        executor_type="cli",
        executor_name=shlex.split(command)[0] if command.strip() else "",
        operation="run",
        risk_level=effective_risk,
    )
    permission = decision.to_dict()
    if command_policy.requires_confirmation:
        permission["requiresConfirmation"] = True
        permission["reason"] = command_policy.reason
    requires_confirmation = decision.requires_confirmation or command_policy.requires_confirmation
    if not decision.allowed or risk.policy == "deny" or not command_policy.allowed:
        write_execution_audit(
            user_id=user_id,
            source_type="manual",
            executor_type="cli",
            executor_name="command",
            operation="run",
            risk_level=effective_risk,
            status="blocked",
            input_summary=command,
            error_code="COMMAND_BLOCKED",
            error_message=command_policy.reason if not command_policy.allowed else risk.reason,
        )
        return {"ok": False, "blocked": True, "risk": risk.to_dict(), "commandPolicy": command_policy.to_dict(), "permission": permission}
    if requires_confirmation and not confirmed:
        return {"ok": False, "requiresConfirmation": True, "risk": risk.to_dict(), "commandPolicy": command_policy.to_dict(), "permission": permission}
    if decision.requires_second_confirmation and not second_confirmed:
        return {
            "ok": False,
            "requiresConfirmation": True,
            "requiresSecondConfirmation": True,
            "risk": risk.to_dict(),
            "commandPolicy": command_policy.to_dict(),
            "permission": permission,
        }
    command_hash = hashlib.sha256(command.encode("utf-8")).hexdigest()
    with postgres_connection() as conn:
        run_row = conn.execute(
            """
            INSERT INTO cli_command_run
              (user_id, workspace_id, command, command_hash, risk_level, status, started_at)
            VALUES (%s, %s, %s, %s, %s, 'running', now())
            RETURNING id
            """,
            (user_id, workspace_id, command, command_hash, effective_risk),
        ).fetchone()
        conn.commit()
    run_id = int(run_row[0])
    try:
        completed = subprocess.run(
            shlex.split(command),
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=max(1, min(timeout_seconds, 120)),
            check=False,
        )
        summary = summarize_output(completed.stdout, completed.stderr)
        status = "completed" if completed.returncode == 0 else "failed"
        with postgres_connection() as conn:
            conn.execute(
                """
                UPDATE cli_command_run
                SET status = %s, exit_code = %s, stdout_ref = %s, stderr_ref = %s,
                    output_summary = %s, completed_at = now()
                WHERE user_id = %s AND id = %s
                """,
                (status, completed.returncode, completed.stdout[:8000], completed.stderr[:8000], summary, user_id, run_id),
            )
            conn.commit()
        write_execution_audit(
            user_id=user_id,
            source_type="manual",
            executor_type="cli",
            executor_name=shlex.split(command)[0],
            operation="run",
            risk_level=effective_risk,
            status=status,
            input_summary=command,
            output_summary=summary,
            error_code=None if completed.returncode == 0 else "NON_ZERO_EXIT",
        )
        return {
            "ok": completed.returncode == 0,
            "id": run_id,
            "command": command,
            "exitCode": completed.returncode,
            "status": status,
            "summary": summary,
            "risk": risk.to_dict(),
            "commandPolicy": command_policy.to_dict(),
        }
    except subprocess.TimeoutExpired as exc:
        summary = summarize_output(exc.stdout or "", exc.stderr or "命令执行超时。")
        with postgres_connection() as conn:
            conn.execute(
                """
                UPDATE cli_command_run
                SET status = 'failed', output_summary = %s, completed_at = now()
                WHERE user_id = %s AND id = %s
                """,
                (summary, user_id, run_id),
            )
            conn.commit()
        write_execution_audit(
            user_id=user_id,
            source_type="manual",
            executor_type="cli",
            executor_name="command",
            operation="run",
            risk_level=effective_risk,
            status="failed",
            input_summary=command,
            output_summary=summary,
            error_code="TIMEOUT",
            error_message="命令执行超时。",
        )
        return {"ok": False, "id": run_id, "status": "failed", "errorCode": "TIMEOUT", "summary": summary}


def list_command_runs(user_id: str, workspace_id: int, limit: int = 50) -> list[dict[str, Any]]:
    if not settings.postgres_enabled:
        return []
    limit = max(1, min(limit, 200))
    with postgres_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, command, risk_level, status, exit_code, output_summary, started_at, completed_at, created_at
            FROM cli_command_run
            WHERE user_id = %s AND workspace_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (user_id, workspace_id, limit),
        ).fetchall()
    return [
        {
            "id": row[0],
            "command": row[1],
            "riskLevel": row[2],
            "status": row[3],
            "exitCode": row[4],
            "summary": row[5],
            "startedAt": row[6].isoformat() if row[6] else None,
            "completedAt": row[7].isoformat() if row[7] else None,
            "createdAt": row[8].isoformat(),
        }
        for row in rows
    ]
