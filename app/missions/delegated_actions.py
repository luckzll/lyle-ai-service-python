from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.agent_tasks.change_preview import _relative_path, apply_workspace_file_patch
from app.agent_tasks.schemas import ExecuteStepRequest, ToolCriterionEvidence, ToolResult
from app.cli.workspace_store import get_workspace
from app.core.config import settings
from app.core.database import postgres_connection
from app.missions.schemas import CompensationRequest, CompensationResponse


_FORBIDDEN_PATH_PARTS = {
    ".git", ".env", ".ssh", "node_modules", "__pycache__", ".venv",
    "target", "dist", "build",
}
_FORBIDDEN_NAMES = {
    "credentials", "credentials.json", "id_rsa", "id_ed25519",
    ".npmrc", ".pypirc",
}


def execute_delegated_action(request: ExecuteStepRequest) -> ToolResult:
    if not request.delegatedWriteAuthorized or not request.authorizationDecisionId:
        return _error("持续委托写操作缺少 Java 授权决策。", "DELEGATION_DECISION_REQUIRED")
    if not settings.postgres_enabled:
        return _error("PostgreSQL 未启用，无法安全保存补偿快照。", "COMPENSATION_STORE_UNAVAILABLE")
    if request.toolRef == "todo.propose":
        return _create_todo(request)
    if request.toolRef == "file.patch.propose":
        return _apply_file_patch(request)
    return _error("该工具不在 V6 R1 可补偿白名单中。", "DELEGATED_TOOL_NOT_SUPPORTED")


def compensate_action(request: CompensationRequest) -> CompensationResponse:
    if not settings.postgres_enabled:
        return CompensationResponse(
            status="FAILED",
            summary="PostgreSQL 未启用，无法读取补偿快照。",
        )
    _ensure_schema()
    now = datetime.now(UTC)
    with postgres_connection() as conn:
        row = conn.execute(
            """
            SELECT snapshot_ref, user_id, tool_ref, payload, status, expires_at,
                   compensation_key
            FROM agent_delegated_compensation
            WHERE snapshot_ref = %s AND user_id = %s
            FOR UPDATE
            """,
            (request.snapshotRef, request.userId),
        ).fetchone()
        if row is None:
            conn.commit()
            return CompensationResponse(status="FAILED", summary="补偿快照不存在或不属于当前用户。")
        payload = row[3] if isinstance(row[3], dict) else json.loads(row[3] or "{}")
        status = str(row[4] or "")
        if status == "SUCCEEDED":
            conn.commit()
            return CompensationResponse(
                status="SUCCEEDED",
                summary="该自动变更已经撤销，本次没有重复执行。",
                evidenceRefs=[request.snapshotRef],
            )
        if row[5] <= now:
            conn.execute(
                """
                UPDATE agent_delegated_compensation
                SET status='EXPIRED', updated_at=%s WHERE snapshot_ref=%s
                """,
                (now, request.snapshotRef),
            )
            conn.commit()
            return CompensationResponse(status="EXPIRED", summary="该自动变更已超过撤销期限。")
        if status not in {"AVAILABLE", "REQUESTED", "PREPARED"}:
            conn.commit()
            return CompensationResponse(status="FAILED", summary=f"补偿快照当前不可执行：{status}。")

        tool_ref = str(row[2] or "")
        outcome = (
            _compensate_todo(conn, request.userId, payload)
            if tool_ref == "todo.propose"
            else _compensate_file(request.userId, payload)
            if tool_ref == "file.patch.propose"
            else ("FAILED", "不支持该工具的补偿。", [])
        )
        next_status, summary, evidence = outcome
        conn.execute(
            """
            UPDATE agent_delegated_compensation
            SET status=%s, compensation_key=%s, updated_at=%s,
                compensated_at=CASE WHEN %s='SUCCEEDED' THEN %s ELSE compensated_at END
            WHERE snapshot_ref=%s
            """,
            (next_status, request.idempotencyKey, now, next_status, now, request.snapshotRef),
        )
        conn.commit()
    return CompensationResponse(
        status=next_status,  # type: ignore[arg-type]
        summary=summary,
        evidenceRefs=[request.snapshotRef, *evidence],
    )


def _create_todo(request: ExecuteStepRequest) -> ToolResult:
    project_id = _positive_int(request.input.get("projectId") or request.projectId)
    if project_id is None:
        return _error("项目任务必须声明 projectId。", "PROJECT_REQUIRED")
    raw_items = request.input.get("items") if isinstance(request.input.get("items"), list) else []
    if len(raw_items) > 1:
        return _error(
            "V6 首版不自动执行批量任务创建；请拆成单个可补偿动作。",
            "DELEGATED_BATCH_WRITE_NOT_SUPPORTED",
        )
    item = raw_items[0] if raw_items and isinstance(raw_items[0], dict) else {}
    title = str(item.get("title") or request.input.get("title") or request.goal).strip()[:200]
    if not title:
        return _error("任务标题不能为空。", "PARAM_MISSING")
    due_at = _parse_datetime(item.get("dueAt") or request.input.get("dueAt"))
    snapshot_ref = _snapshot_ref()
    now = datetime.now(UTC)
    expires_at = now + timedelta(days=7)
    _ensure_schema()
    with postgres_connection() as conn:
        duplicate = conn.execute(
            """
            SELECT id, updated_at FROM agent_todo
            WHERE user_id=%s AND project_id=%s
              AND status='open' AND lower(title)=lower(%s)
            ORDER BY updated_at DESC LIMIT 1
            """,
            (request.userId, project_id, title),
        ).fetchone()
        if duplicate:
            conn.commit()
            ref = f"todo://{duplicate[0]}"
            return ToolResult(
                status="SUCCESS",
                summary=f"已存在相同项目任务「{title}」，本次没有重复创建。",
                dataRef=f"tool-result://{request.stepRunId}",
                evidenceRefs=[ref],
                sideEffect="NONE",
                criterionEvidence=_write_criterion_evidence(request, [ref]),
            )
        row = conn.execute(
            """
            INSERT INTO agent_todo
              (user_id, conversation_id, project_id, title, status, due_at, created_at, updated_at)
            VALUES (%s, %s, %s, %s, 'open', %s, %s, %s)
            RETURNING id, updated_at
            """,
            (
                request.userId,
                request.conversationId or "",
                project_id,
                title,
                due_at,
                now,
                now,
            ),
        ).fetchone()
        payload = {
            "todoId": int(row[0]),
            "projectId": project_id,
            "createdUpdatedAt": row[1].isoformat(),
            "titleHash": hashlib.sha256(title.encode("utf-8")).hexdigest(),
        }
        _insert_snapshot(
            conn,
            snapshot_ref=snapshot_ref,
            user_id=request.userId,
            tool_ref=request.toolRef,
            resource_ref=f"todo://{row[0]}",
            payload=payload,
            expires_at=expires_at,
            status="AVAILABLE",
        )
        conn.commit()
    ref = f"todo://{row[0]}"
    return ToolResult(
        status="SUCCESS",
        summary=f"已在授权边界内创建项目任务「{title}」。",
        dataRef=f"tool-result://{request.stepRunId}",
        evidenceRefs=[ref],
        sideEffect="APPLIED",
        criterionEvidence=_write_criterion_evidence(request, [ref]),
        compensation={
            "operation": "archive_created_todo",
            "snapshotRef": snapshot_ref,
            "resourceRef": ref,
            "expiresAt": expires_at.isoformat(),
            "precondition": {"updatedAt": row[1].isoformat()},
        },
    )


def _apply_file_patch(request: ExecuteStepRequest) -> ToolResult:
    workspace_id = _positive_int(request.input.get("workspaceId"))
    if workspace_id is None:
        return _error("文件 Patch 必须声明 workspaceId。", "WORKSPACE_REQUIRED")
    try:
        relative_path = _relative_path(request.input.get("relativePath"))
        _assert_patch_path(relative_path)
        workspace = get_workspace(request.userId, workspace_id)
        root = Path(str(workspace["rootPath"])).expanduser().resolve()
        target = _resolved_workspace_target(root, relative_path)
    except ValueError as exc:
        return _error(str(exc), "PATH_NOT_ALLOWED")
    before_content = target.read_text(encoding="utf-8") if target.exists() else ""
    before_sha = hashlib.sha256(before_content.encode("utf-8")).hexdigest()
    requested_sha = str(request.input.get("expectedSha256") or before_sha)
    if requested_sha != before_sha:
        return _error("文件已变化，自动 Patch 未执行。", "RESOURCE_VERSION_CHANGED")
    new_content = str(request.input.get("newContent") or "")
    if len(new_content.encode("utf-8")) > 1_000_000:
        return _error("Patch 结果超过 1MB 安全上限。", "PATCH_TOO_LARGE")

    snapshot_ref = _snapshot_ref()
    expires_at = datetime.now(UTC) + timedelta(days=7)
    payload = {
        "workspaceId": workspace_id,
        "relativePath": relative_path,
        "beforeContent": before_content,
        "beforeSha256": before_sha,
        "afterSha256": hashlib.sha256(new_content.encode("utf-8")).hexdigest(),
    }
    _ensure_schema()
    with postgres_connection() as conn:
        _insert_snapshot(
            conn,
            snapshot_ref=snapshot_ref,
            user_id=request.userId,
            tool_ref=request.toolRef,
            resource_ref=f"workspace://{workspace_id}/{relative_path}",
            payload=payload,
            expires_at=expires_at,
            status="PREPARED",
        )
        conn.commit()

    result = apply_workspace_file_patch(request.userId, {
        "workspaceId": workspace_id,
        "relativePath": relative_path,
        "expectedSha256": before_sha,
        "newContent": new_content,
    })
    if not result.get("ok"):
        _update_snapshot_status(snapshot_ref, "FAILED")
        return _error(str(result.get("content") or "文件 Patch 执行失败。"), str(result.get("errorCode") or "PATCH_FAILED"))
    _update_snapshot_status(snapshot_ref, "AVAILABLE")
    ref = f"workspace://{workspace_id}/{relative_path}"
    return ToolResult(
        status="SUCCESS",
        summary=f"已在授权工作区应用可撤销文件 Patch：{relative_path}。",
        dataRef=f"tool-result://{request.stepRunId}",
        evidenceRefs=[ref],
        sideEffect="APPLIED",
        criterionEvidence=_write_criterion_evidence(request, [ref]),
        externalStateHashes={f"workspace-file:{workspace_id}:{relative_path}": payload["afterSha256"]},
        compensation={
            "operation": "restore_workspace_file",
            "snapshotRef": snapshot_ref,
            "resourceRef": ref,
            "expiresAt": expires_at.isoformat(),
            "precondition": {"sha256": payload["afterSha256"]},
        },
    )


def _compensate_todo(conn, user_id: str, payload: dict[str, Any]) -> tuple[str, str, list[str]]:
    todo_id = _positive_int(payload.get("todoId"))
    if todo_id is None:
        return "FAILED", "补偿快照缺少 todoId。", []
    row = conn.execute(
        """
        SELECT status, updated_at, project_id FROM agent_todo
        WHERE user_id=%s AND id=%s
        FOR UPDATE
        """,
        (user_id, todo_id),
    ).fetchone()
    if row is None:
        return "CONFLICTED", "项目任务已不存在，未覆盖外部状态。", [f"todo://{todo_id}"]
    expected = str(payload.get("createdUpdatedAt") or "")
    expected_project_id = _positive_int(payload.get("projectId"))
    if (
        row[1].isoformat() != expected
        or str(row[0]) != "open"
        or expected_project_id is None
        or int(row[2] or 0) != expected_project_id
    ):
        return "CONFLICTED", "项目任务已被继续修改，未自动归档。", [f"todo://{todo_id}"]
    conn.execute(
        """
        UPDATE agent_todo SET status='archived', updated_at=%s
        WHERE user_id=%s AND id=%s
        """,
        (datetime.now(UTC), user_id, todo_id),
    )
    return "SUCCEEDED", "已软归档由持续任务创建的项目任务。", [f"todo://{todo_id}"]


def _compensate_file(user_id: str, payload: dict[str, Any]) -> tuple[str, str, list[str]]:
    workspace_id = _positive_int(payload.get("workspaceId"))
    relative_path = str(payload.get("relativePath") or "")
    if workspace_id is None or not relative_path:
        return "FAILED", "补偿快照缺少工作区文件信息。", []
    try:
        _assert_patch_path(relative_path)
        workspace = get_workspace(user_id, workspace_id)
        root = Path(str(workspace["rootPath"])).expanduser().resolve()
        target = _resolved_workspace_target(root, relative_path)
    except ValueError as exc:
        return "FAILED", str(exc), []
    current = target.read_text(encoding="utf-8") if target.exists() else ""
    current_sha = hashlib.sha256(current.encode("utf-8")).hexdigest()
    if current_sha != str(payload.get("afterSha256") or ""):
        return "CONFLICTED", "文件在自动 Patch 后已继续变化，未覆盖当前内容。", [
            f"workspace://{workspace_id}/{relative_path}"
        ]
    before_content = str(payload.get("beforeContent") or "")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(before_content, encoding="utf-8")
    return "SUCCEEDED", f"已恢复文件 {relative_path} 的修改前内容。", [
        f"workspace://{workspace_id}/{relative_path}"
    ]


def _ensure_schema() -> None:
    with postgres_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_delegated_compensation (
              snapshot_ref VARCHAR(512) PRIMARY KEY,
              user_id VARCHAR(128) NOT NULL,
              tool_ref VARCHAR(255) NOT NULL,
              resource_ref VARCHAR(512) NOT NULL,
              payload JSONB NOT NULL,
              status VARCHAR(32) NOT NULL,
              expires_at TIMESTAMPTZ NOT NULL,
              compensation_key VARCHAR(128),
              created_at TIMESTAMPTZ NOT NULL,
              updated_at TIMESTAMPTZ NOT NULL,
              compensated_at TIMESTAMPTZ
            )
            """
        )
        conn.execute("ALTER TABLE agent_todo ADD COLUMN IF NOT EXISTS project_id BIGINT")
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_agent_todo_user_project_status
            ON agent_todo(user_id, project_id, status, updated_at DESC)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_agent_delegated_compensation_user_status
            ON agent_delegated_compensation(user_id, status, expires_at)
            """
        )
        conn.commit()


def _insert_snapshot(
    conn,
    *,
    snapshot_ref: str,
    user_id: str,
    tool_ref: str,
    resource_ref: str,
    payload: dict[str, Any],
    expires_at: datetime,
    status: str,
) -> None:
    now = datetime.now(UTC)
    conn.execute(
        """
        INSERT INTO agent_delegated_compensation
          (snapshot_ref, user_id, tool_ref, resource_ref, payload, status,
           expires_at, compensation_key, created_at, updated_at, compensated_at)
        VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, NULL, %s, %s, NULL)
        ON CONFLICT (snapshot_ref) DO NOTHING
        """,
        (
            snapshot_ref,
            user_id,
            tool_ref,
            resource_ref,
            json.dumps(payload, ensure_ascii=False),
            status,
            expires_at,
            now,
            now,
        ),
    )


def _update_snapshot_status(snapshot_ref: str, status: str) -> None:
    with postgres_connection() as conn:
        conn.execute(
            """
            UPDATE agent_delegated_compensation
            SET status=%s, updated_at=%s WHERE snapshot_ref=%s
            """,
            (status, datetime.now(UTC), snapshot_ref),
        )
        conn.commit()


def _write_criterion_evidence(
    request: ExecuteStepRequest,
    refs: list[str],
) -> list[ToolCriterionEvidence]:
    result: list[ToolCriterionEvidence] = []
    for raw in request.completionCriteria:
        criterion = _criterion_text(raw)
        normalized = criterion.lower()
        if request.toolRef == "todo.propose" and any(word in normalized for word in ("任务", "待办", "创建")):
            result.append(ToolCriterionEvidence(
                criterion=criterion,
                status="COMPLETE",
                evidenceRefs=refs,
                note="项目内部任务已在授权边界内创建。",
            ))
        if request.toolRef == "file.patch.propose" and any(word in normalized for word in ("文件", "文档", "patch", "修改")):
            result.append(ToolCriterionEvidence(
                criterion=criterion,
                status="COMPLETE",
                evidenceRefs=refs,
                note="工作区文件 Patch 已应用并保存补偿快照。",
            ))
    return result[:20]


def _criterion_text(value: Any) -> str:
    if isinstance(value, str):
        return value[:1000]
    if isinstance(value, dict):
        for key in ("criterion", "title", "description", "text"):
            if value.get(key):
                return str(value[key])[:1000]
    return str(value)[:1000]


def _error(summary: str, code: str) -> ToolResult:
    return ToolResult(
        status="ERROR",
        summary=summary[:8000],
        retryable=False,
        errorCode=code[:80],
    )


def _assert_patch_path(relative_path: str) -> None:
    path = Path(relative_path)
    lowered = {part.lower() for part in path.parts}
    if lowered & _FORBIDDEN_PATH_PARTS:
        raise ValueError("该路径属于 Secret、版本控制或依赖缓存目录，禁止自动修改。")
    if path.name.lower() in _FORBIDDEN_NAMES or path.name.lower().startswith(".env"):
        raise ValueError("该文件可能包含凭据或 Secret，禁止自动修改。")
    allowed_suffixes = {".md", ".txt", ".json", ".yml", ".yaml", ".toml", ".py", ".java", ".ts", ".vue", ".css"}
    if path.suffix.lower() not in allowed_suffixes:
        raise ValueError("该文件类型不在 V6 自动 Patch 白名单中。")


def _resolved_workspace_target(root: Path, relative_path: str) -> Path:
    target = (root / relative_path).resolve()
    if root not in target.parents and target != root:
        raise ValueError("文件路径超出已授权工作区。")
    resolved_relative = target.relative_to(root).as_posix()
    _assert_patch_path(resolved_relative)
    return target


def _snapshot_ref() -> str:
    return f"delegated-compensation://{uuid.uuid4()}"


def _parse_datetime(value: object) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
