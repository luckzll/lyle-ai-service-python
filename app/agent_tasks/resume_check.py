from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.agent_tasks.schemas import ResumeCheckRequest, ResumeCheckResponse
from app.cli.workspace_store import get_workspace
from app.projects.project_store import get_project_overview


def check_resume_state(request: ResumeCheckRequest) -> ResumeCheckResponse:
    workspace_ids = request.permissionSnapshot.get("workspaceIds")
    project_ids = request.permissionSnapshot.get("projectIds")
    workspace_ids = workspace_ids if isinstance(workspace_ids, list) else []
    project_ids = project_ids if isinstance(project_ids, list) else []
    if not workspace_ids and not project_ids:
        return ResumeCheckResponse(
            valid=True,
            changed=False,
            reason="当前计划没有工作区外部状态需要复核。",
            currentExternalState={},
        )

    current: dict[str, str] = {}
    try:
        for raw in workspace_ids[:10]:
            workspace_id = int(raw)
            current[f"workspace:{workspace_id}"] = compute_workspace_state(request.userId, workspace_id)
        for raw in project_ids[:10]:
            project_id = int(raw)
            current[f"project:{project_id}"] = compute_project_state(request.userId, project_id)
    except Exception as exc:
        return ResumeCheckResponse(
            valid=False,
            changed=True,
            reason=f"恢复前无法验证授权项目或工作区：{exc}",
            currentExternalState=current,
        )

    expected = request.expectedExternalState
    changed_keys = [key for key, value in expected.items() if key in current and current[key] != value]
    return ResumeCheckResponse(
        valid=True,
        changed=bool(changed_keys),
        reason=(
            f"检测到外部状态变化：{', '.join(changed_keys)}。继续前需要用户选择重跑或调整计划。"
            if changed_keys else
            "授权项目和工作区仍可访问，关键外部状态未变化。"
        ),
        currentExternalState=current,
    )


def compute_workspace_state(user_id: str, workspace_id: int) -> str:
    workspace = get_workspace(user_id, workspace_id)
    if str(workspace.get("status") or "") != "active":
        raise ValueError(f"工作区 {workspace_id} 未启用")
    root = Path(str(workspace["rootPath"])).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"工作区 {workspace_id} 路径不存在")

    parts = [str(root), str(workspace.get("updatedAt") or "")]
    files_seen = 0
    for path in sorted(root.rglob("*")):
        if files_seen >= 500:
            break
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in {".git", "node_modules", ".venv", ".venv312", "dist", "target", "__pycache__"} for part in relative.parts):
            continue
        stat = path.stat()
        parts.append(f"{relative.as_posix()}:{stat.st_size}:{stat.st_mtime_ns}")
        files_seen += 1
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def project_state_hash(overview: dict[str, Any]) -> str:
    payload = json.dumps(overview, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_project_state(user_id: str, project_id: int) -> str:
    return project_state_hash(get_project_overview(user_id, project_id))
