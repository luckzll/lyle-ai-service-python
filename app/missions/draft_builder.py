from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from app.cli.workspace_store import get_workspace
from app.core.config import settings
from app.missions.schemas import MissionDraftRequest, MissionDraftResponse


PROMPT_VERSION = "lyle-mission-v6-draft-1"


def build_mission_draft(request: MissionDraftRequest) -> MissionDraftResponse:
    template_id = request.templateId or _infer_template(request.goal)
    _validate_scope(template_id, request)
    completion = request.completionCriteria or _default_criteria(template_id)
    constraints: dict[str, Any] = {
        "forbidden": request.forbidden,
        "missionTemplateId": template_id,
    }
    if request.workspaceId:
        constraints["workspaceId"] = request.workspaceId
    if request.writeTarget:
        constraints["writeTarget"] = request.writeTarget
    if request.autoMaintainTasks:
        constraints["taskTitle"] = f"跟进：{request.goal[:120]}"
    if template_id == "release-readiness":
        constraints["maxParallelReadBranches"] = 3
        constraints["parallelJoinPolicy"] = "ALL"

    rules = _rules(template_id, request)
    validity = min(request.validityDays, request.maxValidityDays, 30)
    trigger = _trigger(template_id, request)
    title = (request.title or _title(template_id)).strip()[:255]
    return MissionDraftResponse(
        title=title,
        goal=request.goal.strip(),
        templateId=template_id,
        projectId=request.projectId,
        completionCriteria=completion,
        constraints=constraints,
        grantSummary=_grant_summary(template_id, request),
        grantRules=rules,
        budgetPolicy=_period_budget(template_id),
        singleRunBudget=_single_run_budget(template_id),
        periodBudget=_period_budget(template_id),
        periodType="WEEKLY",
        validityDays=validity,
        notificationPolicy={
            "normalResults": "DIGEST",
            "exceptions": "IMMEDIATE",
            "expiryReminderDays": [7, 1],
        },
        quietHours=request.quietHours or {"start": "22:00", "end": "08:00"},
        triggers=[trigger],
        promptVersion=PROMPT_VERSION,
    )


def _infer_template(goal: str) -> str:
    normalized = goal.lower()
    if any(word in normalized for word in ("发布", "release", "go/no-go", "上线")):
        return "release-readiness"
    if any(word in normalized for word in ("调研", "追踪", "周报", "research", "资料")):
        return "research-weekly"
    return "project-health"


def _validate_scope(template_id: str, request: MissionDraftRequest) -> None:
    if template_id in {"project-health", "release-readiness"} and request.projectId is None:
        raise ValueError("项目健康守护和发布准备复查必须绑定 projectId。")
    if template_id in {"project-health", "release-readiness"} and request.workspaceId is None:
        raise ValueError("项目健康守护和发布准备复查必须绑定 workspaceId。")
    if request.allowFilePatch and (request.workspaceId is None or not request.writeTarget):
        raise ValueError("自动文件 Patch 必须同时提供 workspaceId 和 writeTarget。")


def validate_mission_workspace(request: MissionDraftRequest) -> None:
    """Validate live workspace state at the HTTP boundary.

    The pure draft builder remains deterministic for shadow previews and unit
    tests; production draft requests fail before a Mission can be persisted.
    """
    template_id = request.templateId or _infer_template(request.goal)
    if (
        template_id in {"project-health", "release-readiness"}
        and request.workspaceId is not None
        and settings.postgres_enabled
    ):
        _validate_git_workspace(request.userId, request.workspaceId)


def _validate_git_workspace(user_id: str, workspace_id: int) -> None:
    try:
        workspace = get_workspace(user_id, workspace_id)
    except HTTPException as exc:
        raise ValueError(str(exc.detail)) from exc
    if str(workspace.get("status") or "").lower() != "active":
        raise ValueError("授权工作区当前不可用，不能创建无人值守 Mission。")
    root = Path(str(workspace.get("rootPath") or "")).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("授权工作区根目录不存在，不能创建无人值守 Mission。")
    try:
        check = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("无法验证授权工作区的 Git 状态，不能创建无人值守 Mission。") from exc
    if check.returncode != 0 or check.stdout.strip() != "true":
        raise ValueError("授权工作区根目录不是 Git 工作区；请绑定正确仓库后再启用持续运行。")


def _rules(template_id: str, request: MissionDraftRequest) -> list[dict[str, Any]]:
    project_scope = {"projectIds": [request.projectId]} if request.projectId else {}
    workspace_scope = {"workspaceIds": [request.workspaceId]} if request.workspaceId else {}
    rules: list[dict[str, Any]] = []
    read_tools = {
        "project-health": [
            ("project.inspect", project_scope),
            ("workspace.git_status", workspace_scope),
            ("workspace.docs_check", workspace_scope),
            ("release.check", workspace_scope),
            ("analysis.synthesize", {}),
        ],
        "research-weekly": [
            ("research.search", {}),
            ("analysis.synthesize", {}),
        ],
        "release-readiness": [
            ("project.inspect", project_scope),
            ("workspace.git_status", workspace_scope),
            ("workspace.docs_check", workspace_scope),
            ("release.check", workspace_scope),
            ("analysis.synthesize", {}),
        ],
    }[template_id]
    for index, (tool_ref, scope) in enumerate(read_tools, start=1):
        rules.append({
            "id": f"rule-r0-{index}-{tool_ref.replace('.', '-')}",
            "effect": "allow",
            "toolRefs": [tool_ref],
            "operations": ["search" if tool_ref == "research.search" else "analyze" if tool_ref == "analysis.synthesize" else "read"],
            "resourceScope": scope,
            "allowedFields": [],
            "maxRisk": "R0",
            "requiresReversible": False,
            "maxCallsPerRun": 20,
        })
    if request.autoMaintainTasks and request.projectId:
        rules.append({
            "id": "rule-r1-project-task-create",
            "effect": "allow",
            "toolRefs": ["todo.propose"],
            "operations": ["create"],
            "resourceScope": {"projectIds": [request.projectId]},
            "allowedFields": ["title", "dueAt", "items"],
            "maxRisk": "R1",
            "requiresReversible": True,
            "maxCallsPerRun": 3,
        })
    if request.allowFilePatch and request.workspaceId:
        rules.append({
            "id": "rule-r1-workspace-file-patch",
            "effect": "allow",
            "toolRefs": ["file.patch.propose"],
            "operations": ["patch"],
            "resourceScope": {"workspaceIds": [request.workspaceId]},
            "allowedFields": ["relativePath", "newContent", "expectedSha256"],
            "maxRisk": "R1",
            "requiresReversible": True,
            "maxCallsPerRun": 1,
        })
    return rules


def _trigger(template_id: str, request: MissionDraftRequest) -> dict[str, Any]:
    defaults = {
        "project-health": {"frequency": "WEEKDAYS", "time": "09:00"},
        "research-weekly": {"frequency": "WEEKLY", "dayOfWeek": "FRIDAY", "time": "16:00"},
        "release-readiness": {"frequency": "DAILY", "time": "10:00"},
    }
    return {
        "type": str(request.schedule.get("type") or "FIXED_TIME"),
        "config": request.schedule.get("config") if isinstance(request.schedule.get("config"), dict) else defaults[template_id],
        "timezone": request.timezone,
        "misfirePolicy": str(request.schedule.get("misfirePolicy") or "RUN_ONCE"),
    }


def _single_run_budget(template_id: str) -> dict[str, int]:
    steps = 8 if template_id == "release-readiness" else 6
    return {
        "maxSteps": steps,
        "maxToolCalls": steps * 2,
        "maxModelTokens": 80_000,
        "maxCostMicros": 2_000_000,
        "maxWallTimeMillis": 900_000,
        "maxRetriesPerStep": 2,
        "maxConsecutiveNoProgress": 2,
    }


def _period_budget(template_id: str) -> dict[str, int]:
    multiplier = 5 if template_id == "project-health" else 2
    single = _single_run_budget(template_id)
    return {
        "maxToolCalls": single["maxToolCalls"] * multiplier,
        "maxModelTokens": single["maxModelTokens"] * multiplier,
        "maxCostMicros": single["maxCostMicros"] * multiplier,
        "maxWallTimeMillis": single["maxWallTimeMillis"] * multiplier,
    }


def _default_criteria(template_id: str) -> list[str]:
    return {
        "project-health": ["项目状态与测试证据已检查", "任务和文档缺口已列出", "异常与建议可追溯"],
        "research-weekly": ["指定主题已检索", "新增资料已去重并保留来源", "形成周期报告与冲突说明"],
        "release-readiness": ["项目、Git、文档与测试已复查", "发布阻塞项有证据", "形成 Go/No-Go 建议"],
    }[template_id]


def _title(template_id: str) -> str:
    return {
        "project-health": "项目健康守护",
        "research-weekly": "持续调研周报",
        "release-readiness": "发布准备复查",
    }[template_id]


def _grant_summary(template_id: str, request: MissionDraftRequest) -> str:
    write = "；可自动维护项目内部任务" if request.autoMaintainTasks else ""
    patch = "；可在指定工作区应用可撤销文件 Patch" if request.allowFilePatch else ""
    return f"仅在模板「{_title(template_id)}」声明的项目、工作区、工具、预算和有效期内执行 R0 动作{write}{patch}；R2/R3 始终禁止。"
