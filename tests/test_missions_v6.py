import subprocess

import pytest
from pydantic import ValidationError

from app.agent_tasks.schemas import ExecuteStepRequest
from app.core.config import settings
from app.missions.delegated_actions import _resolved_workspace_target, execute_delegated_action
from app.missions.draft_builder import build_mission_draft, validate_mission_workspace
from app.missions.result_digest import build_mission_digest
from app.missions.schemas import (
    MissionDigestRequest,
    MissionDraftRequest,
    ShadowScenarioRequest,
)
from app.missions.shadow_scenarios import build_shadow_scenarios


def _draft(**overrides):
    value = {
        "userId": "7",
        "goal": "每个工作日上午检查项目健康状态并保留证据",
        "templateId": "project-health",
        "projectId": 9,
        "workspaceId": 47,
        "autoMaintainTasks": True,
        "validityDays": 30,
    }
    value.update(overrides)
    return MissionDraftRequest.model_validate(value)


def _step(**overrides):
    value = {
        "taskId": "1",
        "taskRunId": "2",
        "stepRunId": "3",
        "userId": "7",
        "projectId": 9,
        "conversationId": None,
        "goal": "巡检后创建单个项目任务",
        "completionCriteria": ["创建跟进任务"],
        "toolRef": "todo.propose",
        "input": {"projectId": 9, "title": "跟进巡检异常"},
        "artifacts": [],
        "idempotencyKey": "a" * 64,
        "remainingBudget": {
            "steps": 2,
            "toolCalls": 2,
            "modelTokens": 1000,
            "costMicros": 1000,
            "wallTimeMillis": 1000,
        },
    }
    value.update(overrides)
    return ExecuteStepRequest.model_validate(value)


def test_draft_is_structured_time_bounded_and_r1_is_specific() -> None:
    draft = build_mission_draft(_draft())

    assert draft.validityDays == 30
    assert draft.singleRunBudget["maxSteps"] == 6
    assert draft.periodBudget["maxToolCalls"] == 60
    assert draft.notificationPolicy["normalResults"] == "DIGEST"
    assert draft.notificationPolicy["exceptions"] == "IMMEDIATE"
    read_tools = {rule["toolRefs"][0] for rule in draft.grantRules if rule["maxRisk"] == "R0"}
    assert "release.check" in read_tools
    assert "memory.snapshot" not in read_tools
    todo_rule = next(rule for rule in draft.grantRules if rule["toolRefs"] == ["todo.propose"])
    assert todo_rule["resourceScope"] == {"projectIds": [9]}
    assert todo_rule["requiresReversible"] is True
    assert "*" not in todo_rule["toolRefs"]
    assert set(todo_rule["allowedFields"]) == {"title", "dueAt", "items"}


def test_project_templates_require_project_and_workspace_scope() -> None:
    with pytest.raises(ValueError, match="projectId"):
        build_mission_draft(_draft(projectId=None))
    with pytest.raises(ValueError, match="workspaceId"):
        build_mission_draft(_draft(workspaceId=None))


def test_release_readiness_declares_three_branch_join_policy() -> None:
    draft = build_mission_draft(_draft(
        goal="每天检查发布准备状态",
        templateId="release-readiness",
        autoMaintainTasks=False,
    ))

    assert draft.constraints["maxParallelReadBranches"] == 3
    assert draft.constraints["parallelJoinPolicy"] == "ALL"


def test_project_mission_rejects_non_git_workspace_before_activation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(settings, "postgres_enabled", True)
    monkeypatch.setattr(
        "app.missions.draft_builder.get_workspace",
        lambda *_args: {"rootPath": str(tmp_path), "status": "active"},
    )

    with pytest.raises(ValueError, match="不是 Git 工作区"):
        validate_mission_workspace(_draft())


def test_project_mission_accepts_authorized_git_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    subprocess.run(
        ["git", "init", "-q", str(tmp_path)],
        check=True,
        capture_output=True,
    )
    monkeypatch.setattr(settings, "postgres_enabled", True)
    monkeypatch.setattr(
        "app.missions.draft_builder.get_workspace",
        lambda *_args: {"rootPath": str(tmp_path), "status": "active"},
    )

    validate_mission_workspace(_draft())


def test_strict_schema_rejects_unknown_authority_fields() -> None:
    with pytest.raises(ValidationError):
        MissionDraftRequest.model_validate({
            **_draft().model_dump(),
            "approveAllFutureTools": True,
        })


def test_shadow_always_includes_system_forbidden_r2_and_r3_examples() -> None:
    draft = build_mission_draft(_draft())
    response = build_shadow_scenarios(ShadowScenarioRequest.model_validate({
        "missionId": "10",
        "templateId": draft.templateId,
        "goal": draft.goal,
        "projectId": draft.projectId,
        "constraints": draft.constraints,
        "grantRules": draft.grantRules,
    }))

    assert any(action.toolRef == "todo.propose" and action.riskLevel == "R1" for action in response.actions)
    assert any(action.toolRef == "git.commit" and action.riskLevel == "R2" for action in response.actions)
    assert any(action.toolRef == "resource.delete" and action.riskLevel == "R3" for action in response.actions)


def test_delegated_write_requires_java_decision_before_storage_access() -> None:
    result = execute_delegated_action(_step())

    assert result.status == "ERROR"
    assert result.errorCode == "DELEGATION_DECISION_REQUIRED"
    assert result.sideEffect == "NONE"


def test_batch_todo_write_fails_before_any_database_write(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "postgres_enabled", True)
    result = execute_delegated_action(_step(
        delegatedWriteAuthorized=True,
        grantVersion=1,
        authorizationDecisionId="decision-1",
        input={
            "projectId": 9,
            "items": [{"title": "任务一"}, {"title": "任务二"}],
        },
    ))

    assert result.status == "ERROR"
    assert result.errorCode == "DELEGATED_BATCH_WRITE_NOT_SUPPORTED"
    assert result.sideEffect == "NONE"


def test_delegated_todo_requires_project_scope_before_database_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "postgres_enabled", True)
    result = execute_delegated_action(_step(
        projectId=None,
        delegatedWriteAuthorized=True,
        grantVersion=1,
        authorizationDecisionId="decision-1",
        input={"title": "没有项目范围的任务"},
    ))

    assert result.status == "ERROR"
    assert result.errorCode == "PROJECT_REQUIRED"
    assert result.sideEffect == "NONE"


def test_resolved_symlink_cannot_bypass_sensitive_path_rules(tmp_path) -> None:
    secret = tmp_path / ".env"
    secret.write_text("TOKEN=secret", encoding="utf-8")
    alias = tmp_path / "notes.md"
    alias.symlink_to(secret)

    with pytest.raises(ValueError, match="Secret|凭据"):
        _resolved_workspace_target(tmp_path.resolve(), "notes.md")


def test_digest_keeps_normal_results_separate_from_attention_items() -> None:
    digest = build_mission_digest(MissionDigestRequest.model_validate({
        "missionId": "10",
        "runs": [
            {"status": "COMPLETED", "evidenceRefs": ["agent-task://1"]},
            {"status": "FAILED"},
        ],
        "exceptions": [
            {"status": "OPEN"},
            {"status": "RESOLVED"},
        ],
    }))

    assert digest.completedRuns == 1
    assert digest.attentionCount == 1
    assert digest.evidenceRefs == ["agent-task://1"]
