from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MissionDraftRequest(StrictModel):
    userId: str = Field(min_length=1, max_length=64)
    goal: str = Field(min_length=5, max_length=10_000)
    title: str | None = Field(default=None, max_length=255)
    templateId: Literal["project-health", "research-weekly", "release-readiness"] | None = None
    projectId: int | None = Field(default=None, gt=0)
    workspaceId: int | None = Field(default=None, gt=0)
    timezone: str = Field(default="Asia/Shanghai", min_length=1, max_length=64)
    completionCriteria: list[str] = Field(default_factory=list, max_length=20)
    forbidden: list[str] = Field(default_factory=list, max_length=50)
    autoMaintainTasks: bool = False
    allowFilePatch: bool = False
    writeTarget: dict[str, Any] | None = None
    validityDays: int = Field(default=30, ge=1, le=30)
    maxValidityDays: int = Field(default=30, ge=1, le=30)
    schedule: dict[str, Any] = Field(default_factory=dict)
    quietHours: dict[str, Any] = Field(default_factory=dict)
    requestIdempotencyKey: str | None = Field(default=None, max_length=128)
    sourceMissionId: str | None = Field(default=None, max_length=64)


class MissionDraftResponse(StrictModel):
    title: str = Field(min_length=1, max_length=255)
    goal: str = Field(min_length=5, max_length=10_000)
    templateId: Literal["project-health", "research-weekly", "release-readiness"]
    projectId: int | None = Field(default=None, gt=0)
    completionCriteria: list[str] = Field(min_length=1, max_length=20)
    constraints: dict[str, Any]
    grantSummary: str = Field(min_length=1, max_length=2_000)
    grantRules: list[dict[str, Any]] = Field(min_length=1, max_length=50)
    budgetPolicy: dict[str, int]
    singleRunBudget: dict[str, int]
    periodBudget: dict[str, int]
    periodType: Literal["DAILY", "WEEKLY", "MONTHLY"]
    validityDays: int = Field(ge=1, le=30)
    notificationPolicy: dict[str, Any]
    quietHours: dict[str, Any]
    triggers: list[dict[str, Any]] = Field(min_length=1, max_length=5)
    promptVersion: str = Field(min_length=1, max_length=64)


class ShadowScenarioRequest(StrictModel):
    missionId: str = Field(min_length=1, max_length=64)
    templateId: Literal["project-health", "research-weekly", "release-readiness"]
    goal: str = Field(min_length=1, max_length=10_000)
    projectId: int = Field(default=0, ge=0)
    constraints: dict[str, Any] = Field(default_factory=dict)
    grantRules: list[dict[str, Any]] = Field(default_factory=list, max_length=50)


class ShadowAction(StrictModel):
    title: str = Field(min_length=1, max_length=255)
    toolRef: str = Field(min_length=1, max_length=255)
    operation: str = Field(min_length=1, max_length=64)
    riskLevel: Literal["R0", "R1", "R2", "R3"]
    writeIntent: bool
    input: dict[str, Any] = Field(default_factory=dict)


class ShadowScenarioResponse(StrictModel):
    actions: list[ShadowAction] = Field(min_length=1, max_length=20)
    promptVersion: str = Field(min_length=1, max_length=64)


class CompensationRequest(StrictModel):
    userId: str = Field(min_length=1, max_length=64)
    snapshotRef: str = Field(min_length=1, max_length=512)
    idempotencyKey: str = Field(min_length=1, max_length=128)


class CompensationResponse(StrictModel):
    status: Literal["SUCCEEDED", "CONFLICTED", "FAILED", "EXPIRED"]
    summary: str = Field(min_length=1, max_length=2_000)
    evidenceRefs: list[str] = Field(default_factory=list, max_length=20)


class MissionDigestRequest(StrictModel):
    missionId: str = Field(min_length=1, max_length=64)
    runs: list[dict[str, Any]] = Field(default_factory=list, max_length=100)
    exceptions: list[dict[str, Any]] = Field(default_factory=list, max_length=100)


class MissionDigestResponse(StrictModel):
    summary: str = Field(min_length=1, max_length=8_000)
    completedRuns: int = Field(ge=0)
    attentionCount: int = Field(ge=0)
    evidenceRefs: list[str] = Field(default_factory=list, max_length=100)
