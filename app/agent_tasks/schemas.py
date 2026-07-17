from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Budget(StrictModel):
    maxSteps: int = Field(ge=1, le=20)
    maxToolCalls: int = Field(ge=1, le=40)
    maxModelTokens: int = Field(ge=1, le=200_000)
    maxCostMicros: int = Field(ge=1, le=10_000_000)
    maxWallTimeMillis: int = Field(ge=1, le=3_600_000)
    maxRetriesPerStep: int = Field(ge=1, le=3)
    maxConsecutiveNoProgress: int = Field(ge=1, le=3)


class ToolCatalogItem(StrictModel):
    toolRef: str = Field(min_length=1, max_length=255)
    riskLevel: Literal["READ", "EXTERNAL_READ", "WRITE_LOW"]


class PlanRequest(StrictModel):
    taskId: str = Field(min_length=1, max_length=64)
    userId: str = Field(min_length=1, max_length=64)
    goal: str = Field(min_length=1, max_length=10_000)
    completionCriteria: list[Any] = Field(min_length=1, max_length=20)
    constraints: dict[str, Any] = Field(default_factory=dict)
    templateId: str | None = Field(default=None, max_length=64)
    autonomyLevel: int = Field(ge=0, le=2)
    projectId: int | None = Field(default=None, gt=0)
    conversationId: int | None = Field(default=None, gt=0)
    budget: Budget
    toolCatalog: list[ToolCatalogItem] = Field(min_length=1, max_length=20)


class PlanStep(StrictModel):
    stepKey: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    stepType: Literal["READ", "SEARCH", "TOOL", "ANALYZE", "PROPOSE_CHANGE"]
    title: str = Field(min_length=1, max_length=255)
    purpose: str = Field(min_length=1, max_length=2_000)
    toolRef: str = Field(min_length=1, max_length=255)
    inputTemplate: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list, max_length=20)
    riskLevel: Literal["READ", "EXTERNAL_READ", "WRITE_LOW"]
    expectedArtifact: dict[str, Any] = Field(default_factory=dict)
    writeIntent: bool = False

    @model_validator(mode="after")
    def risk_matches_write_intent(self) -> "PlanStep":
        if self.writeIntent != (self.riskLevel == "WRITE_LOW"):
            raise ValueError("writeIntent 与 riskLevel 不一致")
        return self


class PlanResponse(StrictModel):
    understoodGoal: str = Field(min_length=1, max_length=10_000)
    inferredUnderstanding: list[str] = Field(default_factory=list, max_length=10)
    blockingQuestions: list[str] = Field(default_factory=list, max_length=3)
    accessScope: dict[str, Any]
    budget: Budget
    steps: list[PlanStep] = Field(min_length=1, max_length=20)
    promptVersion: str = Field(min_length=1, max_length=64)


class ExecuteStepRequest(StrictModel):
    taskId: str = Field(min_length=1, max_length=64)
    taskRunId: str = Field(min_length=1, max_length=64)
    stepRunId: str = Field(min_length=1, max_length=64)
    userId: str = Field(min_length=1, max_length=64)
    projectId: int | None = Field(default=None, gt=0)
    conversationId: str | None = Field(default=None, max_length=64)
    goal: str = Field(min_length=1, max_length=10_000)
    completionCriteria: list[Any] = Field(max_length=20)
    toolRef: str = Field(min_length=1, max_length=255)
    input: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[dict[str, Any]] = Field(default_factory=list, max_length=100)
    idempotencyKey: str = Field(min_length=16, max_length=128)
    remainingBudget: dict[str, int]
    delegatedWriteAuthorized: bool = False
    grantVersion: int | None = Field(default=None, ge=1)
    authorizationDecisionId: str | None = Field(default=None, max_length=64)


class ToolUsage(StrictModel):
    modelTokens: int = Field(default=0, ge=0)
    costMicros: int = Field(default=0, ge=0)


class ToolCriterionEvidence(StrictModel):
    criterion: str = Field(min_length=1, max_length=1_000)
    status: Literal["COMPLETE", "INCOMPLETE", "UNVERIFIED"]
    evidenceRefs: list[Any] = Field(default_factory=list, max_length=50)
    note: str = Field(default="", max_length=2_000)


class ToolResult(StrictModel):
    status: Literal["SUCCESS", "PARTIAL", "ERROR"]
    summary: str = Field(min_length=1, max_length=8_000)
    dataRef: str | None = Field(default=None, max_length=512)
    evidenceRefs: list[Any] = Field(default_factory=list, max_length=50)
    sideEffect: Literal["NONE", "PROPOSED", "APPLIED"] = "NONE"
    retryable: bool = False
    errorCode: str | None = Field(default=None, max_length=80)
    contentTrust: Literal["TRUSTED", "UNTRUSTED_EXTERNAL"] = "TRUSTED"
    redacted: bool = False
    sensitivity: Literal["NONE", "PERSONAL", "EXTERNAL_CONTENT"] = "NONE"
    securityFlags: list[str] = Field(default_factory=list, max_length=20)
    criterionEvidence: list[ToolCriterionEvidence] = Field(default_factory=list, max_length=20)
    externalStateHashes: dict[str, str] = Field(default_factory=dict)
    usage: ToolUsage = Field(default_factory=ToolUsage)
    compensation: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def error_fields_match_status(self) -> "ToolResult":
        if self.status == "ERROR" and not self.errorCode:
            raise ValueError("ERROR ToolResult 必须提供 errorCode")
        if self.status != "ERROR" and self.errorCode:
            raise ValueError("成功 ToolResult 不得携带 errorCode")
        return self


class NextActionRequest(StrictModel):
    taskId: str = Field(min_length=1, max_length=64)
    goal: str = Field(min_length=1, max_length=10_000)
    completionCriteria: list[Any] = Field(max_length=20)
    currentStepKey: str = Field(min_length=1, max_length=64)
    toolResult: ToolResult
    remainingStepCount: int = Field(ge=0, le=20)
    budgetUsed: dict[str, Any]
    budgetLimit: Budget
    latestUserInput: dict[str, Any] = Field(default_factory=dict)


class NextActionResponse(StrictModel):
    action: Literal[
        "EXECUTE_STEP", "RETRY_STEP", "REVISE_PLAN", "REQUEST_INPUT",
        "REQUEST_CONFIRMATION", "FINALIZE", "STOP",
    ]
    reason: str = Field(min_length=1, max_length=1_000)
    promptVersion: str = Field(min_length=1, max_length=64)


class ReviewResultRequest(StrictModel):
    taskId: str = Field(min_length=1, max_length=64)
    goal: str = Field(min_length=1, max_length=10_000)
    completionCriteria: list[Any] = Field(min_length=1, max_length=20)
    steps: list[dict[str, Any]] = Field(max_length=100)
    artifacts: list[dict[str, Any]] = Field(max_length=200)
    budgetUsed: dict[str, Any]


class CriterionReview(StrictModel):
    criterion: str = Field(min_length=1, max_length=1_000)
    status: Literal["COMPLETE", "INCOMPLETE", "UNVERIFIED"]
    evidenceRefs: list[Any] = Field(default_factory=list, max_length=50)
    note: str = Field(default="", max_length=2_000)


class ReviewResultResponse(StrictModel):
    outcome: Literal["COMPLETE", "PARTIAL", "FAILED"]
    summary: str = Field(min_length=1, max_length=8_000)
    criteria: list[CriterionReview] = Field(min_length=1, max_length=20)
    evidenceRefs: list[Any] = Field(default_factory=list, max_length=100)
    facts: list[str] = Field(default_factory=list, max_length=50)
    inferences: list[str] = Field(default_factory=list, max_length=50)
    recommendations: list[str] = Field(default_factory=list, max_length=50)
    gaps: list[str] = Field(default_factory=list, max_length=50)
    contentRef: str | None = Field(default=None, max_length=512)
    promptVersion: str = Field(min_length=1, max_length=64)


class ChangePreviewRequest(StrictModel):
    taskId: str = Field(min_length=1, max_length=64)
    taskRunId: str = Field(min_length=1, max_length=64)
    stepRunId: str = Field(min_length=1, max_length=64)
    userId: str = Field(min_length=1, max_length=64)
    conversationId: str | None = Field(default=None, max_length=64)
    goal: str = Field(min_length=1, max_length=10_000)
    toolRef: Literal["todo.propose", "file.patch.propose"]
    input: dict[str, Any]
    idempotencyKey: str = Field(min_length=16, max_length=128)


class ChangePreviewResponse(StrictModel):
    pendingOperationId: int = Field(gt=0)
    resourceRef: str = Field(min_length=1, max_length=512)
    actionType: str = Field(min_length=1, max_length=80)
    summary: str = Field(min_length=1, max_length=2_000)
    before: dict[str, Any] = Field(default_factory=dict)
    after: dict[str, Any] = Field(default_factory=dict)
    impact: str = Field(min_length=1, max_length=2_000)
    undoSupported: bool
    compensation: dict[str, Any] = Field(default_factory=dict)


class OperationStatusRequest(StrictModel):
    userId: str = Field(min_length=1, max_length=64)
    operationId: int = Field(gt=0)


class OperationStatusResponse(StrictModel):
    status: Literal["PENDING", "CONFIRMED", "REJECTED", "FAILED"]
    result: dict[str, Any] = Field(default_factory=dict)


class ResumeCheckRequest(StrictModel):
    userId: str = Field(min_length=1, max_length=64)
    taskId: str = Field(min_length=1, max_length=64)
    permissionSnapshot: dict[str, Any] = Field(default_factory=dict)
    expectedExternalState: dict[str, str] = Field(default_factory=dict)


class ResumeCheckResponse(StrictModel):
    valid: bool
    changed: bool
    reason: str = Field(min_length=1, max_length=2_000)
    currentExternalState: dict[str, str] = Field(default_factory=dict)
