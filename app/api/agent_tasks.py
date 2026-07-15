from fastapi import APIRouter, HTTPException

from app.agent_tasks.change_preview import build_change_preview
from app.agent_tasks.next_action import decide_next_action
from app.agent_tasks.planner import build_plan
from app.agent_tasks.result_reviewer import review_result
from app.agent_tasks.resume_check import check_resume_state
from app.agent_tasks.schemas import (
    ChangePreviewRequest,
    ChangePreviewResponse,
    ExecuteStepRequest,
    NextActionRequest,
    NextActionResponse,
    OperationStatusRequest,
    OperationStatusResponse,
    PlanRequest,
    PlanResponse,
    ReviewResultRequest,
    ReviewResultResponse,
    ResumeCheckRequest,
    ResumeCheckResponse,
    ToolResult,
)
from app.agent_tasks.tool_executor import execute_approved_step
from app.tools.operation_confirmation import get_pending_operation


router = APIRouter()


@router.post("/agent/plan", response_model=PlanResponse)
def plan(request: PlanRequest) -> PlanResponse:
    return build_plan(request)


@router.post("/agent/execute-step", response_model=ToolResult)
def execute_step(request: ExecuteStepRequest) -> ToolResult:
    return execute_approved_step(request)


@router.post("/agent/next-action", response_model=NextActionResponse)
def next_action(request: NextActionRequest) -> NextActionResponse:
    return decide_next_action(request)


@router.post("/agent/review-result", response_model=ReviewResultResponse)
def review(request: ReviewResultRequest) -> ReviewResultResponse:
    return review_result(request)


@router.post("/agent/change-preview", response_model=ChangePreviewResponse)
def change_preview(request: ChangePreviewRequest) -> ChangePreviewResponse:
    try:
        return build_change_preview(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/agent/operation-status", response_model=OperationStatusResponse)
def operation_status(request: OperationStatusRequest) -> OperationStatusResponse:
    try:
        operation = get_pending_operation(request.userId, request.operationId)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return OperationStatusResponse(
        status=str(operation.get("status") or "pending").upper(),
        result=operation.get("result") if isinstance(operation.get("result"), dict) else {},
    )


@router.post("/agent/resume-check", response_model=ResumeCheckResponse)
def resume_check(request: ResumeCheckRequest) -> ResumeCheckResponse:
    return check_resume_state(request)
