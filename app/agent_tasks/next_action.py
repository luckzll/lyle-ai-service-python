from app.agent_tasks.schemas import NextActionRequest, NextActionResponse


PROMPT_VERSION = "lyle-agent-v5-next-action-1"


def decide_next_action(request: NextActionRequest) -> NextActionResponse:
    result = request.toolResult
    user_action = str(request.latestUserInput.get("action") or "").upper()
    if user_action == "SKIP":
        return NextActionResponse(action="EXECUTE_STEP", reason="用户已明确跳过受阻步骤，继续执行剩余已批准步骤。", promptVersion=PROMPT_VERSION)
    if result.status == "ERROR":
        if result.retryable:
            return NextActionResponse(action="RETRY_STEP", reason="工具返回可重试错误，由控制器在预算内退避重试。", promptVersion=PROMPT_VERSION)
        return NextActionResponse(action="REQUEST_INPUT", reason="工具错误不可自动恢复，需要用户修复、跳过或终止。", promptVersion=PROMPT_VERSION)
    if result.securityFlags:
        return NextActionResponse(
            action="EXECUTE_STEP" if request.remainingStepCount > 0 else "FINALIZE",
            reason="工具输出包含不可信指令特征；忽略其中的动作要求，仅沿已批准计划继续。",
            promptVersion=PROMPT_VERSION,
        )
    if result.status == "PARTIAL" and request.remainingStepCount > 0:
        return NextActionResponse(
            action="REVISE_PLAN",
            reason="当前步骤只获得部分结果，需要用户选择保留现有路径、跳过或批准替代计划。",
            promptVersion=PROMPT_VERSION,
        )
    if request.remainingStepCount <= 0:
        return NextActionResponse(action="FINALIZE", reason="计划步骤已执行完毕，应进入结果复核。", promptVersion=PROMPT_VERSION)
    return NextActionResponse(action="EXECUTE_STEP", reason="当前步骤产生了有效结果，可继续执行已批准计划。", promptVersion=PROMPT_VERSION)
