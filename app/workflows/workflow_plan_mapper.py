from __future__ import annotations

from typing import Any


def plan_to_workflow_payload(plan: dict[str, Any], available_tool_names: set[str]) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    for index, plan_step in enumerate(plan.get("steps") or []):
        tool_name = str(plan_step.get("toolName") or "").strip()
        executable_tool = tool_name if tool_name in available_tool_names else ""
        description = str(plan_step.get("description") or "").strip()
        step_type = "mcp_tool" if executable_tool else "summarize"
        steps.append(
            {
                "title": str(plan_step.get("title") or f"步骤 {index + 1}"),
                "stepType": step_type,
                "toolName": executable_tool or None,
                "inputTemplate": {} if executable_tool else {
                    "planStepId": plan_step.get("id"),
                    "description": description,
                },
                "outputKey": f"plan_step_{plan_step.get('id') or index + 1}",
                "requiresConfirmation": _requires_confirmation(plan_step),
                "sortOrder": index,
            }
        )
    return {
        "projectId": plan.get("projectId"),
        "name": f"{plan.get('title') or '未命名计划'} 工作流",
        "description": str(plan.get("objective") or ""),
        "status": "active",
        "triggerType": "manual",
        "createdFromPlanId": plan.get("id"),
        "steps": steps,
    }


def _requires_confirmation(step: dict[str, Any]) -> bool:
    if step.get("status") == "awaiting_user":
        return True
    text = " ".join(
        str(step.get(key) or "")
        for key in ("title", "description", "toolName")
    ).lower()
    return any(keyword in text for keyword in ("确认", "写入", "create", "update", "delete", "send", "external"))
