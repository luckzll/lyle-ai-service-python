from app.core.config import settings
from app.chains.tool_router_chain import route_tools_with_llm
from app.memory.store import format_memory_context, recall_memory
from app.tools.image_intent import is_image_generation_request
from app.tools.tool_registry import execute_tool, format_tool_context, select_tools


def should_use_tool_agent(question: str) -> bool:
    return settings.tool_agent_enabled and (settings.tool_router_llm_enabled or bool(select_tools(question)))


def run_tool_agent(
    user_id: str,
    conversation_id: str | None,
    question: str,
    use_tools: bool,
) -> dict[str, object]:
    decision = decide_tool_use(user_id, conversation_id, question, use_tools)
    if not decision["should_call_tools"]:
        return {
            "tool_results": [],
            "tool_context": "",
            "selected_tools": [],
            "decision": decision,
            "skipped": True,
        }

    selected_tools = list(decision["selected_tools"])
    tool_inputs = decision.get("tool_inputs") or {}
    results = [
        execute_tool(tool_name, user_id, conversation_id, _tool_question(tool_name, question, tool_inputs))
        for tool_name in selected_tools
    ]
    return {
        "tool_results": results,
        "tool_context": format_tool_context(results),
        "selected_tools": selected_tools,
        "decision": decision,
        "skipped": False,
    }


def decide_tool_use(
    user_id: str,
    conversation_id: str | None,
    question: str,
    use_tools: bool,
) -> dict[str, object]:
    rule_selected_tools = select_tools(question)
    if not settings.tool_agent_enabled:
        return {
            "should_call_tools": False,
            "selected_tools": [],
            "tool_inputs": {},
            "reason": "Tool Agent 总开关未开启。",
        }
    if not use_tools:
        return {
            "should_call_tools": False,
            "selected_tools": rule_selected_tools,
            "tool_inputs": {},
            "reason": "Supervisor 未激活工具调用。",
        }
    if not rule_selected_tools and _looks_like_project_goal_write(question):
        return {
            "should_call_tools": False,
            "selected_tools": [],
            "tool_inputs": {},
            "reason": "项目目标写入由 V3 目标候选确认承接，跳过待办工具路由。",
            "router": "v3_project_guard",
            "confidence": 1.0,
            "needs_clarification": False,
        }
    if rule_selected_tools and _can_skip_llm_router(rule_selected_tools):
        return {
            "should_call_tools": True,
            "selected_tools": rule_selected_tools,
            "tool_inputs": {},
            "reason": "问题命中了明确工具触发规则，跳过 LLM Tool Router。",
            "router": "rule",
            "rule_selected_tools": rule_selected_tools,
            "confidence": 1.0,
            "needs_clarification": False,
        }

    router_decision = _route_tools_with_context(user_id, conversation_id, question)
    selected_tools = _merge_tools(rule_selected_tools, router_decision.get("selected_tools") or [])
    if "image_generate" in selected_tools and not is_image_generation_request(question):
        selected_tools = [tool_name for tool_name in selected_tools if tool_name != "image_generate"]
    tool_inputs = router_decision.get("tool_inputs") or {}
    confidence = _decision_confidence(router_decision)

    if not rule_selected_tools and router_decision.get("needs_clarification"):
        return {
            "should_call_tools": False,
            "selected_tools": [],
            "tool_inputs": {},
            "reason": str(router_decision.get("reason") or "工具调用信息不足，需要先澄清。"),
            "router": router_decision.get("router", "llm"),
            "confidence": confidence,
            "needs_clarification": True,
        }

    if not rule_selected_tools and router_decision.get("selected_tools") and confidence < settings.tool_router_confidence_threshold:
        return {
            "should_call_tools": False,
            "selected_tools": list(router_decision.get("selected_tools") or []),
            "tool_inputs": tool_inputs,
            "reason": f"Tool Router 置信度过低（{confidence:.2f}），跳过工具调用。",
            "router": router_decision.get("router", "llm"),
            "confidence": confidence,
            "needs_clarification": False,
        }

    if not selected_tools:
        return {
            "should_call_tools": False,
            "selected_tools": [],
            "tool_inputs": {},
            "reason": str(router_decision.get("reason") or "问题没有匹配到可调用工具。"),
            "router": router_decision.get("router", "rule"),
            "confidence": confidence,
            "needs_clarification": bool(router_decision.get("needs_clarification", False)),
        }
    return {
        "should_call_tools": True,
        "selected_tools": selected_tools,
        "tool_inputs": tool_inputs,
        "reason": _decision_reason(rule_selected_tools, router_decision),
        "router": router_decision.get("router", "rule") if router_decision.get("selected_tools") else "rule",
        "rule_selected_tools": rule_selected_tools,
        "confidence": confidence if router_decision.get("selected_tools") else 1.0,
        "needs_clarification": False,
    }


def _route_tools_with_context(user_id: str, conversation_id: str | None, question: str) -> dict[str, object]:
    if not settings.tool_router_llm_enabled:
        return {"selected_tools": [], "tool_inputs": {}, "reason": "LLM Tool Router 未开启。", "router": "rule"}
    try:
        memory = recall_memory(user_id, conversation_id, question)
        context = format_memory_context(memory)
    except Exception:
        context = ""
    clipped_context = context[: settings.tool_router_context_max_chars]
    return route_tools_with_llm(question, clipped_context)


def _can_skip_llm_router(rule_selected_tools: list[str]) -> bool:
    return all(tool_name in {"todo", "calendar", "health_check", "memory_snapshot", "memory_stats", "work_summary", "list_tools"} for tool_name in rule_selected_tools)


def _merge_tools(rule_tools: list[str], router_tools: object) -> list[str]:
    merged = []
    for tool_name in [*rule_tools, *(router_tools if isinstance(router_tools, list) else [])]:
        if tool_name and tool_name not in merged:
            merged.append(str(tool_name))
    return _prioritize_image_tools(merged)


def _prioritize_image_tools(tools: list[str]) -> list[str]:
    if "image_edit" in tools:
        return [tool for tool in tools if tool not in {"image_generate", "image_analyze"}]
    return tools


def _decision_reason(rule_tools: list[str], router_decision: dict[str, object]) -> str:
    if rule_tools and router_decision.get("selected_tools"):
        return f"规则和 LLM Tool Router 都命中。{router_decision.get('reason') or ''}".strip()
    if router_decision.get("selected_tools"):
        return str(router_decision.get("reason") or "LLM Tool Router 命中。")
    return "问题命中了工具触发规则。"


def _decision_confidence(router_decision: dict[str, object]) -> float:
    try:
        return float(router_decision.get("confidence", 0.75))
    except (TypeError, ValueError):
        return 0.75


def _looks_like_project_goal_write(question: str) -> bool:
    return (
        any(keyword in question for keyword in ["新增目标", "记录目标", "添加目标", "创建目标", "目标：", "目标:"])
        and any(keyword in question for keyword in ["项目", "project", "版本"])
        and not any(keyword in question for keyword in ["创建待办", "加个待办", "待办", "todo", "提醒我", "记得提醒我"])
    )


def _tool_question(tool_name: str, question: str, tool_inputs: object) -> str:
    if not isinstance(tool_inputs, dict):
        return question
    rewritten = str(tool_inputs.get(tool_name) or "").strip()
    if not rewritten:
        return question
    if tool_name in {"image_generate", "image_edit", "web_search"}:
        return rewritten
    return question
