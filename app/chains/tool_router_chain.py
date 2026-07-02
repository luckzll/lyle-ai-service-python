import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.llm.tasks.intent_model import build_intent_task_model, is_intent_model_configured


ALLOWED_TOOLS = {
    "health_check",
    "memory_snapshot",
    "memory_stats",
    "web_search",
    "weather",
    "file_parse",
    "image_analyze",
    "image_generate",
    "image_edit",
    "todo",
    "calendar",
    "list_tools",
}


def route_tools_with_llm(question: str, context: str = "") -> dict[str, object]:
    if not is_intent_model_configured():
        return _empty_decision("Tool Router 模型未配置。")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "你是 Tool Router Agent。根据用户当前输入和最近上下文，判断是否需要调用工具。"
                "只返回 JSON，不要解释。可用工具："
                "health_check=检查服务状态；memory_snapshot=查看用户记忆；memory_stats=记忆统计；"
                "web_search=联网搜索最新/实时/外部信息；weather=查询指定城市天气和短期预报；file_parse=解析文件路径；"
                "image_analyze=识别分析用户给出的图片；image_generate=纯文生图；"
                "image_edit=根据用户上传的图片和修改指令进行改图/图生图；"
                "todo=创建待办；calendar=创建日程；list_tools=列出工具。"
                "如果用户说“再来一个/换成/改成/按刚才风格/这个风格再出一版”等，"
                "并且上下文显示刚才在生成图片，应调用 image_generate，并把 tool_inputs.image_generate "
                "改写成完整生图提示词。"
                "如果用户上传了图片并要求换背景、去掉元素、改风格、替换物体或图生图，应调用 image_edit，"
                "并把 tool_inputs.image_edit 改写成包含图片链接和编辑要求的完整提示词。"
                "如果用户问天气、气温、下雨下雪，且包含城市或地区，应调用 weather；如果缺少城市，设置 needs_clarification=true。"
                "如果当前输入缺少必要对象、时间、文件路径或搜索主题，设置 needs_clarification=true，"
                "不要强行调用工具。confidence 取 0 到 1，表示你对工具选择的置信度。"
                "不要为了普通闲聊调用工具。"
                "返回格式："
                "{{\"should_call_tools\":true,\"selected_tools\":[\"image_generate\"],"
                "\"tool_inputs\":{{\"image_generate\":\"完整提示词\"}},"
                "\"confidence\":0.88,\"needs_clarification\":false,\"reason\":\"...\"}}"
            ),
        ),
        (
            "human",
            "最近上下文：\n{context}\n\n当前用户输入：\n{question}\n\n请只返回 JSON。",
        ),
    ])
    chain = prompt | build_intent_task_model() | StrOutputParser()
    try:
        content = chain.invoke({"question": question, "context": context or "无"})
        parsed = json.loads(str(content or "").strip())
    except Exception:
        return _empty_decision("Tool Router JSON 解析失败。")

    selected = [
        str(item).strip()
        for item in parsed.get("selected_tools", [])
        if str(item).strip() in ALLOWED_TOOLS
    ] if isinstance(parsed, dict) else []
    tool_inputs_raw = parsed.get("tool_inputs", {}) if isinstance(parsed, dict) else {}
    tool_inputs = {
        str(key): str(value)
        for key, value in tool_inputs_raw.items()
        if str(key) in ALLOWED_TOOLS and str(value).strip()
    } if isinstance(tool_inputs_raw, dict) else {}
    confidence = _parse_confidence(parsed.get("confidence", 0.75)) if isinstance(parsed, dict) else 0.0
    needs_clarification = bool(parsed.get("needs_clarification")) if isinstance(parsed, dict) else False
    should_call = (
        bool(parsed.get("should_call_tools"))
        and bool(selected)
        and not needs_clarification
    ) if isinstance(parsed, dict) else False

    return {
        "should_call_tools": should_call,
        "selected_tools": selected if should_call else [],
        "tool_inputs": tool_inputs if should_call else {},
        "reason": str(parsed.get("reason") or "LLM Tool Router 判断。") if isinstance(parsed, dict) else "",
        "confidence": confidence,
        "needs_clarification": needs_clarification,
        "router": "llm",
    }


def _empty_decision(reason: str) -> dict[str, object]:
    return {
        "should_call_tools": False,
        "selected_tools": [],
        "tool_inputs": {},
        "reason": reason,
        "confidence": 0.0,
        "needs_clarification": False,
        "router": "llm",
    }


def _parse_confidence(value: object) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(max(confidence, 0.0), 1.0)
