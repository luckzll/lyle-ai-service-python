import json
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.tasks.judge_model import build_judge_task_model, is_judge_model_configured


def invoke_reviewer_chain(
    question: str,
    answer: str,
    contexts: list[dict[str, str]],
    memory_context: str,
    tool_results: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    if not is_judge_model_configured():
        return _rule_review(question, answer, contexts, memory_context)

    model = build_judge_task_model().bind(response_format={"type": "json_object"})
    response = model.invoke([
        SystemMessage(
            content=(
                "你是 AI 回答质量审核器。只返回 JSON，不要解释。"
                "检查回答是否答非所问、是否乱用记忆、是否在知识库资料不足时编造、是否缺少必要说明。"
                "如果回答中的数据已由工具结果提供，不要判定为编造。"
                "日期判断必须以用户当前日期为准，不要使用模型内部日期。"
                "对于汇总类问题，允许回答按类别概括、合并重复项、只列重点；"
                "不要因为未逐字列出工具结果中的预览项、重复测试项或低价值重复项而判定不通过。"
                "只有当回答的核心数量、状态、截止时间、项目归属或待确认/已执行语义明显错误时才判定不通过。"
                "返回格式："
                "{\"approved\":true,\"score\":0.9,\"reason\":\"通过\",\"rewrite_instruction\":\"\"}"
            )
        ),
        HumanMessage(
            content=(
                f"用户问题：\n{question}\n\n"
                f"回答：\n{answer}\n\n"
                f"当前日期：\n{_current_date_context()}\n\n"
                f"可用记忆：\n{memory_context or '无'}\n\n"
                f"工具结果：\n{_format_tool_results(tool_results or [])}\n\n"
                f"知识库片段：\n{_format_contexts(contexts)}\n\n"
                "请审核回答，只返回 JSON。"
            )
        ),
    ])
    return _parse_review_json(str(response.content or ""))


def _rule_review(
    question: str,
    answer: str,
    contexts: list[dict[str, str]],
    memory_context: str,
) -> dict[str, object]:
    stripped = answer.strip()
    if not stripped:
        return {
            "approved": False,
            "score": 0.0,
            "reason": "回答为空",
            "rewrite_instruction": "请重新回答用户问题，给出明确有用的内容。",
        }
    if "API Key" in stripped or "调用失败" in stripped:
        return {
            "approved": False,
            "score": 0.2,
            "reason": "模型调用失败或配置缺失",
            "rewrite_instruction": "请在模型可用后重新生成回答。",
        }
    return {
        "approved": True,
        "score": 0.8,
        "reason": "规则审核通过",
        "rewrite_instruction": "",
    }


def _parse_review_json(content: str) -> dict[str, object]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return {
            "approved": True,
            "score": 0.5,
            "reason": "审核结果不是合法 JSON，默认放行",
            "rewrite_instruction": "",
        }
    return {
        "approved": bool(parsed.get("approved", True)),
        "score": _safe_float(parsed.get("score"), 0.5),
        "reason": str(parsed.get("reason") or ""),
        "rewrite_instruction": str(parsed.get("rewrite_instruction") or parsed.get("rewriteInstruction") or ""),
    }


def _format_contexts(contexts: list[dict[str, str]]) -> str:
    if not contexts:
        return "无"
    return "\n\n".join(
        f"来源：{item.get('title', '')}\n内容：{item.get('chunk', '')}"
        for item in contexts
    )


def _format_tool_results(tool_results: list[dict[str, object]]) -> str:
    if not tool_results:
        return "无"
    lines: list[str] = []
    for item in tool_results[:8]:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or "")
        ok = bool(item.get("ok"))
        content = str(item.get("content") or "")
        lines.append(f"- {tool}({'成功' if ok else '失败'}): {content[:4000]}")
    return "\n".join(lines) if lines else "无"


def _current_date_context() -> str:
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    return f"{now:%Y-%m-%d %H:%M:%S} Asia/Shanghai"


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
