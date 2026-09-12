from datetime import datetime
from zoneinfo import ZoneInfo

from app.prompts.store import DEFAULT_SYSTEM_PROMPT, get_active_system_prompt


SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT


def get_system_prompt() -> tuple[str, str]:
    return get_active_system_prompt()


def build_user_prompt(question: str, contexts: list[dict[str, str]], memory_context: str = "") -> str:
    runtime_text = _runtime_context_text()
    memory_text = (
        "可用上下文（可能包含记忆、已完成的工具调用结果等）：\n"
        f"{memory_context}\n\n"
        "如果上下文中包含工具调用结果，请直接基于工具结果回答用户问题，"
        "不要声称自己还没有调用工具，也不要只复述工具名。\n"
        "不要向用户解释内部工具的输入格式、接口限制、URL/base64/本地路径等实现细节；"
        "除非工具明确失败，否则只输出面向用户的结论。\n"
        "如果用户要求生成图片，但上下文没有 image_generate 成功结果，"
        "不要输出图片 Markdown 或伪造图片链接，只能说明需要调用图片生成工具。\n\n"
        if memory_context else ""
    )
    if not contexts:
        return f"{runtime_text}{memory_text}用户问题：{question}"

    context_text = "\n\n".join(
        f"来源：{item['title']}\n内容：{item['chunk']}" for item in contexts
    )
    return (
        "请优先根据以下知识库资料回答；如果资料不足，请明确说明不确定。\n\n"
        "只输出回答正文，不要输出参考来源列表，系统会在回答末尾自动追加来源。\n\n"
        f"{context_text}\n\n"
        f"{runtime_text}{memory_text}"
        f"用户问题：{question}"
    )


def _runtime_context_text() -> str:
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    return (
        "系统运行上下文：\n"
        f"- 当前北京时间：{now.strftime('%Y-%m-%d %H:%M:%S %A')}\n"
        "- 时区：Asia/Shanghai（UTC+08:00）\n"
        "- 记忆安全规则：仅对本轮新提取且尚未确认的候选记忆，不得声称‘已记住’或‘已写入长期记忆’；"
        "候选必须由用户确认后才生效。上下文里已有对象必须按各自真实状态回答："
        "active 画像是已确认记忆，open 任务是进行中，rejected 操作是已拒绝且未执行；"
        "不得把不同对象统一描述为待确认。\n\n"
    )
