import json

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings
from app.chat_reasoning import emit_reasoning, finish_reasoning
from app.llm.prompts import build_user_prompt, get_system_prompt
from app.llm.tasks.chat_model import build_chat_task_model, is_chat_model_configured
from app.llm.tasks.memory_model import (
    build_memory_task_model,
    build_summary_task_model,
    is_memory_model_configured,
    is_summary_model_configured,
)


_build_user_prompt = build_user_prompt


def generate_answer(question: str, contexts: list[dict[str, str]], memory_context: str = "") -> tuple[str, dict[str, int]]:
    if not is_chat_model_configured():
        return (
            "DeepSeek API Key 还没有配置。请在环境变量 DEEPSEEK_API_KEY 中填入你的 Key。",
            {"prompt_tokens": 0, "completion_tokens": 0},
        )

    system_prompt, prompt_version = get_system_prompt()
    response = build_chat_task_model().invoke([
        SystemMessage(content=f"{system_prompt}\n\nPrompt-Version: {prompt_version}"),
        HumanMessage(content=build_user_prompt(question, contexts, memory_context)),
    ])
    return str(response.content), _extract_usage(response)


def stream_answer(question: str, contexts: list[dict[str, str]], memory_context: str = ""):
    if not is_chat_model_configured():
        yield "DeepSeek API Key 还没有配置。请在环境变量 DEEPSEEK_API_KEY 中填入你的 Key。"
        return

    system_prompt, prompt_version = get_system_prompt()
    messages = [
        SystemMessage(content=f"{system_prompt}\n\nPrompt-Version: {prompt_version}"),
        HumanMessage(content=build_user_prompt(question, contexts, memory_context)),
    ]
    for chunk in build_chat_task_model(streaming=True).stream(messages):
        for kind, delta in _reasoning_deltas(chunk):
            emit_reasoning(delta, kind)
        content = _answer_delta(chunk.content)
        if content:
            finish_reasoning()
            yield content
    finish_reasoning()


def _reasoning_deltas(chunk):
    """Read allowlisted public text fields, never encrypted data or metadata."""
    additional = getattr(chunk, "additional_kwargs", None) or {}
    blocks = chunk.content if isinstance(chunk.content, list) else []
    reasoning_block = additional.get("reasoning")
    if isinstance(reasoning_block, dict):
        # LangChain's v0.3 Responses adapter omits `type` after the first
        # reasoning chunk; this explicitly named field still carries summaries.
        blocks = [*blocks, {**reasoning_block, "type": "reasoning"}]
    summaries = []
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "reasoning":
            continue
        for item in block.get("summary", []) if isinstance(block.get("summary"), list) else []:
            if isinstance(item, dict) and item.get("type") == "summary_text" and isinstance(item.get("text"), str):
                summaries.append(item["text"])
        metadata = getattr(chunk, "response_metadata", {}) or {}
        if (metadata.get("output_version") == "v1" and metadata.get("model_provider") == "openai"
                and isinstance(block.get("reasoning"), str)):
            # The OpenAI v1 translator maps summary_text to `reasoning`.
            summaries.append(block["reasoning"])
    if summaries:
        for text in summaries:
            if text:
                yield "summary", text
    elif isinstance(additional.get("reasoning_content"), str) and additional["reasoning_content"]:
        yield "reasoning", additional["reasoning_content"]


def _answer_delta(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        block if isinstance(block, str) else block["text"]
        for block in content
        if isinstance(block, str) or (
            isinstance(block, dict) and block.get("type") in {"text", "output_text"}
            and isinstance(block.get("text"), str)
        )
    )


def summarize_memory(existing_summary: str, turns_text: str) -> str:
    if not is_summary_model_configured():
        merged = f"{existing_summary}\n{turns_text}".strip()
        return merged[-3000:]

    response = build_summary_task_model().invoke([
        SystemMessage(
            content=(
                "你是智能体记忆压缩器。请把对话压缩为稳定、可复用的长期记忆。"
                "保留用户身份、偏好、长期目标、项目背景、明确事实；去掉寒暄、重复、临时细节。"
                "如果新信息修正旧信息，以新信息为准。输出中文，不超过 800 字。"
            )
        ),
        HumanMessage(
            content=(
                f"已有长期摘要：\n{existing_summary or '无'}\n\n"
                f"需要压缩的新对话：\n{turns_text}\n\n"
                "请输出更新后的长期摘要。"
            )
        ),
    ])
    content = str(response.content or "").strip()
    if not content:
        merged = f"{existing_summary}\n{turns_text}".strip()
        return merged[-3000:]
    return content


def extract_structured_memories(question: str, recent_turns_text: str) -> list[dict[str, object]]:
    from app.memory.profile_schema import PROFILE_SCHEMA

    if not is_memory_model_configured():
        return []

    model = build_memory_task_model().bind(response_format={"type": "json_object"})
    response = model.invoke([
        SystemMessage(
            content=(
                "你是结构化记忆抽取器。只抽取用户明确陈述的、长期有用的稳定事实。"
                "不要抽取问题、假设、建议、寒暄、临时状态，也不要把助手回答当成用户事实。"
                "可用 key 包括 name、identity、career、food_preference、pet_preference、sport_preference、"
                "entertainment_preference、preference、negative_preference、food_negative_preference、"
                "pet_negative_preference、sport_negative_preference、entertainment_negative_preference、"
                  "current_project、technology_stack、goal、constraint。"
                  "也可使用这些标准 Schema 槽位："
                  + "、".join(f"{path}（{item.label}）" for path, item in PROFILE_SCHEMA.items() if not item.control and not item.sensitive)
                  + "。语言习惯、社交偏好、边界和触发点必须来自用户明确陈述，不得推测心理或健康状况。"
                "偏好必须尽量细分：水果/食物用 food_preference，宠物/动物用 pet_preference，"
                "运动赛事用 sport_preference，唱歌/游戏/影视等用 entertainment_preference；"
                "不喜欢/讨厌类负向偏好使用 negative_preference 或对应细分 negative key；"
                "只有无法分类时才使用 preference。"
                "只返回 JSON 对象，不要输出解释。格式："
                "{\"memories\":[{\"key\":\"food_preference\",\"value\":\"西瓜\",\"confidence\":0.91}]}"
            )
        ),
        HumanMessage(
            content=(
                f"最近对话：\n{recent_turns_text or '无'}\n\n"
                f"当前用户输入：\n{question}\n\n"
                "请返回 JSON 对象。"
            )
        ),
    ])
    content = str(response.content or "").strip()
    return _parse_structured_memory_json(content)


def _extract_usage(response: object) -> dict[str, int]:
    metadata = getattr(response, "response_metadata", {}) or {}
    usage = metadata.get("token_usage") or metadata.get("usage") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
    }


def _parse_structured_memory_json(content: str) -> list[dict[str, object]]:
    if not content:
        return []
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, dict):
        items = parsed.get("memories") or parsed.get("items") or parsed.get("data") or []
    else:
        items = parsed
    if not isinstance(items, list):
        return []

    memories: list[dict[str, object]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        value = str(item.get("value") or "").strip()
        try:
            confidence = float(item.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0
        if key and value:
            memories.append({"key": key, "value": value, "confidence": confidence})
    return memories
