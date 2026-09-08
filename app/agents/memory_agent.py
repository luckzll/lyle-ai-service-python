from app.tools.memory_tool import recall_memory_tool, write_memory_tool
from app.memory.recall_planner import plan_recall

_TRIVIAL_CHAT = {
    "你好",
    "您好",
    "嗨",
    "hi",
    "hello",
    "在吗",
    "谢谢",
    "好的",
    "ok",
}

_MEMORY_CUES = [
    "我的",
    "我之前",
    "记得",
    "记忆",
    "喜欢",
    "讨厌",
    "职业",
    "名字",
    "偏好",
    "上次",
    "继续",
    "刚才",
    "项目",
    "规划",
    "计划",
    "目标",
    "图片",
    "头像",
    "风格",
    "沟通",
    "边界",
]


def run_memory_agent(user_id: str, conversation_id: str | None, question: str) -> dict[str, object]:
    decision = decide_memory_recall(question)
    if not decision["should_recall"]:
        return {
            "memory": {},
            "memory_context": "",
            "decision": decision,
            "skipped": True,
        }

    result = recall_memory_tool(user_id, conversation_id, question, depth=str(decision["depth"]))
    return {
        "memory": result["memory"],
        "memory_context": result["memory_context"],
        "decision": decision,
        "skipped": False,
    }


def write_memory_agent(user_id: str, conversation_id: str | None, question: str, answer: str) -> dict[str, object]:
    return write_memory_tool(user_id, conversation_id, question, answer)


def decide_memory_recall(question: str) -> dict[str, object]:
    normalized = question.strip().lower()
    from app.memory.fact_store import enabled as facts_enabled
    if normalized and facts_enabled() and normalized in _TRIVIAL_CHAT:
        return {"should_recall": True, "depth": "light", "reason": "简短对话只带入精简核心画像和当前会话上下文。"}
    if not normalized or normalized in _TRIVIAL_CHAT:
        return {
            "should_recall": False,
            "depth": "skip",
            "reason": "短问候或确认语，不需要召回历史记忆。",
        }
    scene = str(plan_recall(question, allow_model=False).get("intent") or "default")
    if any(cue in question for cue in _MEMORY_CUES) or scene not in {"minimal", "default"}:
        return {
            "should_recall": True,
            "depth": "full",
            "reason": "问题包含个人信息、偏好或历史线索，需要完整记忆召回。",
        }
    return {
        "should_recall": True,
        "depth": "light",
        "reason": "普通对话保留轻量记忆召回，用于维持上下文连续性。",
    }
