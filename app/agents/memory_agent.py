from app.tools.memory_tool import recall_memory_tool, write_memory_tool

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
    "我",
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

    result = recall_memory_tool(user_id, conversation_id, question)
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
    if normalized in _TRIVIAL_CHAT:
        return {
            "should_recall": False,
            "depth": "skip",
            "reason": "短问候或确认语，不需要召回历史记忆。",
        }
    if any(cue in question for cue in _MEMORY_CUES):
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
