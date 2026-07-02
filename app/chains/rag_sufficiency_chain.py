import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import settings
from app.llm.tasks.judge_model import build_judge_task_model, is_judge_model_configured


def judge_rag_sufficiency(question: str, contexts: list[dict[str, str]]) -> dict[str, object]:
    if not settings.rag_sufficiency_check_enabled:
        return {"sufficient": True, "reason": "RAG sufficiency check disabled", "score": 1.0}
    if not contexts:
        return {"sufficient": False, "reason": "没有检索到知识库资料", "score": 0.0}
    if not is_judge_model_configured():
        return _rule_sufficiency(question, contexts)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "你是 RAG 资料充分性判断器。只返回 JSON，不要解释。"
                "判断知识库片段是否足够回答用户问题。"
                "返回格式：{\"sufficient\":true,\"score\":0.8,\"reason\":\"资料覆盖问题\"}"
            ),
        ),
        ("human", "用户问题：{question}\n\n知识库片段：\n{contexts}\n\n请只返回 JSON。"),
    ])
    chain = prompt | build_judge_task_model() | StrOutputParser()
    try:
        content = chain.invoke({"question": question, "contexts": _format_contexts(contexts)})
        parsed = json.loads(content)
    except Exception:
        return _rule_sufficiency(question, contexts)

    if not isinstance(parsed, dict):
        return _rule_sufficiency(question, contexts)
    return {
        "sufficient": bool(parsed.get("sufficient", True)),
        "score": _safe_float(parsed.get("score"), 0.5),
        "reason": str(parsed.get("reason") or ""),
    }


def _rule_sufficiency(question: str, contexts: list[dict[str, str]]) -> dict[str, object]:
    text = " ".join(item.get("chunk", "") for item in contexts)
    overlap = len(set(question) & set(text))
    sufficient = overlap >= max(2, min(6, len(set(question)) // 3))
    return {
        "sufficient": sufficient,
        "score": min(1.0, overlap / 12),
        "reason": "规则判断：问题关键词与资料有重合" if sufficient else "规则判断：资料与问题重合较少",
    }


def _format_contexts(contexts: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"来源：{item.get('title', '')}\n内容：{item.get('chunk', '')}"
        for item in contexts
    )


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
