import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.llm.tasks.intent_model import build_intent_task_model, is_intent_model_configured


MEMORY_KEYWORDS = ["我之前", "记得", "记忆", "我喜欢", "我的"]


def classify_intent(question: str, knowledge_base_ids: list[str]) -> str:
    if not is_intent_model_configured():
        return classify_intent_by_rule(question, knowledge_base_ids)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "你是意图分类器。只返回 JSON，不要解释。"
                "intent 只能是 memory、knowledge、chat。"
                "memory=用户询问个人历史、偏好、记忆；knowledge=需要查知识库；chat=普通对话。"
            ),
        ),
        (
            "human",
            "用户问题：{question}\n是否选择知识库：{has_knowledge_base}\n返回格式：{{\"intent\":\"chat\"}}",
        ),
    ])
    chain = prompt | build_intent_task_model() | StrOutputParser()
    try:
        content = chain.invoke({
            "question": question,
            "has_knowledge_base": bool(knowledge_base_ids),
        })
        intent = str(json.loads(content).get("intent") or "").strip()
    except Exception:
        return classify_intent_by_rule(question, knowledge_base_ids)

    return intent if intent in {"memory", "knowledge", "chat"} else classify_intent_by_rule(question, knowledge_base_ids)


def classify_intent_by_rule(question: str, knowledge_base_ids: list[str]) -> str:
    if any(keyword in question for keyword in MEMORY_KEYWORDS):
        return "memory"
    if knowledge_base_ids:
        return "knowledge"
    return "chat"
