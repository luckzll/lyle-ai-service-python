import json
from typing import Literal
from pydantic import BaseModel, ConfigDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.llm.tasks.intent_model import build_intent_task_model, is_intent_model_configured
from app.chat_execution_plan import is_self_contained, needs_personal_memory, needs_tool_planning, references_knowledge


class IntentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: Literal["memory", "knowledge", "chat"]


def classify_intent(question: str, knowledge_base_ids: list[str]) -> str:
    # Cheap, unambiguous requests do not need a model just to decide that no
    # specialist is required. Explicit memory/document requests remain intact.
    if is_self_contained(question):
        return "chat"
    if needs_personal_memory(question) or references_knowledge(question):
        return classify_intent_by_rule(question, knowledge_base_ids)
    if needs_tool_planning(question):
        return "chat"
    if not is_intent_model_configured():
        return classify_intent_by_rule(question, knowledge_base_ids)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "你是意图分类器。只返回 JSON，不要解释。"
                "intent 只能是 memory、knowledge、chat。"
                "memory=用户明确需要个人历史、偏好、记忆；knowledge=本问题需要所选知识库中的证据；chat=普通对话或外部工具任务。"
                "知识库是否选择只代表可用范围，不能仅因为选择了知识库就判knowledge。"
                "数学、翻译、改写和通用概念问答不需要无关知识库；当前会话的上下文延续也不等于长期记忆召回。"
            ),
        ),
        (
            "human",
            "用户问题：{question}\n是否选择知识库：{has_knowledge_base}\n返回格式：{{\"intent\":\"chat\"}}",
        ),
    ])
    from app.core.config import settings
    model = build_intent_task_model()
    if settings.native_tool_calling_enabled:
        try:
            classifier = prompt | model.with_structured_output(IntentDecision, method="function_calling")
            return classifier.invoke({"question": question, "has_knowledge_base": bool(knowledge_base_ids)}).intent
        except Exception:
            return classify_intent_by_rule(question, knowledge_base_ids)
    chain = prompt | model | StrOutputParser()
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
    if needs_personal_memory(question):
        return "memory"
    if references_knowledge(question):
        return "knowledge"
    if is_self_contained(question) or needs_tool_planning(question):
        return "chat"
    # Availability is not intent. Without a model or a document reference there
    # is no evidence that this request needs the selected private collection.
    return "chat"
