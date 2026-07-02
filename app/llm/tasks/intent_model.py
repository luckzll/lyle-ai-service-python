from langchain_openai import ChatOpenAI

from app.llm.model_router import TASK_INTENT, build_chat_model_for_task, is_task_model_configured


def build_intent_task_model(streaming: bool = False) -> ChatOpenAI:
    return build_chat_model_for_task(TASK_INTENT, streaming=streaming)


def is_intent_model_configured() -> bool:
    return is_task_model_configured(TASK_INTENT)
