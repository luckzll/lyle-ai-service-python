from langchain_openai import ChatOpenAI

from app.llm.model_router import TASK_CHAT, build_chat_model_for_task, is_task_model_configured


def build_chat_task_model(streaming: bool = False) -> ChatOpenAI:
    return build_chat_model_for_task(TASK_CHAT, streaming=streaming)


def is_chat_model_configured() -> bool:
    return is_task_model_configured(TASK_CHAT)
