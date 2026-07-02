from langchain_openai import ChatOpenAI

from app.llm.model_router import (
    TASK_MEMORY,
    TASK_SUMMARY,
    build_chat_model_for_task,
    is_task_model_configured,
)


def build_memory_task_model(streaming: bool = False) -> ChatOpenAI:
    return build_chat_model_for_task(TASK_MEMORY, streaming=streaming)


def build_summary_task_model(streaming: bool = False) -> ChatOpenAI:
    return build_chat_model_for_task(TASK_SUMMARY, streaming=streaming)


def is_memory_model_configured() -> bool:
    return is_task_model_configured(TASK_MEMORY)


def is_summary_model_configured() -> bool:
    return is_task_model_configured(TASK_SUMMARY)
