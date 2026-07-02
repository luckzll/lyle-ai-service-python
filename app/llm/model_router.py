from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.llm.providers.base import ChatModelConfig, OpenAICompatibleProvider
from app.llm.providers.deepseek import DeepSeekProvider
from app.llm.providers.local_ollama import LocalOllamaProvider
from app.llm.providers.openai import OpenAIProvider
from app.llm.providers.qwen import QwenProvider


TASK_CHAT = "chat"
TASK_INTENT = "intent"
TASK_MEMORY = "memory"
TASK_SUMMARY = "summary"
TASK_JUDGE = "judge"


@lru_cache
def _providers() -> dict[str, OpenAICompatibleProvider]:
    providers = [
        DeepSeekProvider(),
        OpenAIProvider(),
        QwenProvider(),
        LocalOllamaProvider(),
    ]
    return {provider.name: provider for provider in providers}


def build_chat_model_for_task(task: str, streaming: bool = False) -> ChatOpenAI:
    config = get_model_config(task)
    provider = get_provider(config.provider)
    return provider.build_chat_model(config, streaming=streaming)


def is_task_model_configured(task: str) -> bool:
    config = get_model_config(task)
    provider = get_provider(config.provider)
    return provider.is_configured(config)


def get_provider(provider_name: str) -> OpenAICompatibleProvider:
    provider = _providers().get(provider_name)
    if not provider:
        raise ValueError(f"Unsupported LLM provider: {provider_name}")
    return provider


def get_model_config(task: str) -> ChatModelConfig:
    provider = _task_provider(task)
    return ChatModelConfig(
        provider=provider,
        model=_task_model(task),
        api_key=_provider_api_key(provider),
        base_url=_provider_base_url(provider),
        timeout=settings.llm_timeout_seconds,
    )


def _task_provider(task: str) -> str:
    defaults = {
        TASK_CHAT: settings.chat_model_provider,
        TASK_INTENT: settings.intent_model_provider,
        TASK_MEMORY: settings.memory_extract_model_provider,
        TASK_SUMMARY: settings.summary_model_provider,
        TASK_JUDGE: settings.judge_model_provider,
    }
    return defaults.get(task, settings.chat_model_provider).strip().lower()


def _task_model(task: str) -> str:
    defaults = {
        TASK_CHAT: settings.chat_model_name,
        TASK_INTENT: settings.intent_model_name,
        TASK_MEMORY: settings.memory_extract_model_name,
        TASK_SUMMARY: settings.summary_model_name,
        TASK_JUDGE: settings.judge_model_name,
    }
    return defaults.get(task, settings.chat_model_name).strip()


def _provider_api_key(provider: str) -> str:
    if provider == "deepseek":
        return settings.deepseek_api_key
    if provider == "openai":
        return settings.openai_api_key
    if provider == "qwen":
        return settings.qwen_api_key
    return ""


def _provider_base_url(provider: str) -> str:
    if provider == "deepseek":
        return settings.deepseek_base_url
    if provider == "openai":
        return settings.openai_base_url
    if provider == "qwen":
        return settings.qwen_base_url
    if provider == "local_ollama":
        return settings.ollama_base_url
    return settings.deepseek_base_url
