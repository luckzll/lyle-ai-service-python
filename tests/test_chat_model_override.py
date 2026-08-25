import asyncio

import pytest

from app.core.config import settings
from app.core.runtime_config import current_chat_model_override, temporary_ai_config
from app.llm.model_router import TASK_CHAT, TASK_INTENT, get_model_config


def test_request_chat_model_override_only_changes_chat_task(monkeypatch):
    monkeypatch.setattr(settings, "chat_model_provider", "deepseek")
    monkeypatch.setattr(settings, "chat_model_name", "deepseek-chat")
    monkeypatch.setattr(settings, "intent_model_provider", "qwen")
    monkeypatch.setattr(settings, "intent_model_name", "qwen-flash")

    with temporary_ai_config({
        "CHAT_MODEL_PROVIDER": "qwen",
        "CHAT_MODEL_NAME": "deepseek-v4-pro",
    }):
        chat = get_model_config(TASK_CHAT)
        intent = get_model_config(TASK_INTENT)
        assert (chat.provider, chat.model) == ("qwen", "deepseek-v4-pro")
        assert (intent.provider, intent.model) == ("qwen", "qwen-flash")

    assert current_chat_model_override() is None
    restored = get_model_config(TASK_CHAT)
    assert (restored.provider, restored.model) == ("deepseek", "deepseek-chat")


def test_request_chat_model_override_rejects_incomplete_or_unknown_routes():
    with pytest.raises(ValueError, match="provided together"):
        with temporary_ai_config({"CHAT_MODEL_NAME": "deepseek-v4-pro"}):
            pass

    with pytest.raises(ValueError, match="Unsupported request chat model"):
        with temporary_ai_config({
            "CHAT_MODEL_PROVIDER": "openai",
            "CHAT_MODEL_NAME": "attacker/model",
        }):
            pass


def test_request_chat_model_overrides_are_isolated_between_concurrent_tasks():
    async def selected(provider: str, model: str) -> tuple[str, str]:
        with temporary_ai_config({
            "CHAT_MODEL_PROVIDER": provider,
            "CHAT_MODEL_NAME": model,
        }):
            await asyncio.sleep(0)
            config = get_model_config(TASK_CHAT)
            return config.provider, config.model

    async def run() -> list[tuple[str, str]]:
        return await asyncio.gather(
            selected("qwen", "qwen3.7-flash"),
            selected("openai", "openai/gpt-5.5"),
            selected("openai", "anthropic/claude-opus-4.6"),
            selected("openai", "google/gemini-3.5-flash"),
            selected("qwen", "kimi-k3"),
        )

    assert asyncio.run(run()) == [
        ("qwen", "qwen3.7-flash"),
        ("openai", "openai/gpt-5.5"),
        ("openai", "anthropic/claude-opus-4.6"),
        ("openai", "google/gemini-3.5-flash"),
        ("qwen", "kimi-k3"),
    ]
