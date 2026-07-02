from app.core.config import settings
from app.llm.model_router import (
    TASK_CHAT,
    TASK_INTENT,
    TASK_JUDGE,
    TASK_MEMORY,
    TASK_SUMMARY,
    get_model_config,
    is_task_model_configured,
)


def test_model_router_uses_task_specific_model(monkeypatch):
    monkeypatch.setattr(settings, "deepseek_api_key", "test-key")
    monkeypatch.setattr(settings, "chat_model_provider", "deepseek")
    monkeypatch.setattr(settings, "chat_model_name", "deepseek-chat")
    monkeypatch.setattr(settings, "intent_model_provider", "local_ollama")
    monkeypatch.setattr(settings, "intent_model_name", "qwen2.5:7b")
    monkeypatch.setattr(settings, "ollama_base_url", "http://localhost:11434/v1")

    chat_config = get_model_config(TASK_CHAT)
    intent_config = get_model_config(TASK_INTENT)

    assert chat_config.provider == "deepseek"
    assert chat_config.model == "deepseek-chat"
    assert chat_config.api_key == "test-key"
    assert intent_config.provider == "local_ollama"
    assert intent_config.model == "qwen2.5:7b"
    assert intent_config.base_url == "http://localhost:11434/v1"


def test_model_router_configuration_checks(monkeypatch):
    monkeypatch.setattr(settings, "deepseek_api_key", "")
    monkeypatch.setattr(settings, "chat_model_provider", "deepseek")
    monkeypatch.setattr(settings, "memory_extract_model_provider", "deepseek")
    monkeypatch.setattr(settings, "summary_model_provider", "deepseek")
    monkeypatch.setattr(settings, "judge_model_provider", "deepseek")

    assert not is_task_model_configured(TASK_CHAT)
    assert not is_task_model_configured(TASK_MEMORY)
    assert not is_task_model_configured(TASK_SUMMARY)
    assert not is_task_model_configured(TASK_JUDGE)

    monkeypatch.setattr(settings, "intent_model_provider", "local_ollama")
    monkeypatch.setattr(settings, "intent_model_name", "qwen2.5:7b")
    monkeypatch.setattr(settings, "ollama_base_url", "http://localhost:11434/v1")

    assert is_task_model_configured(TASK_INTENT)
