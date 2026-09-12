from app.core.config import settings
from app.core.runtime_config import current_rag_config, current_review_config, temporary_ai_config


def test_temporary_ai_config_overrides_known_fields_and_restores_values():
    old_reviewer_enabled = settings.reviewer_enabled
    old_reviewer_max_retry = settings.reviewer_max_retry
    old_rerank_provider = settings.rerank_provider

    with temporary_ai_config({
        "REVIEWER_ENABLED": "true",
        "REVIEWER_MAX_RETRY": "3",
        "RERANK_PROVIDER": "lexical",
        "UNKNOWN_KEY": "ignored",
    }):
        assert current_review_config() == (True, 3)
        assert settings.reviewer_enabled == old_reviewer_enabled
        assert settings.reviewer_max_retry == old_reviewer_max_retry
        assert current_rag_config().rerank_provider == "lexical"
        assert settings.rerank_provider == old_rerank_provider

    assert settings.reviewer_enabled == old_reviewer_enabled
    assert settings.reviewer_max_retry == old_reviewer_max_retry
    assert settings.rerank_provider == old_rerank_provider


def test_temporary_ai_config_accepts_empty_config():
    old_prompt_version = settings.prompt_version
    with temporary_ai_config({}):
        assert settings.prompt_version == old_prompt_version
    assert settings.prompt_version == old_prompt_version


def test_temporary_ai_config_ignores_security_sensitive_keys():
    old_allowed_names = settings.tool_allowed_names
    old_agent_enabled = settings.tool_agent_enabled
    old_web_search_enabled = settings.tool_web_search_enabled
    old_internal_token = settings.internal_api_token

    with temporary_ai_config({
        "TOOL_ALLOWED_NAMES": "health_check,evil_tool",
        "TOOL_AGENT_ENABLED": "true",
        "TOOL_WEB_SEARCH_ENABLED": "true",
        "INTERNAL_API_TOKEN": "attacker-token",
    }):
        assert settings.tool_allowed_names == old_allowed_names
        assert settings.tool_agent_enabled == old_agent_enabled
        assert settings.tool_web_search_enabled == old_web_search_enabled
        assert settings.internal_api_token == old_internal_token


def test_review_policy_is_isolated_between_overlapping_threads(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    monkeypatch.setattr(settings, "reviewer_enabled", False)
    monkeypatch.setattr(settings, "reviewer_max_retry", 1)
    barrier = Barrier(2)

    def read(enabled, retries):
        with temporary_ai_config({"REVIEWER_ENABLED": enabled, "REVIEWER_MAX_RETRY": str(retries)}):
            barrier.wait(timeout=3)
            snapshot = current_review_config()
            assert settings.reviewer_enabled is False
            assert settings.reviewer_max_retry == 1
            barrier.wait(timeout=3)
        assert current_review_config() == (False, 1)
        return snapshot

    with ThreadPoolExecutor(2) as executor:
        first = executor.submit(read, "true", 3)
        second = executor.submit(read, "false", 0)
        assert first.result(timeout=5) == (True, 3)
        assert second.result(timeout=5) == (False, 0)


def test_nested_review_override_restores_after_error(monkeypatch):
    import pytest

    monkeypatch.setattr(settings, "reviewer_enabled", False)
    monkeypatch.setattr(settings, "reviewer_max_retry", 1)
    with temporary_ai_config({"REVIEWER_ENABLED": "true"}):
        with pytest.raises(ValueError):
            with temporary_ai_config({"REVIEWER_MAX_RETRY": "invalid"}):
                pass
        assert current_review_config() == (True, 1)
        with temporary_ai_config({"REVIEWER_MAX_RETRY": "4"}):
            assert current_review_config() == (True, 4)
        assert current_review_config() == (True, 1)
    assert current_review_config() == (False, 1)
