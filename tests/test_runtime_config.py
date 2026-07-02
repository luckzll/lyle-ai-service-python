from app.core.config import settings
from app.core.runtime_config import temporary_ai_config


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
        assert settings.reviewer_enabled is True
        assert settings.reviewer_max_retry == 3
        assert settings.rerank_provider == "lexical"

    assert settings.reviewer_enabled == old_reviewer_enabled
    assert settings.reviewer_max_retry == old_reviewer_max_retry
    assert settings.rerank_provider == old_rerank_provider


def test_temporary_ai_config_accepts_empty_config():
    old_prompt_version = settings.prompt_version
    with temporary_ai_config({}):
        assert settings.prompt_version == old_prompt_version
    assert settings.prompt_version == old_prompt_version
