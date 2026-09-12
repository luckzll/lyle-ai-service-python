import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Iterator

from app.core.config import settings
from app.rag.config import RAG_FIELDS, RagConfig, default_rag_config, normalize_rag_config

logger = logging.getLogger(__name__)

ALLOWED_CHAT_MODELS = {
    ("qwen", "qwen3.7-flash"),
    ("openai", "openai/gpt-5.5"),
    ("openai", "anthropic/claude-opus-4.6"),
    ("openai", "google/gemini-3.5-flash"),
    ("qwen", "qwen3.7-plus"),
    ("qwen", "qwen3.8-max"),
    ("qwen", "deepseek-v4-flash"),
    ("qwen", "deepseek-v4-pro"),
    ("qwen", "kimi-k3"),
}

_chat_model_override: ContextVar[tuple[str, str] | None] = ContextVar(
    "lyle_chat_model_override",
    default=None,
)

# Review policy must stay stable for each stream, including concurrent turns
# with different overrides. Never mutate the process-wide review switches.
_review_override: ContextVar[tuple[bool, int] | None] = ContextVar("lyle_review_override", default=None)


def current_review_config() -> tuple[bool, int]:
    return _review_override.get() or (settings.reviewer_enabled, settings.reviewer_max_retry)


_rag_override: ContextVar[RagConfig | None] = ContextVar("lyle_rag_config", default=None)


def current_rag_config() -> RagConfig:
    return _rag_override.get() or default_rag_config()

# 允许通过请求 aiConfig 临时覆盖的配置：仅限模型行为 / RAG 质量 / 数值上限类字段。
CONFIG_FIELD_MAP = {
    "REVIEWER_ENABLED": "reviewer_enabled",
    "REVIEWER_MAX_RETRY": "reviewer_max_retry",
    "EVALUATION_JUDGE_ENABLED": "evaluation_judge_enabled",
    "TOOL_ROUTER_LLM_ENABLED": "tool_router_llm_enabled",
    "TOOL_ROUTER_CONTEXT_MAX_CHARS": "tool_router_context_max_chars",
    "RERANK_ENABLED": "rerank_enabled",
    "RERANK_PROVIDER": "rerank_provider",
    "RERANK_USE_FP16": "rerank_use_fp16",
    "RERANK_FALLBACK_ENABLED": "rerank_fallback_enabled",
    "RAG_QUERY_REWRITE_ENABLED": "rag_query_rewrite_enabled",
    "RAG_MULTI_QUERY_COUNT": "rag_multi_query_count",
    "RAG_CONTEXT_TOP_K": "rag_context_top_k",
    "RAG_SUFFICIENCY_CHECK_ENABLED": "rag_sufficiency_check_enabled",
    "RAG_OPTIMIZED_RETRIEVAL_ENABLED": "rag_optimized_retrieval_enabled",
    "RAG_PARALLEL_WORKERS": "rag_parallel_workers",
    "RAG_RETRIEVAL_CANDIDATE_LIMIT": "rag_retrieval_candidate_limit",
    "RAG_FUSION_MODE": "rag_fusion_mode",
    "RAG_ADAPTIVE_RETRIEVAL_ENABLED": "rag_adaptive_retrieval_enabled",
    "RAG_GLOBAL_RERANK_LIMIT": "rag_global_rerank_limit",
    "RAG_AUXILIARY_MODEL_ENABLED": "rag_auxiliary_model_enabled",
    "RAG_CANDIDATE_CACHE_ENABLED": "rag_candidate_cache_enabled",
    "RAG_CONTEXTUAL_QUERY_ENABLED": "rag_contextual_query_enabled",
    "TOOL_WEB_SEARCH_PROVIDER": "tool_web_search_provider",
    "TOOL_WEB_SEARCH_TOP_K": "tool_web_search_top_k",
    "TOOL_FILE_PARSE_MAX_CHARS": "tool_file_parse_max_chars",
}

# 安全相关配置（工具白名单、工具/审计开关等）禁止通过请求覆盖，只能由服务端环境变量控制。
BLOCKED_CONFIG_KEYS = {
    "TOOL_ALLOWED_NAMES",
    "TOOL_AGENT_ENABLED",
    "TOOL_AUDIT_ENABLED",
    "TOOL_DATABASE_QUERY_ENABLED",
    "TOOL_WEB_SEARCH_ENABLED",
    "TOOL_WEATHER_ENABLED",
    "TOOL_FILE_PARSE_ENABLED",
    "TOOL_FILE_PARSE_ALLOWED_DIRS",
    "TOOL_IMAGE_ANALYZE_ENABLED",
    "TOOL_IMAGE_GENERATE_ENABLED",
    "TOOL_IMAGE_EDIT_ENABLED",
    "TOOL_TODO_ENABLED",
    "TOOL_CALENDAR_ENABLED",
    "INTERNAL_API_TOKEN",
}


@contextmanager
def temporary_ai_config(config: dict[str, str] | None) -> Iterator[None]:
    config = config or {}

    old_values: dict[str, object] = {}
    model_token = None
    review_token = None
    rag_token = None
    previous_model_override = _chat_model_override.get()
    previous_review_override = _review_override.get()
    previous_rag_override = _rag_override.get()
    try:
        model_override = _validated_chat_model_override(config)
        if model_override is not None:
            model_token = _chat_model_override.set(model_override)

        review_enabled, review_retry = current_review_config()
        if "REVIEWER_ENABLED" in config or "REVIEWER_MAX_RETRY" in config:
            review_token = _review_override.set((
                _coerce_value(config["REVIEWER_ENABLED"], review_enabled) if "REVIEWER_ENABLED" in config else review_enabled,
                _coerce_value(config["REVIEWER_MAX_RETRY"], review_retry) if "REVIEWER_MAX_RETRY" in config else review_retry,
            ))

        rag_config = current_rag_config()
        rag_updates = {
            CONFIG_FIELD_MAP[key]: _coerce_value(value, getattr(rag_config, CONFIG_FIELD_MAP[key]))
            for key, value in config.items() if CONFIG_FIELD_MAP.get(key) in RAG_FIELDS
        }
        # Freeze even when no override was requested so child workers see the
        # same policy throughout this request.
        rag_token = _rag_override.set(normalize_rag_config(replace(rag_config, **rag_updates)))

        for key, value in config.items():
            if key in {"REVIEWER_ENABLED", "REVIEWER_MAX_RETRY"}:
                continue
            if key in BLOCKED_CONFIG_KEYS:
                logger.warning("Ignoring security-sensitive aiConfig key: %s", key)
                continue
            field = CONFIG_FIELD_MAP.get(key)
            if field in RAG_FIELDS:
                continue
            if not field or not hasattr(settings, field):
                continue
            old_values[field] = getattr(settings, field)
            setattr(settings, field, _coerce_value(value, old_values[field]))
        yield
    finally:
        for field, value in old_values.items():
            setattr(settings, field, value)
        if rag_token is not None:
            try:
                _rag_override.reset(rag_token)
            except ValueError:
                _rag_override.set(previous_rag_override)
        if review_token is not None:
            try:
                _review_override.reset(review_token)
            except ValueError:
                _review_override.set(previous_review_override)
        if model_token is not None:
            try:
                _chat_model_override.reset(model_token)
            except ValueError:
                # Starlette may advance a synchronous streaming generator in a
                # copied worker context. A ContextVar token can only be reset in
                # the exact context that created it, so restore the prior value
                # explicitly when the response crosses that boundary.
                _chat_model_override.set(previous_model_override)


def current_chat_model_override() -> tuple[str, str] | None:
    return _chat_model_override.get()


def _validated_chat_model_override(config: dict[str, str]) -> tuple[str, str] | None:
    provider_value = config.get("CHAT_MODEL_PROVIDER")
    model_value = config.get("CHAT_MODEL_NAME")
    if provider_value is None and model_value is None:
        return None
    if not provider_value or not model_value:
        raise ValueError("CHAT_MODEL_PROVIDER and CHAT_MODEL_NAME must be provided together")
    selected = (provider_value.strip().lower(), model_value.strip())
    if selected not in ALLOWED_CHAT_MODELS:
        raise ValueError("Unsupported request chat model")
    return selected


def _coerce_value(value: str, old_value: object) -> object:
    if isinstance(old_value, bool):
        return str(value).strip().lower() == "true"
    if isinstance(old_value, int) and not isinstance(old_value, bool):
        return int(value)
    if isinstance(old_value, float):
        return float(value)
    return value
