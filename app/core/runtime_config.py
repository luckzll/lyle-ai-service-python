import logging
from contextlib import contextmanager
from typing import Iterator

from app.core.config import settings

logger = logging.getLogger(__name__)

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
    if not config:
        yield
        return

    old_values: dict[str, object] = {}
    for key, value in config.items():
        if key in BLOCKED_CONFIG_KEYS:
            logger.warning("Ignoring security-sensitive aiConfig key: %s", key)
            continue
        field = CONFIG_FIELD_MAP.get(key)
        if not field or not hasattr(settings, field):
            continue
        old_values[field] = getattr(settings, field)
        setattr(settings, field, _coerce_value(value, old_values[field]))

    try:
        yield
    finally:
        for field, value in old_values.items():
            setattr(settings, field, value)


def _coerce_value(value: str, old_value: object) -> object:
    if isinstance(old_value, bool):
        return str(value).strip().lower() == "true"
    if isinstance(old_value, int) and not isinstance(old_value, bool):
        return int(value)
    if isinstance(old_value, float):
        return float(value)
    return value
