from contextlib import contextmanager
from typing import Iterator

from app.core.config import settings


CONFIG_FIELD_MAP = {
    "REVIEWER_ENABLED": "reviewer_enabled",
    "REVIEWER_MAX_RETRY": "reviewer_max_retry",
    "EVALUATION_JUDGE_ENABLED": "evaluation_judge_enabled",
    "TOOL_AGENT_ENABLED": "tool_agent_enabled",
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
    "TOOL_ALLOWED_NAMES": "tool_allowed_names",
    "TOOL_AUDIT_ENABLED": "tool_audit_enabled",
    "TOOL_DATABASE_QUERY_ENABLED": "tool_database_query_enabled",
    "TOOL_WEB_SEARCH_ENABLED": "tool_web_search_enabled",
    "TOOL_WEB_SEARCH_PROVIDER": "tool_web_search_provider",
    "TOOL_WEB_SEARCH_TOP_K": "tool_web_search_top_k",
    "TOOL_WEATHER_ENABLED": "tool_weather_enabled",
    "TOOL_FILE_PARSE_ENABLED": "tool_file_parse_enabled",
    "TOOL_FILE_PARSE_MAX_CHARS": "tool_file_parse_max_chars",
    "TOOL_IMAGE_ANALYZE_ENABLED": "tool_image_analyze_enabled",
    "TOOL_IMAGE_GENERATE_ENABLED": "tool_image_generate_enabled",
    "TOOL_IMAGE_EDIT_ENABLED": "tool_image_edit_enabled",
    "TOOL_TODO_ENABLED": "tool_todo_enabled",
    "TOOL_CALENDAR_ENABLED": "tool_calendar_enabled",
}


@contextmanager
def temporary_ai_config(config: dict[str, str] | None) -> Iterator[None]:
    if not config:
        yield
        return

    old_values: dict[str, object] = {}
    for key, value in config.items():
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
