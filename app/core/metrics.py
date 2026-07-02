from collections import defaultdict
from threading import Lock
from time import perf_counter, time

from fastapi import Request
from fastapi.responses import Response

from app.core.config import settings
from app.rag.reranker import get_reranker_status

STARTED_AT = time()
REQUEST_DURATION_BUCKETS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
_lock = Lock()
_request_counts: dict[tuple[str, str, int], int] = defaultdict(int)
_request_duration_sum: dict[tuple[str, str], float] = defaultdict(float)
_request_duration_count: dict[tuple[str, str], int] = defaultdict(int)
_request_duration_buckets: dict[tuple[str, str, float], int] = defaultdict(int)


async def metrics_middleware(request: Request, call_next) -> Response:
    started_at = perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        elapsed = perf_counter() - started_at
        record_request_metric(request.method, _normalize_path(request.url.path), status_code, elapsed)


def record_request_metric(method: str, path: str, status_code: int, elapsed_seconds: float) -> None:
    method = method.upper()
    with _lock:
        _request_counts[(method, path, status_code)] += 1
        _request_duration_sum[(method, path)] += elapsed_seconds
        _request_duration_count[(method, path)] += 1
        for bucket in REQUEST_DURATION_BUCKETS:
            if elapsed_seconds <= bucket:
                _request_duration_buckets[(method, path, bucket)] += 1


def collect_metrics() -> dict[str, object]:
    return {
        "uptime_seconds": round(time() - STARTED_AT, 3),
        "features": {
            "redis_enabled": settings.redis_enabled,
            "postgres_enabled": settings.postgres_enabled,
            "minio_enabled": settings.minio_enabled,
            "qdrant_enabled": settings.qdrant_enabled,
            "milvus_enabled": settings.milvus_enabled,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model,
            "prompt_version": settings.prompt_version,
            "agent_graph": "multi-agent",
            "reviewer_enabled": settings.reviewer_enabled,
            "tool_agent_enabled": settings.tool_agent_enabled,
            "rerank_enabled": settings.rerank_enabled,
            "reranker": get_reranker_status(),
        },
        "models": {
            "chat": {"provider": settings.chat_model_provider, "model": settings.chat_model_name},
            "intent": {"provider": settings.intent_model_provider, "model": settings.intent_model_name},
            "memory": {"provider": settings.memory_extract_model_provider, "model": settings.memory_extract_model_name},
            "summary": {"provider": settings.summary_model_provider, "model": settings.summary_model_name},
            "judge": {"provider": settings.judge_model_provider, "model": settings.judge_model_name},
        },
        "tools": {
            "allowed_names": settings.tool_allowed_names,
            "audit_enabled": settings.tool_audit_enabled,
            "database_query_enabled": settings.tool_database_query_enabled,
            "web_search_enabled": settings.tool_web_search_enabled,
            "web_search_provider": settings.tool_web_search_provider,
            "file_parse_enabled": settings.tool_file_parse_enabled,
            "todo_enabled": settings.tool_todo_enabled,
            "calendar_enabled": settings.tool_calendar_enabled,
        },
        "limits": {
            "python_rate_limit_per_minute": settings.python_rate_limit_per_minute,
            "memory_short_term_limit": settings.memory_short_term_limit,
            "memory_compression_trigger_limit": settings.memory_compression_trigger_limit,
            "memory_recall_limit": settings.memory_recall_limit,
            "structured_memory_extractor": settings.structured_memory_extractor,
            "structured_memory_confidence_threshold": settings.structured_memory_confidence_threshold,
            "structured_memory_recent_turns": settings.structured_memory_recent_turns,
            "postgres_pool_min_size": settings.postgres_pool_min_size,
            "postgres_pool_max_size": settings.postgres_pool_max_size,
        },
        "requests": _request_metric_snapshot(),
    }


def collect_prometheus_metrics() -> str:
    metrics = collect_metrics()
    features = metrics["features"]
    limits = metrics["limits"]
    lines = [
        "# HELP lyle_ai_uptime_seconds Python AI service uptime.",
        "# TYPE lyle_ai_uptime_seconds gauge",
        f"lyle_ai_uptime_seconds {metrics['uptime_seconds']}",
        "# HELP lyle_ai_feature_enabled Feature enablement flags.",
        "# TYPE lyle_ai_feature_enabled gauge",
        _flag("redis", features["redis_enabled"]),
        _flag("postgres", features["postgres_enabled"]),
        _flag("minio", features["minio_enabled"]),
        _flag("qdrant", features["qdrant_enabled"]),
        _flag("milvus", features["milvus_enabled"]),
        _flag("reviewer", features["reviewer_enabled"]),
        _flag("tool_agent", features["tool_agent_enabled"]),
        _flag("rerank", features["rerank_enabled"]),
        "# HELP lyle_ai_limit Runtime numeric limits.",
        "# TYPE lyle_ai_limit gauge",
        _gauge("python_rate_limit_per_minute", limits["python_rate_limit_per_minute"]),
        _gauge("memory_short_term_limit", limits["memory_short_term_limit"]),
        _gauge("memory_compression_trigger_limit", limits["memory_compression_trigger_limit"]),
        _gauge("memory_recall_limit", limits["memory_recall_limit"]),
        _gauge("postgres_pool_min_size", limits["postgres_pool_min_size"]),
        _gauge("postgres_pool_max_size", limits["postgres_pool_max_size"]),
        "# HELP lyle_ai_http_requests_total Total HTTP requests.",
        "# TYPE lyle_ai_http_requests_total counter",
        *_prometheus_request_counts(),
        "# HELP lyle_ai_http_request_duration_seconds HTTP request duration histogram.",
        "# TYPE lyle_ai_http_request_duration_seconds histogram",
        *_prometheus_request_duration_histograms(),
    ]
    return "\n".join(lines) + "\n"


def _flag(name: str, enabled: object) -> str:
    return f'lyle_ai_feature_enabled{{feature="{name}"}} {1 if enabled else 0}'


def _gauge(name: str, value: object) -> str:
    return f'lyle_ai_limit{{name="{name}"}} {value}'


def _request_metric_snapshot() -> dict[str, object]:
    with _lock:
        return {
            "total": sum(_request_counts.values()),
            "by_status": {
                str(status): sum(
                    count for (_, _, item_status), count in _request_counts.items()
                    if item_status == status
                )
                for status in sorted({status for (_, _, status) in _request_counts})
            },
            "duration_count": sum(_request_duration_count.values()),
        }


def _prometheus_request_counts() -> list[str]:
    with _lock:
        items = sorted(_request_counts.items())
    return [
        (
            f'lyle_ai_http_requests_total{{method="{method}",path="{path}",'
            f'status="{status}"}} {count}'
        )
        for (method, path, status), count in items
    ]


def _prometheus_request_duration_histograms() -> list[str]:
    with _lock:
        path_keys = sorted(_request_duration_count.keys())
        bucket_items = dict(_request_duration_buckets)
        count_items = dict(_request_duration_count)
        sum_items = dict(_request_duration_sum)

    lines: list[str] = []
    for method, path in path_keys:
        cumulative = 0
        for bucket in REQUEST_DURATION_BUCKETS:
            cumulative += bucket_items.get((method, path, bucket), 0)
            lines.append(
                f'lyle_ai_http_request_duration_seconds_bucket{{method="{method}",path="{path}",le="{bucket}"}} {cumulative}'
            )
        lines.append(
            f'lyle_ai_http_request_duration_seconds_bucket{{method="{method}",path="{path}",le="+Inf"}} {count_items[(method, path)]}'
        )
        lines.append(
            f'lyle_ai_http_request_duration_seconds_count{{method="{method}",path="{path}"}} {count_items[(method, path)]}'
        )
        lines.append(
            f'lyle_ai_http_request_duration_seconds_sum{{method="{method}",path="{path}"}} {round(sum_items[(method, path)], 6)}'
        )
    return lines


def _normalize_path(path: str) -> str:
    parts = []
    for part in path.split("/"):
        if not part:
            continue
        if part.isdigit() or _looks_like_id(part):
            parts.append("{id}")
        else:
            parts.append(part)
    return "/" + "/".join(parts)


def _looks_like_id(value: str) -> bool:
    if len(value) >= 16 and all(ch.isalnum() or ch in "-_" for ch in value):
        return True
    if "-" in value and len(value) >= 8:
        return True
    return False
