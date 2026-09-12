import hashlib
import math
from functools import lru_cache
from threading import Lock

import httpx

from app.core.config import settings
from app.rag.cache import BoundedTTLCache
from app.rag.cache_scope import current_retrieval_cache_scope, current_retrieval_user
from app.usage.manual import begin_external_call, estimate_text_tokens, finish_external_call


EMBEDDING_DIMENSION = settings.embedding_dimension
_query_cache = BoundedTTLCache(max_entries=256, max_bytes=8 * 1024 * 1024)
_readiness_lock = Lock()
_readiness = {
    "ready": False,
    "provider": settings.embedding_provider.lower(),
    "model": settings.embedding_model,
    "message": "向量模型尚未完成初始化。",
}


def embed_text(text: str) -> list[float]:
    provider = settings.embedding_provider.lower()
    try:
        if provider == "fastembed":
            vector = _embed_with_fastembed(text)
        elif provider in {"openai", "openai_compatible"}:
            vector = _embed_with_openai_compatible(text)
        else:
            vector = _embed_with_hash(text, settings.embedding_dimension)
    except Exception as error:
        _set_embedding_readiness(False, f"向量服务初始化失败：{type(error).__name__}")
        raise
    _set_embedding_readiness(True, "向量服务已就绪。")
    return vector


def embedding_identity() -> tuple[str, ...]:
    """Endpoint and credentials partition cache entries without retaining secrets."""
    return (
        settings.embedding_provider.lower(), settings.embedding_model, str(settings.embedding_dimension),
        hashlib.sha256(settings.embedding_base_url.rstrip("/").encode("utf-8")).hexdigest(),
        hashlib.sha256(settings.embedding_api_key.encode("utf-8")).hexdigest(),
    )


def query_cache_identity() -> tuple[str, ...] | None:
    # Only internal request scopes can establish this user identity. A cache
    # without it would turn a pure helper into an accidental cross-user store.
    trusted = current_retrieval_cache_scope()
    principal = (trusted.user_id, trusted.conversation_id) if trusted else current_retrieval_user()
    from app.rag.query_context import current_query_context

    context = current_query_context()
    if principal is None and context is not None and context.user_id:
        principal = (context.user_id, context.conversation_id)
    if principal is None:
        return None
    # Do not combine a different user's history with the established principal.
    history = context.history_fingerprint if context is not None and context.user_id == principal[0] else ""
    conversation = str(principal[1] or (context.conversation_id if context is not None and context.user_id == principal[0] else ""))
    return str(principal[0]), conversation, str(history or "")


def embed_query(text: str) -> list[float]:
    """Cache query vectors only; document ingestion and memory writes use embed_text."""
    principal = query_cache_identity()
    if not settings.embedding_cache_enabled or not text or principal is None:
        return embed_text(text)
    identity = embedding_identity()
    key = (principal, identity, hashlib.sha256(text.encode("utf-8")).hexdigest())
    _query_cache.configure(max_entries=settings.embedding_cache_max_entries, max_bytes=settings.embedding_cache_max_bytes)

    def cacheable(vector: list[float]) -> bool:
        return (len(vector) == int(identity[2]) and bool(vector)
                and all(isinstance(value, (int, float)) and math.isfinite(value) for value in vector))

    return _query_cache.get_or_compute(
        key, lambda: embed_text(text), ttl_seconds=settings.embedding_cache_ttl_seconds,
        cacheable=cacheable, valid=lambda: embedding_identity() == identity,
    ).value


def clear_embedding_query_cache() -> None:
    _query_cache.invalidate()


def warm_up_embedding_model() -> dict[str, object]:
    """Load local model assets before the service receives user traffic."""
    provider = settings.embedding_provider.lower()
    try:
        if provider == "fastembed":
            _embed_with_fastembed("Lyle 服务就绪检查")
        elif provider in {"openai", "openai_compatible"}:
            if not settings.embedding_api_key:
                raise ValueError("EMBEDDING_API_KEY is required for OpenAI-compatible embeddings.")
        else:
            _embed_with_hash("Lyle 服务就绪检查", settings.embedding_dimension)
    except Exception as error:
        _set_embedding_readiness(False, f"向量模型预热失败：{type(error).__name__}")
        raise
    _set_embedding_readiness(True, "向量模型已加载。")
    return get_embedding_readiness()


def get_embedding_readiness() -> dict[str, object]:
    with _readiness_lock:
        return dict(_readiness)


def _set_embedding_readiness(ready: bool, message: str) -> None:
    with _readiness_lock:
        _readiness.update({
            "ready": ready,
            "provider": settings.embedding_provider.lower(),
            "model": settings.embedding_model,
            "message": message,
        })


def _fastembed_model():
    return _cached_fastembed_model(settings.embedding_model)


@lru_cache(maxsize=1)
def _cached_fastembed_model(model_name: str):
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model_name)


def _embed_with_fastembed(text: str) -> list[float]:
    vector = next(_fastembed_model().embed([text or ""]))
    return [float(value) for value in vector]


def _embed_with_openai_compatible(text: str) -> list[float]:
    if not settings.embedding_api_key:
        raise ValueError("EMBEDDING_API_KEY is required for OpenAI-compatible embeddings.")

    headers = {
        "Authorization": f"Bearer {settings.embedding_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.embedding_model,
        "input": text or "",
    }
    estimated_input = estimate_text_tokens(text)
    estimated_cost = round(estimated_input * settings.ai_usage_embedding_price_cny_per_million)
    ticket = begin_external_call(
        estimated_input_tokens=estimated_input,
        estimated_cost_micros=estimated_cost,
    )
    try:
        with httpx.Client(timeout=settings.embedding_timeout_seconds) as client:
            response = client.post(
                f"{settings.embedding_base_url.rstrip('/')}/embeddings",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
    except Exception as error:
        finish_external_call(
            ticket,
            call_type="EMBEDDING",
            task_type="EMBEDDING",
            provider=settings.embedding_provider,
            model=settings.embedding_model,
            input_tokens=estimated_input,
            estimated_cost_micros=estimated_cost,
            status="FAILED",
            error_category=type(error).__name__,
        )
        raise
    usage = data.get("usage") if isinstance(data, dict) else {}
    provider_tokens = int((usage or {}).get("total_tokens") or (usage or {}).get("prompt_tokens") or 0)
    finish_external_call(
        ticket,
        call_type="EMBEDDING",
        task_type="EMBEDDING",
        provider=settings.embedding_provider,
        model=settings.embedding_model,
        input_tokens=provider_tokens or estimated_input,
        usage_source="PROVIDER_REPORTED" if provider_tokens else "ESTIMATED",
        estimated_cost_micros=round(
            (provider_tokens or estimated_input) * settings.ai_usage_embedding_price_cny_per_million
        ),
    )
    return [float(value) for value in data["data"][0]["embedding"]]


def _embed_with_hash(text: str, dimension: int) -> list[float]:
    vector = [0.0 for _ in range(dimension)]
    tokens = [token.strip() for token in text.lower().replace("\n", " ").split(" ") if token.strip()]
    if not tokens:
        tokens = [text[:64] or "empty"]

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]
