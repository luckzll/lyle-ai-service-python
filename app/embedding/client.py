import hashlib
import math
from functools import lru_cache
from threading import Lock

import httpx

from app.core.config import settings


EMBEDDING_DIMENSION = settings.embedding_dimension
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


@lru_cache(maxsize=1)
def _fastembed_model():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=settings.embedding_model)


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
    with httpx.Client(timeout=settings.embedding_timeout_seconds) as client:
        response = client.post(
            f"{settings.embedding_base_url.rstrip('/')}/embeddings",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
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
