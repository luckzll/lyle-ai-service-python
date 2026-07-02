import hashlib
import math
from functools import lru_cache

import httpx

from app.core.config import settings


EMBEDDING_DIMENSION = settings.embedding_dimension


def embed_text(text: str) -> list[float]:
    provider = settings.embedding_provider.lower()
    if provider == "fastembed":
        return _embed_with_fastembed(text)
    if provider in {"openai", "openai_compatible"}:
        return _embed_with_openai_compatible(text)
    return _embed_with_hash(text, settings.embedding_dimension)


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
