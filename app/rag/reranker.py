from functools import lru_cache
import logging
import math
import re
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


def rerank_items(
    query: str,
    items: list[dict[str, Any]],
    text_key: str,
    top_k: int,
) -> list[dict[str, Any]]:
    if not items:
        return []
    if not settings.rerank_enabled:
        return items[:top_k]

    try:
        if settings.rerank_provider == "flag_embedding":
            return _rerank_with_flag_embedding(query, items, text_key, top_k)
    except Exception as exc:
        if not settings.rerank_fallback_enabled:
            raise
        logger.exception("Reranker failed, falling back to lexical rerank. provider=%s", settings.rerank_provider)

    return _rerank_with_lexical_score(query, items, text_key, top_k)


def get_reranker_status() -> dict[str, object]:
    status = {
        "enabled": settings.rerank_enabled,
        "provider": settings.rerank_provider,
        "model": settings.rerank_model,
        "fallbackEnabled": settings.rerank_fallback_enabled,
        "ready": True,
        "message": "ok",
    }
    if not settings.rerank_enabled or settings.rerank_provider != "flag_embedding":
        return status
    try:
        _import_flag_reranker()
    except Exception as exc:
        status["ready"] = False
        status["message"] = (
            "FlagEmbedding is not installed. "
            "Install with: python -m pip install -r requirements-reranker.txt"
        )
        status["error"] = str(exc)
    return status


def _rerank_with_flag_embedding(
    query: str,
    items: list[dict[str, Any]],
    text_key: str,
    top_k: int,
) -> list[dict[str, Any]]:
    model = _flag_reranker_model()
    pairs = [[query, str(item.get(text_key) or "")] for item in items]
    scores = model.compute_score(pairs, normalize=True)
    if not isinstance(scores, list):
        scores = [float(scores)]

    scored = []
    for item, score in zip(items, scores):
        scored.append({
            **item,
            "rerank_score": float(score),
            "rerank_backend": "flag_embedding",
        })
    return sorted(scored, key=lambda item: float(item.get("rerank_score") or 0), reverse=True)[:top_k]


@lru_cache(maxsize=1)
def _flag_reranker_model():
    FlagReranker = _import_flag_reranker()

    return FlagReranker(settings.rerank_model, use_fp16=settings.rerank_use_fp16)


def _import_flag_reranker():
    from FlagEmbedding import FlagReranker

    return FlagReranker


def _rerank_with_lexical_score(
    query: str,
    items: list[dict[str, Any]],
    text_key: str,
    top_k: int,
) -> list[dict[str, Any]]:
    query_tokens = _tokens(query)
    scored = []
    for index, item in enumerate(items):
        text = str(item.get(text_key) or "")
        score = _lexical_score(query_tokens, _tokens(text))
        original_score = _normalise_float(item.get("score"))
        scored.append({
            **item,
            "rerank_score": score + original_score * 0.05,
            "rerank_backend": "lexical",
            "_original_index": index,
        })

    ranked = sorted(
        scored,
        key=lambda item: (
            float(item.get("rerank_score") or 0),
            -int(item.get("_original_index") or 0),
        ),
        reverse=True,
    )
    return [
        {key: value for key, value in item.items() if key != "_original_index"}
        for item in ranked[:top_k]
    ]


def _tokens(text: str) -> set[str]:
    lowered = text.lower()
    ascii_words = set(re.findall(r"[a-z0-9_]{2,}", lowered))
    cjk_chars = set(re.findall(r"[\u4e00-\u9fff]", lowered))
    cjk_bigrams = {
        lowered[index:index + 2]
        for index in range(max(len(lowered) - 1, 0))
        if re.fullmatch(r"[\u4e00-\u9fff]{2}", lowered[index:index + 2])
    }
    return ascii_words | cjk_chars | cjk_bigrams


def _lexical_score(query_tokens: set[str], text_tokens: set[str]) -> float:
    if not query_tokens or not text_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens)
    precision = overlap / len(text_tokens)
    recall = overlap / len(query_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _normalise_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(number) or math.isinf(number):
        return 0.0
    return number
