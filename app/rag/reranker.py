from functools import lru_cache
import logging
import math
import re
from typing import Any

from app.core.runtime_config import current_rag_config
from app.usage.manual import begin_external_call, estimate_text_tokens, finish_external_call

logger = logging.getLogger(__name__)


def rerank_items(
    query: str,
    items: list[dict[str, Any]],
    text_key: str,
    top_k: int,
) -> list[dict[str, Any]]:
    if not items:
        return []
    config = current_rag_config()
    if not config.rerank_enabled:
        return items[:top_k]

    try:
        if config.rerank_provider == "flag_embedding":
            return _rerank_with_flag_embedding(query, items, text_key, top_k)
        if config.rerank_provider == "fastembed":
            return _rerank_with_fastembed(query, items, text_key, top_k)
    except Exception as exc:
        if not config.rerank_fallback_enabled:
            raise
        logger.warning("Reranker failed; using lexical fallback. provider=%s error_type=%s", config.rerank_provider, type(exc).__name__)

    return _rerank_with_lexical_score(query, items, text_key, top_k)


def get_reranker_status() -> dict[str, object]:
    config = current_rag_config()
    status = {
        "enabled": config.rerank_enabled,
        "provider": config.rerank_provider,
        "model": config.rerank_model,
        "fallbackEnabled": config.rerank_fallback_enabled,
        "ready": True,
        "message": "ok",
    }
    if config.rerank_enabled and config.rerank_provider == "fastembed":
        try:
            _fastembed_reranker_model()
        except Exception as exc:
            status.update(ready=False, message="预下载 FastEmbed 重排模型后再启用。", error=type(exc).__name__)
        return status
    if not config.rerank_enabled or config.rerank_provider != "flag_embedding":
        return status
    try:
        _import_flag_reranker()
    except Exception as exc:
        status["ready"] = False
        status["message"] = (
            "FlagEmbedding is not installed. "
            "Install with: python -m pip install -r requirements/optional/reranker.txt"
        )
        status["error"] = str(exc)
    return status


def _rerank_with_flag_embedding(
    query: str,
    items: list[dict[str, Any]],
    text_key: str,
    top_k: int,
) -> list[dict[str, Any]]:
    config = current_rag_config()
    model = _flag_reranker_model()
    pairs = [[query, str(item.get(text_key) or "")] for item in items]
    estimated_input = sum(
        estimate_text_tokens(str(left)) + estimate_text_tokens(str(right))
        for left, right in pairs
    )
    ticket = begin_external_call(estimated_input_tokens=estimated_input)
    try:
        scores = model.compute_score(pairs, normalize=True)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        if not isinstance(scores, (list, tuple)):
            scores = [scores]
        scores = [float(score) for score in scores]
        if len(scores) != len(items) or not all(math.isfinite(score) for score in scores):
            raise ValueError("Reranker returned invalid candidate scores")
    except Exception as error:
        finish_external_call(
            ticket,
            call_type="RERANK",
            task_type="RERANK",
            provider=config.rerank_provider,
            model=config.rerank_model,
            input_tokens=estimated_input,
            status="FAILED",
            error_category=type(error).__name__,
        )
        raise
    finish_external_call(
        ticket,
        call_type="RERANK",
        task_type="RERANK",
        provider=config.rerank_provider,
        model=config.rerank_model,
        input_tokens=estimated_input,
        usage_source="ESTIMATED",
        estimated_cost_micros=0,
    )
    scored = []
    for item, score in zip(items, scores):
        scored.append({
            **item,
            "rerank_score": float(score),
            "rerank_backend": "flag_embedding",
        })
    return sorted(scored, key=lambda item: float(item.get("rerank_score") or 0), reverse=True)[:top_k]


def _flag_reranker_model():
    config = current_rag_config()
    return _cached_flag_reranker_model(config.rerank_model, config.rerank_use_fp16)


def _fastembed_reranker_model():
    config = current_rag_config()
    return _cached_fastembed_model(config.rerank_model, config.rerank_threads,
                                  config.rerank_cache_dir, config.rerank_local_files_only)


@lru_cache(maxsize=1)
def _cached_fastembed_model(model_name, threads, cache_dir, local_files_only):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=model_name, threads=threads, cache_dir=cache_dir or None,
                            local_files_only=local_files_only)


def _rerank_with_fastembed(query, items, text_key, top_k):
    config = current_rag_config()
    # Context is search-only: always return the original source text and offsets.
    texts = [str(item.get("retrieval_text") or item.get(text_key) or "") for item in items]
    estimated_input = sum(estimate_text_tokens(query) + estimate_text_tokens(text) for text in texts)
    ticket = begin_external_call(estimated_input_tokens=estimated_input)
    try:
        model = _fastembed_reranker_model()
        scores = [float(score) for score in model.rerank(query, texts, batch_size=config.rerank_batch_size)]
        if len(scores) != len(items) or not all(math.isfinite(score) for score in scores):
            raise ValueError("Reranker returned invalid candidate scores")
        # FastEmbed returns cross-encoder logits. Sigmoid matches FlagEmbedding's
        # normalize=True contract without numerical overflow or changing order.
        scores = [1 / (1 + math.exp(-score)) if score >= 0 else math.exp(score) / (1 + math.exp(score))
                  for score in scores]
    except Exception as error:
        finish_external_call(ticket, call_type="RERANK", task_type="RERANK", provider="fastembed",
                             model=config.rerank_model, input_tokens=estimated_input,
                             status="FAILED", error_category=type(error).__name__)
        raise
    finish_external_call(ticket, call_type="RERANK", task_type="RERANK", provider="fastembed",
                         model=config.rerank_model, input_tokens=estimated_input,
                         usage_source="ESTIMATED", estimated_cost_micros=0)
    scored = [{**item, "rerank_score": score, "rerank_backend": "fastembed"}
              for item, score in zip(items, scores)]
    return sorted(scored, key=lambda item: item["rerank_score"], reverse=True)[:top_k]


@lru_cache(maxsize=1)
def _cached_flag_reranker_model(model_name: str, use_fp16: bool):
    FlagReranker = _import_flag_reranker()

    return FlagReranker(model_name, use_fp16=use_fp16)


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
        # A keyword count and a vector similarity have different scales. The
        # retrieval prior is a small tie-breaker, never the dominant relevance score.
        original_score = max(0.0, min(1.0, _normalise_float(item.get("score"))))
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
