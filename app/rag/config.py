"""Immutable per-request retrieval policy, shared by parallel query workers."""
from dataclasses import dataclass, fields, replace

from app.core.config import settings


@dataclass(frozen=True)
class RagConfig:
    rerank_enabled: bool
    rerank_provider: str
    rerank_model: str
    rerank_threads: int
    rerank_batch_size: int
    rerank_cache_dir: str
    rerank_local_files_only: bool
    rerank_use_fp16: bool
    rerank_fallback_enabled: bool
    rerank_candidate_limit: int
    rag_query_rewrite_enabled: bool
    rag_multi_query_count: int
    rag_context_top_k: int
    rag_parent_child_expand_enabled: bool
    rag_parent_child_sibling_limit: int
    rag_sufficiency_check_enabled: bool
    rag_optimized_retrieval_enabled: bool
    rag_parallel_workers: int
    rag_retrieval_candidate_limit: int
    rag_fusion_mode: str
    rag_adaptive_retrieval_enabled: bool
    rag_global_rerank_limit: int
    rag_auxiliary_model_enabled: bool
    rag_candidate_cache_enabled: bool
    rag_candidate_cache_ttl_seconds: int
    rag_candidate_cache_max_entries: int
    rag_candidate_cache_max_bytes: int
    rag_contextual_query_enabled: bool
    rag_context_history_turns: int
    rag_context_history_max_chars: int


RAG_FIELDS = frozenset(field.name for field in fields(RagConfig))


def normalize_rag_config(config: RagConfig) -> RagConfig:
    if config.rag_fusion_mode not in {"legacy", "rrf"}:
        raise ValueError("Unsupported RAG fusion mode")
    if config.rerank_provider not in {"lexical", "flag_embedding", "fastembed"}:
        raise ValueError("Unsupported RAG reranker provider")
    # These are resource bounds, never an end-to-end deadline.
    top_k = max(1, min(50, config.rag_context_top_k))
    return replace(config,
        rag_multi_query_count=max(1, min(4, config.rag_multi_query_count)),
        rag_context_top_k=top_k,
        rag_parallel_workers=max(1, min(8, config.rag_parallel_workers)),
        rag_retrieval_candidate_limit=max(1, min(100, config.rag_retrieval_candidate_limit)),
        rerank_candidate_limit=max(1, min(100, config.rerank_candidate_limit)),
        rag_parent_child_sibling_limit=max(0, min(5, config.rag_parent_child_sibling_limit)),
        rag_global_rerank_limit=max(top_k, min(100, config.rag_global_rerank_limit)),
        rag_candidate_cache_ttl_seconds=max(1, min(300, config.rag_candidate_cache_ttl_seconds)),
        rag_candidate_cache_max_entries=max(1, min(1024, config.rag_candidate_cache_max_entries)),
        rag_candidate_cache_max_bytes=max(1024, min(64 * 1024 * 1024, config.rag_candidate_cache_max_bytes)),
        rag_context_history_turns=max(1, min(12, config.rag_context_history_turns)),
        rag_context_history_max_chars=max(128, min(4000, config.rag_context_history_max_chars)),
    )


def default_rag_config() -> RagConfig:
    return normalize_rag_config(RagConfig(**{name: getattr(settings, name) for name in RAG_FIELDS}))
