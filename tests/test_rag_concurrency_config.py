from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from threading import Barrier, BoundedSemaphore, Event, get_ident

import pytest

from app.core.config import settings
from app.core.runtime_config import current_rag_config, temporary_ai_config
from app.rag import concurrency


def test_bounded_map_overlaps_work_preserves_order_and_request_context():
    context = ContextVar("test_rag_context", default="none")
    barrier = Barrier(2)

    def operation(value):
        assert context.get() == "current-request"
        barrier.wait(timeout=3)
        return value * 2

    token = context.set("current-request")
    try:
        assert concurrency.bounded_map(operation, [3, 1], max_workers=2) == [6, 2]
    finally:
        context.reset(token)


def test_nested_parallel_work_stays_in_parent_worker_without_deadlock():
    barrier = Barrier(2)

    def outer(value):
        worker_id = get_ident()
        barrier.wait(timeout=3)
        inner = concurrency.bounded_map(lambda item: (get_ident(), item), [value, value + 1], max_workers=2)
        assert all(thread_id == worker_id for thread_id, _ in inner)
        return [item for _, item in inner]

    assert concurrency.bounded_map(outer, [1, 3], max_workers=2) == [[1, 2], [3, 4]]


def test_saturated_executor_falls_back_inline_without_extra_threads(monkeypatch):
    slots = BoundedSemaphore(1)
    slots.acquire()
    monkeypatch.setattr(concurrency, "_slots", slots)
    caller = get_ident()
    try:
        assert concurrency.bounded_map(lambda value: (get_ident(), value), [1, 2], max_workers=2) == [(caller, 1), (caller, 2)]
    finally:
        slots.release()


def test_failed_worker_releases_slot_and_does_not_pollute_next_context(monkeypatch):
    slots = BoundedSemaphore(1)
    monkeypatch.setattr(concurrency, "_slots", slots)

    def fail(value):
        if value == 1:
            raise ValueError("controlled failure")
        return value

    with pytest.raises(ValueError, match="controlled failure"):
        concurrency.bounded_map(fail, [1, 2], max_workers=2)
    assert slots.acquire(timeout=3)
    slots.release()
    assert not concurrency._inside_worker.get()
    assert concurrency.bounded_map(lambda value: value + 1, [1, 2], max_workers=2) == [2, 3]


def test_concurrent_rag_overrides_do_not_modify_process_defaults(monkeypatch):
    monkeypatch.setattr(settings, "rag_multi_query_count", 3)
    monkeypatch.setattr(settings, "rerank_enabled", False)
    barrier = Barrier(2)

    def run(count, enabled):
        with temporary_ai_config({"RAG_MULTI_QUERY_COUNT": str(count), "RERANK_ENABLED": enabled}):
            barrier.wait(timeout=3)
            config = current_rag_config()
            assert settings.rag_multi_query_count == 3
            assert settings.rerank_enabled is False
            child = concurrency.bounded_map(lambda _: current_rag_config(), [0, 1], max_workers=2)
            assert child == [config, config]
            barrier.wait(timeout=3)
        assert current_rag_config().rag_multi_query_count == 3
        return config.rag_multi_query_count, config.rerank_enabled

    with ThreadPoolExecutor(2) as pool:
        one = pool.submit(run, 1, "true")
        two = pool.submit(run, 4, "false")
        assert one.result(timeout=5) == (1, True)
        assert two.result(timeout=5) == (4, False)


def test_rag_resource_settings_are_bounded_and_unknown_fusion_rejected():
    with temporary_ai_config({"RAG_MULTI_QUERY_COUNT": "10000", "RAG_PARALLEL_WORKERS": "999",
                              "RAG_RETRIEVAL_CANDIDATE_LIMIT": "99999", "RAG_CONTEXT_TOP_K": "0"}):
        config = current_rag_config()
        assert (config.rag_multi_query_count, config.rag_parallel_workers, config.rag_retrieval_candidate_limit, config.rag_context_top_k) == (4, 8, 100, 1)
    with pytest.raises(ValueError, match="fusion"):
        with temporary_ai_config({"RAG_FUSION_MODE": "unknown"}):
            pass


def test_rag_policy_nested_override_restores_and_defaults_are_conservative(monkeypatch):
    monkeypatch.setattr(settings, "rag_fusion_mode", "legacy")
    monkeypatch.setattr(settings, "rag_adaptive_retrieval_enabled", False)
    assert not current_rag_config().rag_adaptive_retrieval_enabled
    with temporary_ai_config({"RAG_FUSION_MODE": "rrf"}):
        assert current_rag_config().rag_fusion_mode == "rrf"
        with temporary_ai_config({"RAG_ADAPTIVE_RETRIEVAL_ENABLED": "true"}):
            assert current_rag_config().rag_adaptive_retrieval_enabled
            assert current_rag_config().rag_fusion_mode == "rrf"
        assert not current_rag_config().rag_adaptive_retrieval_enabled
    assert current_rag_config().rag_fusion_mode == "legacy"


def test_untrusted_request_cannot_select_a_downloaded_reranker_model(monkeypatch):
    monkeypatch.setattr(settings, "rerank_model", "approved-model")
    with temporary_ai_config({"RERANK_MODEL": "untrusted/model"}):
        assert current_rag_config().rerank_model == "approved-model"


@pytest.mark.parametrize("top_k,limit,expected", [(100000, 40, 50), (5, 100000, 100), (5, -1, 5)])
def test_global_rerank_budget_cannot_escape_normalized_resource_bounds(top_k, limit, expected):
    with temporary_ai_config({"RAG_CONTEXT_TOP_K": str(top_k), "RAG_GLOBAL_RERANK_LIMIT": str(limit)}):
        config = current_rag_config()
        assert config.rag_context_top_k <= config.rag_global_rerank_limit <= 100
        assert config.rag_global_rerank_limit == expected


@pytest.mark.parametrize("overrides", [None, {}])
def test_requests_without_overrides_still_freeze_rag_policy(monkeypatch, overrides):
    monkeypatch.setattr(settings, "rag_parallel_workers", 2)
    with temporary_ai_config(overrides):
        snapshot = current_rag_config()
        monkeypatch.setattr(settings, "rag_parallel_workers", 1)
        assert current_rag_config() is snapshot
        assert current_rag_config().rag_parallel_workers == 2
    assert current_rag_config().rag_parallel_workers == 1
