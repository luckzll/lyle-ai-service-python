from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock

import pytest

from app.rag.cache import BoundedTTLCache


def test_ttl_lru_and_mutable_values_are_isolated():
    now = [0.0]
    cache = BoundedTTLCache(2, 100_000, clock=lambda: now[0])
    first = cache.get_or_compute("a", lambda: [{"position": 1}], ttl_seconds=5)
    first.value[0]["position"] = 999
    assert cache.get_or_compute("a", lambda: pytest.fail("cache miss"), ttl_seconds=5).value == [{"position": 1}]
    cache.get_or_compute("b", lambda: [2], ttl_seconds=5)
    cache.get_or_compute("a", lambda: [], ttl_seconds=5)
    cache.get_or_compute("c", lambda: [3], ttl_seconds=5)
    assert cache.stats()["entries"] == 2
    assert cache.get_or_compute("b", lambda: [20], ttl_seconds=5).status == "miss"
    now[0] = 6
    assert cache.stats()["entries"] == 0
    assert cache.get_or_compute("b", lambda: [30], ttl_seconds=5).value == [30]


def test_single_flight_shares_one_load_across_simultaneous_requests():
    cache = BoundedTTLCache(8, 100_000)
    barrier, started, release, lock = Barrier(5), Event(), Event(), Lock()
    calls = 0

    def load():
        nonlocal calls
        with lock:
            calls += 1
        started.set()
        assert release.wait(3)
        return {"vector": [1.0, 2.0]}

    def run():
        barrier.wait(3)
        return cache.get_or_compute("same", load, ttl_seconds=10)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run) for _ in range(4)]
        barrier.wait(3)
        assert started.wait(3)
        assert cache.stats()["inflight"] == 1
        release.set()
        results = [future.result(3) for future in futures]
    assert calls == 1 and cache.stats()["inflight"] == 0
    results[0].value["vector"][0] = 999
    assert results[1].value["vector"] == [1.0, 2.0]
    assert cache.get_or_compute("same", load, ttl_seconds=10).status == "hit"


def test_invalidation_detaches_inflight_and_old_result_cannot_refill():
    cache = BoundedTTLCache(2, 100_000)
    started, release = Event(), Event()

    def old_load():
        started.set()
        assert release.wait(3)
        return ["old"]

    with ThreadPoolExecutor(max_workers=1) as pool:
        old = pool.submit(lambda: cache.get_or_compute("same", old_load, ttl_seconds=10))
        assert started.wait(3)
        cache.invalidate()
        assert cache.get_or_compute("same", lambda: ["new"], ttl_seconds=10).value == ["new"]
        release.set()
        assert old.result(3).value == ["old"]
    assert cache.get_or_compute("same", lambda: pytest.fail("old load overwrote cache"), ttl_seconds=10).value == ["new"]


def test_memory_budget_empty_results_errors_and_partial_results_are_not_retained():
    cache = BoundedTTLCache(3, 1000)
    calls = []
    for _ in range(2):
        assert cache.get_or_compute("large", lambda: "x" * 2000, ttl_seconds=10).status == "miss"
        assert cache.get_or_compute("empty", lambda: [], ttl_seconds=10).status == "miss"
        assert cache.get_or_compute("partial", lambda: [1], cacheable=lambda _: False, ttl_seconds=10).status == "miss"
        with pytest.raises(ValueError):
            cache.get_or_compute("error", lambda: calls.append(1) or (_ for _ in ()).throw(ValueError("offline")), ttl_seconds=10)
    assert len(calls) == 2
    assert cache.stats() == {"entries": 0, "bytes": 0, "inflight": 0}


def test_inflight_capacity_is_bounded_and_saturation_computes_without_queueing():
    cache = BoundedTTLCache(1, 100_000)
    started, release = Event(), Event()

    def load():
        started.set()
        assert release.wait(3)
        return [1]

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(lambda: cache.get_or_compute("slow", load, ttl_seconds=10))
        assert started.wait(3)
        result = cache.get_or_compute("other", lambda: [2], ttl_seconds=10)
        assert result.status == "bypass" and result.value == [2]
        assert cache.stats()["inflight"] == 1
        release.set()
        future.result(3)


def test_validity_changes_during_load_prevent_cache_population():
    cache = BoundedTTLCache(2, 100_000)
    valid = [True]

    def load():
        valid[0] = False
        return [1]

    assert cache.get_or_compute("key", load, ttl_seconds=10, valid=lambda: valid[0]).value == [1]
    assert cache.stats()["entries"] == 0


def test_explicit_nullable_values_can_be_cached():
    cache = BoundedTTLCache(2, 100_000)
    cache.get_or_compute("key", lambda: None, ttl_seconds=10, cacheable=lambda _: True)
    assert cache.get_or_compute("key", lambda: pytest.fail("miss"), ttl_seconds=10).value is None


def test_cache_copy_error_does_not_fail_a_successful_retrieval():
    class Uncopyable(dict):
        def __deepcopy__(self, memo):
            raise ValueError("cannot copy this metadata")

    cache = BoundedTTLCache(2, 100_000)
    result = cache.get_or_compute("key", lambda: Uncopyable(answer="available"), ttl_seconds=10)
    assert result.value["answer"] == "available" and result.status == "bypass"
    assert cache.stats() == {"entries": 0, "bytes": 0, "inflight": 0}
