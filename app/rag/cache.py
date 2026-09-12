"""Process-local, bounded caches. They never establish authorization or freshness."""
from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from sys import getsizeof
from threading import Event, RLock, get_ident
from time import monotonic
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class CacheResult(Generic[T]):
    value: T
    status: str  # hit, shared, miss or bypass


@dataclass
class _Entry:
    value: Any
    expires_at: float
    size: int


@dataclass
class _Flight:
    epoch: int
    owner: int = field(default_factory=get_ident)
    done: Event = field(default_factory=Event)
    value: Any = None
    error: BaseException | None = None
    shareable: bool = True


def estimated_size(value: Any) -> int:
    """Count Python container storage, rather than pretending JSON bytes equal RAM."""
    seen: set[int] = set()

    def size(item: Any) -> int:
        if id(item) in seen:
            return 0
        seen.add(id(item))
        own = getsizeof(item)
        if isinstance(item, Mapping):
            return own + sum(size(key) + size(child) for key, child in item.items())
        if isinstance(item, (list, tuple, set, frozenset)):
            return own + sum(size(child) for child in item)
        return own

    return size(value)


class BoundedTTLCache:
    """TTL/LRU cache with bounded flights and no retained exceptions.

    Invalidating detaches every in-flight load. It may finish for its original
    callers, but cannot refill the cache or be joined by a subsequent request.
    Values are copied on both sides so downstream citation edits cannot poison
    another request. No loader runs while the cache lock is held.
    """

    def __init__(self, max_entries: int, max_bytes: int, *, clock: Callable[[], float] = monotonic):
        self._lock = RLock()
        self._entries: OrderedDict[Hashable, _Entry] = OrderedDict()
        self._flights: dict[Hashable, _Flight] = {}
        self._bytes = 0
        self._epoch = 0
        self._clock = clock
        self.configure(max_entries=max_entries, max_bytes=max_bytes)

    def configure(self, *, max_entries: int, max_bytes: int) -> None:
        with self._lock:
            self._max_entries = max(0, int(max_entries))
            self._max_bytes = max(0, int(max_bytes))
            self._trim()

    def invalidate(self, predicate: Callable[[Hashable], bool] | None = None) -> None:
        with self._lock:
            self._epoch += 1
            self._flights.clear()
            for key in list(self._entries):
                if predicate is None or predicate(key):
                    self._bytes -= self._entries.pop(key).size

    def stats(self) -> dict[str, int]:
        with self._lock:
            self._expire()
            return {"entries": len(self._entries), "bytes": self._bytes, "inflight": len(self._flights)}

    def get_or_compute(
        self,
        key: Hashable,
        loader: Callable[[], T],
        *,
        ttl_seconds: float,
        cacheable: Callable[[T], bool] = bool,
        valid: Callable[[], bool] = lambda: True,
    ) -> CacheResult[T]:
        if ttl_seconds <= 0 or not valid():
            return CacheResult(loader(), "bypass")
        flight = None
        owner = False
        hit = None
        hit_found = False
        with self._lock:
            self._expire()
            if self._max_entries <= 0 or self._max_bytes <= 0:
                bypass = True
            elif key in self._entries:
                self._entries.move_to_end(key)
                try:
                    hit = deepcopy(self._entries[key].value)
                    hit_found = True
                    bypass = False
                except Exception:
                    self._bytes -= self._entries.pop(key).size
                    bypass = True
            else:
                flight = self._flights.get(key)
                if flight is None and len(self._flights) < self._max_entries:
                    flight = _Flight(self._epoch)
                    self._flights[key] = flight
                    owner = True
                # Saturation or recursively joining our own loader must not queue.
                bypass = flight is None or (flight.owner == get_ident() and not owner)
        if hit_found:
            return CacheResult(hit, "hit") if valid() else CacheResult(loader(), "bypass")
        if bypass:
            return CacheResult(loader(), "bypass")
        if not owner:
            flight.done.wait()
            if flight.error is not None:
                raise flight.error
            if not flight.shareable:
                return CacheResult(loader(), "bypass")
            # A caller with a changed authorization/version must use a fresh load.
            return CacheResult(deepcopy(flight.value), "shared") if valid() else CacheResult(loader(), "bypass")
        try:
            value = loader()
            try:
                should_store = cacheable(value) and valid()
                snapshot = deepcopy(value)
                entry_size = estimated_size((key, snapshot)) + getsizeof(_Entry(snapshot, 0, 0))
            except Exception:
                # Cache bookkeeping cannot turn a successful provider response
                # into a failed answer. Uncopyable values must not be shared.
                flight.shareable = False
                return CacheResult(value, "bypass")
            with self._lock:
                current = self._flights.get(key)
                if (should_store and current is flight and flight.epoch == self._epoch
                        and entry_size <= self._max_bytes and self._max_entries > 0):
                    if key in self._entries:
                        self._bytes -= self._entries.pop(key).size
                    self._entries[key] = _Entry(snapshot, self._clock() + ttl_seconds, entry_size)
                    self._bytes += entry_size
                    self._trim()
                flight.value = snapshot
            return CacheResult(value, "miss")
        except BaseException as error:
            flight.error = error
            raise
        finally:
            with self._lock:
                if self._flights.get(key) is flight:
                    self._flights.pop(key, None)
                flight.done.set()

    def _expire(self) -> None:
        now = self._clock()
        for key, entry in list(self._entries.items()):
            if entry.expires_at <= now:
                self._bytes -= self._entries.pop(key).size

    def _trim(self) -> None:
        while self._entries and (len(self._entries) > self._max_entries or self._bytes > self._max_bytes):
            _, entry = self._entries.popitem(last=False)
            self._bytes -= entry.size
