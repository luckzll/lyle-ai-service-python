"""Request authorization/version context supplied only by the internal API.

The values here are not proof by themselves: Java must validate ownership and
build the manifest from its database, never forward a browser-provided scope.
Without that complete contract candidate caching stays disabled.
"""
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
import json
from threading import Lock
from types import MappingProxyType
from typing import Any, Iterator

from app.rag.cache import BoundedTTLCache


@dataclass(frozen=True)
class RetrievalCacheScope:
    user_id: str
    knowledge_base_ids: tuple[str, ...]
    knowledge_base_revisions: tuple[tuple[str, str], ...]
    authorization_revision: str
    allowed_document_ids: tuple[tuple[str, tuple[str, ...]], ...]
    document_revisions: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
    conversation_id: str = ""
    _document_ids_by_kb: Mapping[str, frozenset[str]] = field(init=False, compare=False, hash=False, repr=False)
    _document_revisions_by_kb: Mapping[str, Mapping[str, str]] = field(init=False, compare=False, hash=False, repr=False)
    _fingerprint: str = field(init=False, compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        # Store loops authorize every candidate. Build immutable lookup views
        # once per request rather than copying every KB's documents per chunk.
        object.__setattr__(self, "_document_ids_by_kb", MappingProxyType({
            kb: frozenset(documents) for kb, documents in self.allowed_document_ids
        }))
        object.__setattr__(self, "_document_revisions_by_kb", MappingProxyType({
            kb: MappingProxyType(dict(revisions)) for kb, revisions in self.document_revisions
        }))
        object.__setattr__(self, "_fingerprint", hashlib.sha256(json.dumps([
            self.user_id, self.conversation_id, self.knowledge_base_ids,
            self.knowledge_base_revisions, self.authorization_revision, self.allowed_document_ids, self.document_revisions,
        ], ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest())

    def fingerprint(self) -> str:
        return self._fingerprint


_scope: ContextVar[RetrievalCacheScope | None] = ContextVar("retrieval_cache_scope", default=None)
_user: ContextVar[tuple[str, str] | None] = ContextVar("retrieval_query_user", default=None)
candidate_cache = BoundedTTLCache(max_entries=128, max_bytes=8 * 1024 * 1024)
_NO_DOCUMENT_IDS: frozenset[str] = frozenset()
_NO_DOCUMENT_REVISIONS: Mapping[str, str] = MappingProxyType({})


def current_retrieval_cache_scope() -> RetrievalCacheScope | None:
    return _scope.get()


def current_retrieval_user() -> tuple[str, str] | None:
    return _user.get()


def allowed_document_ids(knowledge_base_id: str) -> frozenset[str] | None:
    scope = _scope.get()
    if scope is None:
        return None
    # A trusted scope that does not grant this KB authorizes no documents.
    return scope._document_ids_by_kb.get(str(knowledge_base_id), _NO_DOCUMENT_IDS)


def allowed_document_revisions(knowledge_base_id: str) -> Mapping[str, str] | None:
    scope = _scope.get()
    if scope is None:
        return None
    return scope._document_revisions_by_kb.get(str(knowledge_base_id), _NO_DOCUMENT_REVISIONS)


@contextmanager
def retrieval_cache_scope(
    user_id: str,
    knowledge_base_ids: list[str],
    revisions: Mapping[str, str] | None = None,
    authorization_revision: str | None = None,
    conversation_id: str = "",
    allowed_document_ids: Mapping[str, Sequence[str]] | None = None,
    document_revisions: Mapping[str, Mapping[str, str]] | None = None,
) -> Iterator[RetrievalCacheScope | None]:
    user = str(user_id).strip() if user_id is not None else ""
    conversation = str(conversation_id or "")
    scope = _validated_scope(user, knowledge_base_ids, revisions, authorization_revision, conversation, allowed_document_ids, document_revisions)
    token = _scope.set(scope)
    user_token = _user.set((user, conversation) if user else None)
    try:
        yield scope
    finally:
        _scope.reset(token)
        _user.reset(user_token)


def _validated_scope(user, kb_ids, revisions, authorization_revision, conversation_id, allowed, document_revisions):
    if (not user or not kb_ids or not isinstance(revisions, Mapping) or not isinstance(allowed, Mapping)
            or not isinstance(document_revisions, Mapping)):
        return None
    if not isinstance(authorization_revision, str) or not authorization_revision.strip():
        return None
    keys = tuple(sorted(set(str(kb).strip() for kb in kb_ids)))
    if "" in keys or set(revisions) != set(keys) or set(allowed) != set(keys) or set(document_revisions) != set(keys):
        return None
    revision_pairs, document_pairs, document_version_pairs = [], [], []
    for kb in keys:
        revision, documents = revisions[kb], allowed[kb]
        if not isinstance(revision, str) or not revision.strip():
            return None
        if not isinstance(documents, (list, tuple)) or any(not isinstance(doc, str) or not doc.strip() for doc in documents):
            return None
        versions = document_revisions[kb]
        if not isinstance(versions, Mapping) or any(
            not isinstance(versions.get(doc), str) or not versions[doc].strip() for doc in documents
        ):
            return None
        revision_pairs.append((kb, revision))
        document_pairs.append((kb, tuple(sorted(set(documents)))))
        document_version_pairs.append((kb, tuple(sorted((doc, versions[doc]) for doc in set(documents)))))
    return RetrievalCacheScope(user, keys, tuple(revision_pairs), authorization_revision, tuple(document_pairs), tuple(document_version_pairs), conversation_id)


@dataclass
class CandidateCacheAttempt:
    _healthy: bool = True
    _lock: Lock = field(default_factory=Lock)

    @property
    def healthy(self) -> bool:
        with self._lock:
            return self._healthy

    def fail(self) -> None:
        with self._lock:
            self._healthy = False


_attempt: ContextVar[CandidateCacheAttempt | None] = ContextVar("candidate_cache_attempt", default=None)


@contextmanager
def candidate_cache_attempt() -> Iterator[CandidateCacheAttempt]:
    attempt = CandidateCacheAttempt()
    token = _attempt.set(attempt)
    try:
        yield attempt
    finally:
        _attempt.reset(token)


def mark_candidate_uncacheable() -> None:
    attempt = _attempt.get()
    if attempt is not None:
        attempt.fail()


def invalidate_candidate_cache(_knowledge_base_ids: Any = None) -> None:
    # Conservative invalidation also detaches all in-flight loads. This is a
    # local safety hook; cross-process correctness comes from the trusted DB
    # manifest in every request, not from this in-memory invalidation.
    candidate_cache.invalidate()
