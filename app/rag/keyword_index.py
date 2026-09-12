"""Versioned in-process keyword postings with the existing lexical score formula.

The store owns the lock: synchronize and take snapshots while holding it, then
score the immutable snapshots outside it. This index adds no storage authority.
"""
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import math

from app.rag.chunk_context import keyword_text


class VersionedChunkList(list):
    """List-compatible storage whose ordinary mutations invalidate the index."""
    def __init__(self, values=(), *, lock):
        super().__init__(values)
        self.version = 0
        self._lock = lock

    def _mutate(self, method, *args, **kwargs):
        with self._lock:
            try:
                return method(*args, **kwargs)
            finally:
                # Even a partially failed extend/sort may have changed a list.
                self.version += 1

    def __setitem__(self, key, value):
        return self._mutate(super().__setitem__, key, value)

    def __delitem__(self, key):
        return self._mutate(super().__delitem__, key)

    def append(self, value):
        return self._mutate(super().append, value)

    def extend(self, values):
        return self._mutate(super().extend, values)

    def insert(self, index, value):
        return self._mutate(super().insert, index, value)

    def pop(self, index=-1):
        return self._mutate(super().pop, index)

    def remove(self, value):
        return self._mutate(super().remove, value)

    def clear(self):
        return self._mutate(super().clear)

    def reverse(self):
        return self._mutate(super().reverse)

    def sort(self, *args, **kwargs):
        return self._mutate(super().sort, *args, **kwargs)

    def __iadd__(self, values):
        self._mutate(super().__iadd__, values)
        return self

    def __imul__(self, count):
        self._mutate(super().__imul__, count)
        return self


@dataclass(frozen=True)
class KeywordCandidate:
    chunk: object
    order: int
    term_counts: Counter
    document_length: int
    average_length: float
    document_count: int
    frequencies: dict[str, int]


class KeywordIndex:
    def __init__(self):
        self.source = None
        self.version = None
        self.tokenizer = None
        self._next_id = 0
        self._entries = {}
        self._order = {}
        self._postings = defaultdict(lambda: defaultdict(set))
        self._kb_ids = defaultdict(set)
        self._doc_ids = defaultdict(set)
        self._doc_lengths = Counter()
        self._doc_revision_ids = defaultdict(set)
        self._doc_revision_lengths = Counter()
        self._lengths = Counter()

    def synchronize(self, chunks, tokenize):
        if self.source is chunks and self.version == chunks.version and self.tokenizer is tokenize:
            return
        # Reuse existing immutable chunks and tokenization, removing only changed
        # rows. A list reorder updates ties, not statistics or document tokens.
        existing = defaultdict(deque)
        if self.tokenizer is tokenize:
            for identity, (chunk, terms, length) in self._entries.items():
                existing[chunk].append(identity)
        kept = set()
        order = {}
        for position, chunk in enumerate(chunks):
            identity = existing[chunk].popleft() if existing[chunk] else self._add(chunk, tokenize)
            kept.add(identity)
            order[identity] = position
        for identity in self._entries.keys() - kept:
            self._remove(identity)
        self._order = order
        self.source, self.version, self.tokenizer = chunks, chunks.version, tokenize

    def _add(self, chunk, tokenize):
        self._next_id += 1
        identity = self._next_id
        terms = Counter(tokenize(keyword_text(chunk).lower()))
        length = sum(terms.values())
        self._entries[identity] = (chunk, terms, length)
        kb = chunk.knowledge_base_id
        self._kb_ids[kb].add(identity)
        self._doc_ids[(kb, chunk.doc_id)].add(identity)
        self._doc_lengths[(kb, chunk.doc_id)] += length
        revision_key = (kb, chunk.doc_id, getattr(chunk, "retrieval_revision", "legacy"))
        self._doc_revision_ids[revision_key].add(identity)
        self._doc_revision_lengths[revision_key] += length
        self._lengths[kb] += length
        for term in terms:
            self._postings[kb][term].add(identity)
        return identity

    def _remove(self, identity):
        chunk, terms, length = self._entries.pop(identity)
        kb = chunk.knowledge_base_id
        self._kb_ids[kb].remove(identity)
        document = (kb, chunk.doc_id)
        self._doc_ids[document].remove(identity)
        self._doc_lengths[document] -= length
        if not self._doc_ids[document]:
            del self._doc_ids[document]
            del self._doc_lengths[document]
        revision_key = (*document, getattr(chunk, "retrieval_revision", "legacy"))
        self._doc_revision_ids[revision_key].remove(identity)
        self._doc_revision_lengths[revision_key] -= length
        if not self._doc_revision_ids[revision_key]:
            del self._doc_revision_ids[revision_key]
            del self._doc_revision_lengths[revision_key]
        self._lengths[kb] -= length
        for term in terms:
            posting = self._postings[kb][term]
            posting.remove(identity)
            if not posting:
                del self._postings[kb][term]
        if not self._kb_ids[kb]:
            del self._kb_ids[kb]
            del self._lengths[kb]
            del self._postings[kb]

    def candidates(self, query_terms, knowledge_base_ids, limit, allowed_documents=None, allowed_revisions=None):
        identities = set()
        allowed = set(knowledge_base_ids)
        visible, document_counts, lengths = {}, {}, {}
        for kb in allowed:
            documents = (allowed_documents or {}).get(kb)
            if documents is None:
                visible[kb] = self._kb_ids.get(kb, set())
                document_counts[kb] = len(visible[kb])
                lengths[kb] = self._lengths[kb]
            else:
                revisions = (allowed_revisions or {}).get(kb)
                if revisions is None:
                    visible[kb] = set().union(*(self._doc_ids.get((kb, doc), set()) for doc in documents))
                    lengths[kb] = sum(self._doc_lengths[(kb, doc)] for doc in documents)
                else:
                    keys = [(kb, doc, revisions[doc]) for doc in documents if doc in revisions]
                    visible[kb] = set().union(*(self._doc_revision_ids.get(key, set()) for key in keys))
                    lengths[kb] = sum(self._doc_revision_lengths[key] for key in keys)
                document_counts[kb] = len(visible[kb])
            if query_terms:
                postings = self._postings.get(kb, {})
                for term in query_terms:
                    identities.update(set(postings.get(term, ())) & visible[kb])
            else:
                identities.update(visible[kb])
        ordered = sorted(identities, key=self._order.__getitem__)
        if not query_terms:
            ordered = ordered[:limit]
        snapshots = []
        for identity in ordered:
            chunk, counts, length = self._entries[identity]
            kb = chunk.knowledge_base_id
            total = document_counts[kb]
            snapshots.append(KeywordCandidate(
                chunk, self._order[identity], counts, max(length, 1),
                max(lengths[kb] / total, 1.0), total,
                {term: len(set(self._postings[kb].get(term, ())) & visible[kb]) for term in query_terms},
            ))
        return snapshots


def score_candidate(query_terms, candidate: KeywordCandidate) -> float:
    if not query_terms:
        return 0.01
    score, matched = 0.0, 0
    k1, b = 1.5, 0.75
    for term in query_terms:
        tf = candidate.term_counts.get(term, 0)
        if tf <= 0:
            continue
        matched += 1
        containing = candidate.frequencies[term]
        idf = math.log(1 + (candidate.document_count - containing + 0.5) / (containing + 0.5))
        length_norm = tf + k1 * (1 - b + b * (candidate.document_length / candidate.average_length))
        score += idf * (tf * (k1 + 1) / max(length_norm, 0.0001))
    if not matched:
        return 0.0
    coverage = matched / len(query_terms)
    normalized = score / (score + 4.0) if score > 0 else 0.0
    return min(1.0, normalized * 0.72 + coverage * 0.28)
