from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
import sys
from types import SimpleNamespace

import pytest

from app.core.config import settings
from app.embedding import client
from app.rag.cache_scope import retrieval_cache_scope


@pytest.fixture(autouse=True)
def cache_setup(monkeypatch):
    client.clear_embedding_query_cache()
    monkeypatch.setattr(settings, "embedding_cache_enabled", True)
    monkeypatch.setattr(settings, "embedding_dimension", 2)
    monkeypatch.setattr(settings, "embedding_provider", "openai")
    monkeypatch.setattr(settings, "embedding_model", "embed-test")
    monkeypatch.setattr(settings, "embedding_base_url", "https://first.invalid/v1")
    monkeypatch.setattr(settings, "embedding_api_key", "test-secret")
    yield
    client.clear_embedding_query_cache()


def test_query_cache_skips_provider_usage_and_returns_independent_vectors(monkeypatch):
    calls = []
    monkeypatch.setattr(client, "embed_text", lambda text: calls.append(text) or [0.1, 0.9])
    with retrieval_cache_scope("user", ["kb"], conversation_id="conversation"):
        first = client.embed_query("question")
        first[0] = 999
        assert client.embed_query("question") == [0.1, 0.9]
    assert calls == ["question"]


def test_unscoped_queries_and_document_embeddings_are_never_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(client, "_embed_with_openai_compatible", lambda text: calls.append(text) or [0.1, 0.9])
    client.embed_query("query")
    client.embed_query("query")
    with retrieval_cache_scope("user", ["kb"]):
        client.embed_text("document")
        client.embed_text("document")
    assert calls == ["query", "query", "document", "document"]


@pytest.mark.parametrize("field,new_value", [
    ("embedding_provider", "fastembed"), ("embedding_model", "other-model"),
    ("embedding_base_url", "https://second.invalid/v1"), ("embedding_api_key", "other-secret"),
    ("embedding_dimension", 3),
])
def test_vector_model_endpoint_credentials_and_dimension_partition_cache(monkeypatch, field, new_value):
    calls = []
    monkeypatch.setattr(client, "embed_text", lambda text: calls.append(text) or [0.5] * settings.embedding_dimension)
    with retrieval_cache_scope("user", ["kb"]):
        client.embed_query("same question")
        monkeypatch.setattr(settings, field, new_value)
        client.embed_query("same question")
    assert len(calls) == 2
    assert "test-secret" not in repr(client.embedding_identity())


def test_users_and_conversations_never_share_vector_cache_entries(monkeypatch):
    calls = []
    monkeypatch.setattr(client, "embed_text", lambda text: calls.append(text) or [0.1, 0.9])
    for user, conversation in [("one", "a"), ("two", "a"), ("one", "b"), ("one", "a")]:
        with retrieval_cache_scope(user, ["kb"], conversation_id=conversation):
            client.embed_query("question")
    assert len(calls) == 3


@pytest.mark.parametrize("vector", [[], [0.1], [float("nan"), 0.2], [float("inf"), 0.2]])
def test_invalid_or_empty_vectors_are_not_cached(monkeypatch, vector):
    calls = []
    monkeypatch.setattr(client, "embed_text", lambda text: calls.append(text) or vector)
    with retrieval_cache_scope("user", ["kb"]):
        client.embed_query("question")
        client.embed_query("question")
    assert len(calls) == 2


def test_embedding_exception_is_not_cached_or_retried_automatically(monkeypatch):
    calls = []

    def load(text):
        calls.append(text)
        raise TimeoutError("provider unavailable")

    monkeypatch.setattr(client, "embed_text", load)
    with retrieval_cache_scope("user", ["kb"]):
        for _ in range(2):
            with pytest.raises(TimeoutError):
                client.embed_query("question")
    assert len(calls) == 2


def test_simultaneous_query_embeddings_use_one_provider_call(monkeypatch):
    gate, started, release = Barrier(3), Event(), Event()
    calls = []

    def load(text):
        calls.append(text)
        started.set()
        assert release.wait(3)
        return [0.1, 0.9]

    def run():
        with retrieval_cache_scope("user", ["kb"]):
            gate.wait(3)
            return client.embed_query("question")

    monkeypatch.setattr(client, "embed_text", load)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run) for _ in range(2)]
        gate.wait(3)
        assert started.wait(3)
        release.set()
        assert [future.result(3) for future in futures] == [[0.1, 0.9], [0.1, 0.9]]
    assert calls == ["question"]


def test_local_embedding_model_cache_is_partitioned_by_model_name(monkeypatch):
    loaded = []
    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=lambda model_name: loaded.append(model_name) or object()))
    client._cached_fastembed_model.cache_clear()
    try:
        monkeypatch.setattr(settings, "embedding_model", "model-one")
        first = client._fastembed_model()
        assert client._fastembed_model() is first
        monkeypatch.setattr(settings, "embedding_model", "model-two")
        assert client._fastembed_model() is not first
        assert loaded == ["model-one", "model-two"]
    finally:
        client._cached_fastembed_model.cache_clear()
