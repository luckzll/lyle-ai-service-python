import pytest

from app.core.config import settings
from app.core.runtime_config import temporary_ai_config
from app.rag.reranker import get_reranker_status, rerank_items


def test_lexical_reranker_prefers_query_overlap(monkeypatch):
    monkeypatch.setattr(settings, "rerank_enabled", True)
    monkeypatch.setattr(settings, "rerank_provider", "lexical")

    ranked = rerank_items(
        "用户喜欢西瓜",
        [
            {"text": "用户正在写 Java 代码", "score": 0.9},
            {"text": "用户喜欢吃西瓜", "score": 0.1},
        ],
        "text",
        1,
    )

    assert ranked[0]["text"] == "用户喜欢吃西瓜"
    assert ranked[0]["rerank_backend"] == "lexical"


def test_reranker_returns_original_order_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "rerank_enabled", False)

    ranked = rerank_items(
        "用户喜欢西瓜",
        [
            {"text": "A"},
            {"text": "B"},
        ],
        "text",
        1,
    )

    assert ranked == [{"text": "A"}]


def test_flag_embedding_reranker_uses_model_scores(monkeypatch):
    monkeypatch.setattr(settings, "rerank_enabled", True)
    monkeypatch.setattr(settings, "rerank_provider", "flag_embedding")
    monkeypatch.setattr(settings, "rerank_fallback_enabled", False)

    class FakeReranker:
        def compute_score(self, pairs, normalize=True):
            assert normalize is True
            return [0.2, 0.95]

    monkeypatch.setattr("app.rag.reranker._flag_reranker_model", lambda: FakeReranker())

    ranked = rerank_items(
        "用户喜欢什么",
        [
            {"text": "用户喜欢苹果", "score": 0.9},
            {"text": "用户喜欢西瓜", "score": 0.1},
        ],
        "text",
        1,
    )

    assert ranked[0]["text"] == "用户喜欢西瓜"
    assert ranked[0]["rerank_backend"] == "flag_embedding"
    assert ranked[0]["rerank_score"] == 0.95


def test_flag_embedding_status_reports_missing_dependency(monkeypatch):
    monkeypatch.setattr(settings, "rerank_enabled", True)
    monkeypatch.setattr(settings, "rerank_provider", "flag_embedding")
    monkeypatch.setattr(
        "app.rag.reranker._import_flag_reranker",
        lambda: (_ for _ in ()).throw(ImportError("missing FlagEmbedding")),
    )

    status = get_reranker_status()

    assert status["ready"] is False
    assert "FlagEmbedding is not installed" in str(status["message"])


@pytest.mark.parametrize("scores", [[0.7], [float("nan"), 0.4], [float("inf"), 0.2], [0.3, 0.7, 0.9]])
def test_invalid_model_score_batch_falls_back_without_dropping_candidates(monkeypatch, scores):
    monkeypatch.setattr(settings, "rerank_enabled", True)
    monkeypatch.setattr(settings, "rerank_provider", "flag_embedding")
    monkeypatch.setattr(settings, "rerank_fallback_enabled", True)

    class Model:
        def compute_score(self, *_args, **_kwargs):
            return scores

    monkeypatch.setattr("app.rag.reranker._flag_reranker_model", lambda: Model())
    rows = [{"text": "unrelated", "score": 100}, {"text": "precise answer", "score": 0.1}]
    ranked = rerank_items("precise answer", rows, "text", 2)
    assert len(ranked) == 2
    assert ranked[0]["text"] == "precise answer"
    assert all(row["rerank_backend"] == "lexical" for row in ranked)


def test_invalid_model_batch_still_raises_when_fallback_disabled(monkeypatch):
    monkeypatch.setattr(settings, "rerank_enabled", True)
    monkeypatch.setattr(settings, "rerank_provider", "flag_embedding")
    monkeypatch.setattr(settings, "rerank_fallback_enabled", False)

    class Model:
        def compute_score(self, *_args, **_kwargs):
            return [0.5]

    monkeypatch.setattr("app.rag.reranker._flag_reranker_model", lambda: Model())
    with pytest.raises(ValueError, match="invalid candidate scores"):
        rerank_items("question", [{"text": "one"}, {"text": "two"}], "text", 2)


def test_lexical_tie_breaker_cannot_be_dominated_by_unbounded_keyword_score(monkeypatch):
    monkeypatch.setattr(settings, "rerank_enabled", True)
    monkeypatch.setattr(settings, "rerank_provider", "lexical")
    rows = [
        {"text": "stock market news", "score": 1000, "raw_score": 1000, "fusion_score": 0.032},
        {"text": "battery storage temperature", "score": 0.1, "raw_score": 0.1, "fusion_score": 0.016},
    ]
    ranked = rerank_items("battery storage temperature", rows, "text", 2)
    assert ranked[0]["text"] == "battery storage temperature"
    assert ranked[0]["raw_score"] == 0.1 and ranked[0]["fusion_score"] == 0.016
    assert ranked[1]["score"] == 1000


def test_request_reranker_override_is_used_without_changing_process_settings(monkeypatch):
    monkeypatch.setattr(settings, "rerank_enabled", False)
    monkeypatch.setattr(settings, "rerank_provider", "lexical")
    rows = [{"text": "unrelated"}, {"text": "exact answer"}]
    with temporary_ai_config({"RERANK_ENABLED": "true"}):
        assert rerank_items("exact answer", rows, "text", 1)[0]["text"] == "exact answer"
        assert settings.rerank_enabled is False
    assert rerank_items("exact answer", rows, "text", 1) == rows[:1]
