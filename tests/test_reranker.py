from app.core.config import settings
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
