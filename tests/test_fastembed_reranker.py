import math
import pytest

from app.core.runtime_config import temporary_ai_config
from app.rag import reranker


class Model:
    def rerank(self, query, texts, batch_size):
        assert texts == ["context a", "source b"]
        return [-10, 10]


def test_fastembed_scores_original_citations_without_mutating_inputs(monkeypatch):
    monkeypatch.setattr(reranker, "_fastembed_reranker_model", lambda: Model())
    items = [{"chunk": "source a", "retrieval_text": "context a", "doc_id": "a", "startChar": "10"},
             {"chunk": "source b", "doc_id": "b", "startChar": "20"}]
    with temporary_ai_config({"RERANK_ENABLED": "true", "RERANK_PROVIDER": "fastembed"}):
        ranked = reranker.rerank_items("query", items, "chunk", 2)
    assert [item["doc_id"] for item in ranked] == ["b", "a"]
    assert ranked[1]["chunk"] == "source a" and ranked[1]["startChar"] == "10"
    assert ranked[0]["rerank_score"] == pytest.approx(1 / (1 + math.exp(-10)))
    assert "rerank_score" not in items[0]


@pytest.mark.parametrize("scores", [[float("nan"), 0], [0], [float("inf"), 0]])
def test_fastembed_invalid_scores_are_not_success(monkeypatch, scores):
    class Invalid:
        def rerank(self, *args, **kwargs): return scores
    monkeypatch.setattr(reranker, "_fastembed_reranker_model", lambda: Invalid())
    with temporary_ai_config({"RERANK_ENABLED": "true", "RERANK_PROVIDER": "fastembed", "RERANK_FALLBACK_ENABLED": "false"}):
        with pytest.raises(ValueError):
            reranker.rerank_items("query", [{"chunk": "a"}, {"chunk": "b"}], "chunk", 2)
