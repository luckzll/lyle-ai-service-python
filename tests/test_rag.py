from pathlib import Path

from app.core.config import settings
from app.chains.query_rewrite_chain import rewrite_rag_queries
from app.chains.rag_sufficiency_chain import judge_rag_sufficiency
from app.rag.retriever import LyleKnowledgeRetriever
from app.rag.retriever import retrieve_context_with_diagnostics
from app.rag.store import add_document, search


def test_rag_in_memory_fallback(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "rerank_enabled", False)
    document = tmp_path / "knowledge.txt"
    document.write_text("海蓝星是一颗被海洋覆盖的虚构星球。", encoding="utf-8")

    add_document("file-1", "kb-1", str(document))
    results = search("海蓝星是什么", ["kb-1"])

    assert results
    assert "海蓝星" in results[0]["chunk"]

    documents = LyleKnowledgeRetriever(knowledge_base_ids=["kb-1"], top_k=1).invoke("海蓝星是什么")
    assert documents
    assert "海蓝星" in documents[0].page_content


def test_rag_reranks_in_memory_candidates(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "rerank_enabled", True)
    monkeypatch.setattr(settings, "rerank_provider", "lexical")
    monkeypatch.setattr(settings, "rerank_candidate_limit", 20)

    banana = tmp_path / "banana.txt"
    apple = tmp_path / "apple.txt"
    banana.write_text("香蕉是一种水果，适合做奶昔。", encoding="utf-8")
    apple.write_text("苹果手机可以安装移动应用，也可以调试前端页面。", encoding="utf-8")

    add_document("file-banana", "kb-rerank", str(banana))
    add_document("file-apple", "kb-rerank", str(apple))

    results = search("苹果手机应用", ["kb-rerank"], top_k=1)

    assert len(results) == 1
    assert "苹果手机" in results[0]["chunk"]
    assert results[0]["rerank_backend"] == "lexical"


def test_query_rewrite_rule_fallback(monkeypatch):
    monkeypatch.setattr(settings, "rag_query_rewrite_enabled", True)
    monkeypatch.setattr(settings, "rag_multi_query_count", 3)
    monkeypatch.setattr(settings, "deepseek_api_key", "")

    queries = rewrite_rag_queries("这份文档讲了什么？")

    assert queries[0] == "这份文档讲了什么？"
    assert len(queries) >= 2


def test_rag_diagnostics_multi_query(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "rerank_enabled", False)
    monkeypatch.setattr(settings, "rag_query_rewrite_enabled", True)
    monkeypatch.setattr(settings, "rag_multi_query_count", 2)
    monkeypatch.setattr(settings, "rag_context_top_k", 2)
    monkeypatch.setattr(settings, "rag_sufficiency_check_enabled", True)
    monkeypatch.setattr(settings, "deepseek_api_key", "")

    document = tmp_path / "agent.txt"
    document.write_text("多 Agent 架构包含 Supervisor、Memory、RAG、Writer。", encoding="utf-8")
    add_document("file-agent", "kb-agent", str(document))

    result = retrieve_context_with_diagnostics("多 Agent 架构讲了什么？", ["kb-agent"])

    assert result["queries"]
    assert result["contexts"]
    assert result["sufficiency"]["sufficient"] is True
    assert result["attribution"]["contextCount"] >= 1
    assert result["citationQuality"]["level"] in {"low", "medium", "high"}


def test_rag_diagnostics_reports_badcase_attribution(monkeypatch):
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "rerank_enabled", False)
    monkeypatch.setattr(settings, "rag_query_rewrite_enabled", False)
    monkeypatch.setattr(settings, "rag_sufficiency_check_enabled", True)

    result = retrieve_context_with_diagnostics("完全不存在的问题", ["empty-kb"])

    assert result["contexts"] == []
    assert result["attribution"]["status"] == "needs_attention"
    assert "没有命中文档片段" in result["attribution"]["reasons"]
    assert result["citationQuality"]["level"] == "empty"


def test_rag_sufficiency_reports_empty_contexts(monkeypatch):
    monkeypatch.setattr(settings, "rag_sufficiency_check_enabled", True)

    result = judge_rag_sufficiency("项目架构是什么", [])

    assert result["sufficient"] is False
