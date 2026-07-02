from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from app.core.config import settings
from app.chains.query_rewrite_chain import rewrite_rag_queries
from app.chains.rag_sufficiency_chain import judge_rag_sufficiency
from app.rag.store import search


def retrieve_context(question: str, knowledge_base_ids: list[str]) -> list[dict[str, str]]:
    retriever = LyleKnowledgeRetriever(knowledge_base_ids=knowledge_base_ids, top_k=settings.rag_context_top_k)
    documents = retriever.invoke(question)
    return [
        {
            "doc_id": str(document.metadata.get("doc_id", "")),
            "title": str(document.metadata.get("title", "")),
            "chunk": document.page_content,
            "score": str(document.metadata.get("score", "")),
            "backend": str(document.metadata.get("backend", "")),
        }
        for document in documents
    ]


def retrieve_context_with_diagnostics(question: str, knowledge_base_ids: list[str]) -> dict[str, object]:
    queries = rewrite_rag_queries(question)
    contexts = _retrieve_multi_query_contexts(queries, knowledge_base_ids, settings.rag_context_top_k)
    sufficiency = judge_rag_sufficiency(question, contexts)
    attribution = _diagnose_rag_badcase(question, knowledge_base_ids, queries, contexts, sufficiency)
    return {
        "contexts": contexts,
        "queries": queries,
        "sufficiency": sufficiency,
        "attribution": attribution,
        "citationQuality": _score_citation_quality(contexts),
    }


def _retrieve_multi_query_contexts(
    queries: list[str],
    knowledge_base_ids: list[str],
    top_k: int,
) -> list[dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    for query in queries:
        retriever = LyleKnowledgeRetriever(knowledge_base_ids=knowledge_base_ids, top_k=top_k)
        for document in retriever.invoke(query):
            doc_id = str(document.metadata.get("doc_id", ""))
            chunk = document.page_content
            key = f"{doc_id}:{chunk}"
            score = _safe_float(document.metadata.get("score"))
            existing = merged.get(key)
            if existing and _safe_float(existing.get("score")) >= score:
                continue
            merged[key] = {
                "doc_id": doc_id,
                "title": str(document.metadata.get("title", "")),
                "chunk": chunk,
                "score": str(score),
                "backend": str(document.metadata.get("backend", "")),
                "query": query,
            }
    return sorted(
        merged.values(),
        key=lambda item: _safe_float(item.get("score")),
        reverse=True,
    )[:top_k]


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _diagnose_rag_badcase(
    question: str,
    knowledge_base_ids: list[str],
    queries: list[str],
    contexts: list[dict[str, str]],
    sufficiency: dict[str, object],
) -> dict[str, object]:
    top_score = max((_safe_float(item.get("score")) for item in contexts), default=0.0)
    reasons: list[str] = []
    suggestions: list[str] = []

    if not knowledge_base_ids:
        reasons.append("未选择知识库")
        suggestions.append("先选择至少一个知识库再提问。")
    if not contexts:
        reasons.append("没有命中文档片段")
        suggestions.append("确认文件已完成入库，或换一个更贴近文档原文的问法。")
    elif top_score < 0.45:
        reasons.append("最高召回分数偏低")
        suggestions.append("尝试补充实体名、文档标题或关键词；必要时重建索引。")
    if contexts and not bool(sufficiency.get("sufficient", True)):
        reasons.append("命中文档不足以回答问题")
        suggestions.append("补充相关资料，或调整文档分块让完整段落能被一起召回。")
    if len(queries) <= 1:
        suggestions.append("可开启或增强 Query Rewrite，让同一问题生成多个检索 query。")

    if not reasons:
        reasons.append("召回结果基本可用")
        suggestions.append("优先检查 Writer 是否正确引用了命中文档。")

    return {
        "status": "ok" if reasons == ["召回结果基本可用"] else "needs_attention",
        "reasons": reasons,
        "suggestions": suggestions,
        "queryCount": len(queries),
        "contextCount": len(contexts),
        "topScore": round(top_score, 4),
        "sufficiencyScore": _safe_float(sufficiency.get("score")),
    }


def _score_citation_quality(contexts: list[dict[str, str]]) -> dict[str, object]:
    if not contexts:
        return {
            "level": "empty",
            "score": 0.0,
            "reason": "没有可引用片段。",
        }
    scores = [_safe_float(item.get("score")) for item in contexts]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    if avg_score >= 0.75:
        level = "high"
        reason = "命中文档平均分较高，引用质量较好。"
    elif avg_score >= 0.45:
        level = "medium"
        reason = "命中文档有一定相关性，建议人工确认关键片段。"
    else:
        level = "low"
        reason = "命中文档相关性偏低，回答容易跑偏。"
    return {
        "level": level,
        "score": round(avg_score, 4),
        "reason": reason,
    }


class LyleKnowledgeRetriever(BaseRetriever):
    knowledge_base_ids: list[str] = Field(default_factory=list)
    top_k: int = 5

    def _get_relevant_documents(self, query: str) -> list[Document]:
        results = search(query, self.knowledge_base_ids, self.top_k)
        return [
            Document(
                page_content=item.get("chunk", ""),
                metadata={
                    "doc_id": item.get("doc_id", ""),
                    "title": item.get("title", ""),
                    "score": item.get("score", ""),
                    "backend": item.get("backend", ""),
                },
            )
            for item in results
        ]
