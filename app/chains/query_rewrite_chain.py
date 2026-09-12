import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.runtime_config import current_rag_config
from app.llm.tasks.intent_model import build_intent_task_model, is_intent_model_configured
from app.rag.query_context import current_query_context


def rewrite_rag_queries(question: str, max_queries: int | None = None) -> list[str]:
    config = current_rag_config()
    limit = max_queries or config.rag_multi_query_count
    context = current_query_context()
    originals = list(context.additional_queries) if context and question == context.effective_question else []

    def keep_intent(queries: list[str]) -> list[str]:
        # Keep the resolved subject first while reserving a query slot for the
        # user's original wording. The retriever handles a one-query fallback.
        return _dedupe_queries([question, *originals, *queries], limit)

    if not config.rag_query_rewrite_enabled or limit <= 1:
        return keep_intent([question])
    if not config.rag_auxiliary_model_enabled or not is_intent_model_configured():
        return keep_intent(_rule_queries(question, limit))

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "你是知识库检索 query 改写器。只返回 JSON，不要解释。"
                "请把用户问题改写成适合向量检索的中文查询，保留原意，避免引入新事实。"
                "返回格式：{{\"queries\":[\"原问题\",\"改写1\"]}}"
            ),
        ),
        ("human", "用户问题：{question}\n最多返回 {limit} 个 query。"),
    ])
    chain = prompt | build_intent_task_model() | StrOutputParser()
    try:
        content = chain.invoke({"question": question, "limit": limit})
        parsed = json.loads(content)
        queries = parsed.get("queries") if isinstance(parsed, dict) else parsed
    except Exception:
        return keep_intent(_rule_queries(question, limit))

    if not isinstance(queries, list):
        return keep_intent(_rule_queries(question, limit))
    return keep_intent([str(item) for item in queries])


def _rule_queries(question: str, limit: int) -> list[str]:
    variants = [question]
    cleaned = question.strip().replace("？", "").replace("?", "")
    if cleaned and cleaned != question:
        variants.append(cleaned)
    if any(keyword in question for keyword in ["讲了什么", "是什么", "介绍"]):
        variants.append(f"{cleaned} 核心内容 摘要 说明")
    return _dedupe_queries(variants, limit)


def _dedupe_queries(queries: list[str], limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for query in queries:
        value = query.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
        if len(result) >= limit:
            break
    return result or [queries[0]]
