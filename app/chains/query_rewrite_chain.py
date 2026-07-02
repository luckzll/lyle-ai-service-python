import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import settings
from app.llm.tasks.intent_model import build_intent_task_model, is_intent_model_configured


def rewrite_rag_queries(question: str, max_queries: int | None = None) -> list[str]:
    limit = max_queries or settings.rag_multi_query_count
    if not settings.rag_query_rewrite_enabled or limit <= 1:
        return [question]
    if not is_intent_model_configured():
        return _rule_queries(question, limit)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "你是知识库检索 query 改写器。只返回 JSON，不要解释。"
                "请把用户问题改写成适合向量检索的中文查询，保留原意，避免引入新事实。"
                "返回格式：{\"queries\":[\"原问题\",\"改写1\"]}"
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
        return _rule_queries(question, limit)

    if not isinstance(queries, list):
        return _rule_queries(question, limit)
    return _dedupe_queries([question, *[str(item) for item in queries]], limit)


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
