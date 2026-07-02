from langchain_core.tools import tool

from app.rag.retriever import retrieve_context, retrieve_context_with_diagnostics


def retrieve_knowledge_tool(question: str, knowledge_base_ids: list[str]) -> list[dict[str, str]]:
    return retrieve_context(question, knowledge_base_ids)


def retrieve_knowledge_diagnostics_tool(question: str, knowledge_base_ids: list[str]) -> dict[str, object]:
    return retrieve_context_with_diagnostics(question, knowledge_base_ids)


@tool("retrieve_knowledge")
def retrieve_knowledge_langchain_tool(question: str, knowledge_base_ids: list[str]) -> list[dict[str, str]]:
    """Retrieve relevant chunks from selected knowledge bases."""
    return retrieve_knowledge_tool(question, knowledge_base_ids)
