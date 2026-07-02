from langchain_core.tools import tool

from app.memory.store import format_memory_context, recall_memory, remember_interaction


def recall_memory_tool(
    user_id: str,
    conversation_id: str | None,
    question: str,
    include_vector: bool = True,
) -> dict[str, object]:
    memory = recall_memory(user_id, conversation_id, question, include_vector=include_vector)
    return {
        "memory": memory,
        "memory_context": format_memory_context(memory),
    }


def write_memory_tool(user_id: str, conversation_id: str | None, question: str, answer: str) -> dict[str, object]:
    return remember_interaction(user_id, conversation_id, question, answer)


@tool("recall_memory")
def recall_memory_langchain_tool(user_id: str, conversation_id: str | None, question: str) -> dict[str, object]:
    """Recall user memory from Redis, PostgreSQL, Qdrant, and Milvus."""
    return recall_memory_tool(user_id, conversation_id, question)


@tool("write_memory")
def write_memory_langchain_tool(user_id: str, conversation_id: str | None, question: str, answer: str) -> str:
    """Write a completed interaction into memory stores."""
    write_memory_tool(user_id, conversation_id, question, answer)
    return "ok"
