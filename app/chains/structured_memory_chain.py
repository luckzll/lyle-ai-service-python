from app.llm.client import extract_structured_memories


def invoke_structured_memory_chain(question: str, recent_turns_text: str) -> list[dict[str, object]]:
    return extract_structured_memories(question, recent_turns_text)
