from app.chains.intent_chain import classify_intent
from app.agents.tool_agent import should_use_tool_agent


def run_supervisor_agent(question: str, knowledge_base_ids: list[str]) -> dict[str, object]:
    intent = classify_intent(question, knowledge_base_ids)
    use_tools = should_use_tool_agent(question)
    return {
        "intent": intent,
        "use_memory": True,
        "use_rag": intent == "knowledge",
        "use_tools": use_tools,
        "next_agents": _next_agents(intent, use_tools),
    }


def _next_agents(intent: str, use_tools: bool) -> list[str]:
    agents = ["memory_agent"]
    if intent == "knowledge":
        agents.append("rag_agent")
    if use_tools:
        agents.append("tool_agent")
    agents.append("writer_agent")
    return agents
