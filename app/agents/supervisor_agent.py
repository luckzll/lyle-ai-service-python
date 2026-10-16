from app.chains.intent_chain import classify_intent
from app.agents.tool_agent import should_use_tool_agent
from app.chat_execution_plan import is_self_contained, needs_personal_memory, needs_tool_planning, references_knowledge, requests_external_tools
from app.chat_attachments import current_attachments
from app.tools.tool_registry import select_tools


def run_supervisor_agent(question: str, knowledge_base_ids: list[str]) -> dict[str, object]:
    intent = classify_intent(question, knowledge_base_ids)
    use_memory = needs_personal_memory(question) or (intent == "memory" and not is_self_contained(question))
    use_rag = bool(knowledge_base_ids) and (references_knowledge(question) or intent == "knowledge")
    if is_self_contained(question) or (requests_external_tools(question) and not references_knowledge(question)):
        use_rag = False
    attachments = current_attachments()
    has_attachments = bool(attachments and attachments.files)
    tool_signal = needs_tool_planning(question, has_attachments=has_attachments) or bool(select_tools(question))
    use_tools = should_use_tool_agent(question) and (has_attachments or (tool_signal and not is_self_contained(question)))
    if references_knowledge(question) and not has_attachments and not requests_external_tools(question):
        # A document search is the RAG worker's job, not a second web search.
        use_tools = use_tools and any(name != "web_search" for name in select_tools(question))
    return {
        "intent": intent,
        "use_memory": use_memory,
        "use_rag": use_rag,
        "use_tools": use_tools,
        "next_agents": _next_agents(use_memory, use_rag, use_tools),
    }


def _next_agents(use_memory: bool, use_rag: bool, use_tools: bool) -> list[str]:
    agents = ["memory_agent"] if use_memory else []
    if use_rag:
        agents.append("rag_agent")
    if use_tools:
        agents.append("tool_agent")
    agents.append("writer_agent")
    return agents
