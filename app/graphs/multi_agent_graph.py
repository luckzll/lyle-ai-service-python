import operator
import logging
from typing import Annotated, TypedDict
from time import perf_counter

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.agents.memory_agent import run_memory_agent
from app.agents.rag_agent import run_rag_agent
from app.agents.reviewer_agent import run_reviewer_agent
from app.agents.supervisor_agent import run_supervisor_agent
from app.agents.tool_agent import run_tool_agent
from app.agents.writer_agent import run_writer_agent
from app.core.config import settings


logger = logging.getLogger(__name__)


class MultiAgentState(TypedDict):
    user_id: str
    conversation_id: str | None
    question: str
    knowledge_base_ids: list[str]
    supervisor: dict[str, object]
    memory: dict[str, object]
    memory_context: str
    contexts: list[dict[str, str]]
    rag_queries: list[str]
    rag_sufficiency: dict[str, object]
    tool_results: list[dict[str, object]]
    tool_context: str
    candidate_memories: list[dict[str, object]]
    answer: str
    intent: str
    review: dict[str, object]
    review_attempts: int
    review_feedback: str
    agent_trace: list[dict[str, object]]
    agent_outputs: Annotated[list[dict[str, object]], operator.add]


def supervisor_node(state: MultiAgentState) -> MultiAgentState:
    started = perf_counter()
    supervisor = run_supervisor_agent(state["question"], state["knowledge_base_ids"])
    logger.info(
        "context_graph timing supervisor_done. user_id=%s conversation_id=%s elapsed=%.3fs intent=%s use_tools=%s use_rag=%s",
        state["user_id"],
        state["conversation_id"],
        perf_counter() - started,
        supervisor.get("intent"),
        supervisor.get("use_tools"),
        supervisor.get("use_rag"),
    )
    return {
        **state,
        "supervisor": supervisor,
        "intent": str(supervisor["intent"]),
        "agent_trace": [
            *state["agent_trace"],
            {
                "agent": "supervisor_agent",
                "intent": supervisor["intent"],
                "nextAgents": supervisor["next_agents"],
                "elapsedMs": _elapsed_ms(started),
            },
        ],
    }


def memory_agent_node(state: MultiAgentState) -> MultiAgentState:
    if not state["supervisor"].get("use_memory", True):
        return {
            "agent_outputs": [{
                "agent": "memory_agent",
                "memory": {},
                "memory_context": "",
                "trace": {"agent": "memory_agent", "skipped": True},
            }],
        }

    started = perf_counter()
    result = run_memory_agent(state["user_id"], state["conversation_id"], state["question"])
    logger.info(
        "context_graph timing memory_done. user_id=%s conversation_id=%s elapsed=%.3fs skipped=%s memory_context_chars=%s",
        state["user_id"],
        state["conversation_id"],
        perf_counter() - started,
        result.get("skipped"),
        len(str(result.get("memory_context") or "")),
    )
    decision = result.get("decision") or {}
    memory = result.get("memory") or {}
    return {
        "agent_outputs": [{
            "agent": "memory_agent",
            "memory": memory,
            "memory_context": str(result["memory_context"]),
            "trace": {
                "agent": "memory_agent",
                "memoryContextChars": len(str(result["memory_context"])),
                "memorySummary": _memory_trace_summary(memory),
                "decision": decision,
                "skipped": result.get("skipped", False),
                "elapsedMs": _elapsed_ms(started),
            },
        }],
    }


def rag_agent_node(state: MultiAgentState) -> MultiAgentState:
    started = perf_counter()
    result = run_rag_agent(
        state["question"],
        state["knowledge_base_ids"],
        bool(state["supervisor"].get("use_rag", False)),
    )
    contexts = result.get("contexts") or []
    queries = result.get("queries") or []
    sufficiency = result.get("sufficiency") or {}
    decision = result.get("decision") or {}
    logger.info(
        "context_graph timing rag_done. user_id=%s conversation_id=%s elapsed=%.3fs skipped=%s contexts=%s queries=%s",
        state["user_id"],
        state["conversation_id"],
        perf_counter() - started,
        result.get("skipped"),
        len(contexts),
        len(queries),
    )
    return {
        "agent_outputs": [{
            "agent": "rag_agent",
            "contexts": contexts,
            "rag_queries": queries,
            "rag_sufficiency": sufficiency,
            "trace": {
                "agent": "rag_agent",
                "contextCount": len(contexts),
                "queryCount": len(queries),
                "sufficient": sufficiency.get("sufficient"),
                "skipped": result.get("skipped", False),
                "decision": decision,
                "elapsedMs": _elapsed_ms(started),
            },
        }],
    }


def writer_agent_node(state: MultiAgentState) -> MultiAgentState:
    started = perf_counter()
    writer_input = _writer_input_debug(
        state["memory_context"],
        state["tool_context"],
        state["contexts"],
    )
    answer = run_writer_agent(
        state["question"],
        state["contexts"],
        state["memory_context"],
        state["tool_context"],
        state["review_feedback"],
    )
    logger.info(
        "multi_agent timing writer_done. user_id=%s conversation_id=%s elapsed=%.3fs answer_chars=%s",
        state["user_id"],
        state["conversation_id"],
        perf_counter() - started,
        len(answer),
    )
    return {
        **state,
        "answer": answer,
        "review_feedback": "",
        "agent_trace": [
            *state["agent_trace"],
            {
                "agent": "writer_agent",
                "answerChars": len(answer),
                "rewriteAttempt": state["review_attempts"],
                "writerInput": writer_input,
                "elapsedMs": _elapsed_ms(started),
            },
        ],
    }


def tool_agent_node(state: MultiAgentState) -> MultiAgentState:
    started = perf_counter()
    result = run_tool_agent(
        state["user_id"],
        state["conversation_id"],
        state["question"],
        bool(state["supervisor"].get("use_tools", False)),
    )
    tool_results = result.get("tool_results") or []
    decision = result.get("decision") or {}
    logger.info(
        "context_graph timing tool_done. user_id=%s conversation_id=%s elapsed=%.3fs skipped=%s selected_tools=%s",
        state["user_id"],
        state["conversation_id"],
        perf_counter() - started,
        result.get("skipped"),
        result.get("selected_tools") or [],
    )
    return {
        "agent_outputs": [{
            "agent": "tool_agent",
            "tool_results": tool_results,
            "tool_context": str(result.get("tool_context") or ""),
            "trace": {
                "agent": "tool_agent",
                "selectedTools": result.get("selected_tools") or [],
                "toolCount": len(tool_results),
                "skipped": result.get("skipped", False),
                "decision": decision,
                "elapsedMs": _elapsed_ms(started),
            },
        }],
    }


def route_after_supervisor(state: MultiAgentState) -> list[Send]:
    supervisor = state["supervisor"]
    sends = []
    if supervisor.get("use_memory", True):
        sends.append(Send("memory_agent", state))
    if supervisor.get("use_rag", False):
        sends.append(Send("rag_agent", state))
    if supervisor.get("use_tools", False):
        sends.append(Send("tool_agent", state))
    return sends or [Send("memory_agent", state)]


def gather_agent_node(state: MultiAgentState) -> MultiAgentState:
    started = perf_counter()
    outputs = state.get("agent_outputs", [])
    by_agent = {str(item.get("agent")): item for item in outputs}
    memory_output = by_agent.get("memory_agent", {})
    rag_output = by_agent.get("rag_agent", {})
    tool_output = by_agent.get("tool_agent", {})
    ordered_trace = [
        item["trace"]
        for item in sorted(outputs, key=lambda item: _agent_order(str(item.get("agent"))))
        if item.get("trace")
    ]
    next_state = {
        **state,
        "memory": memory_output.get("memory") or {},
        "memory_context": str(memory_output.get("memory_context") or ""),
        "contexts": rag_output.get("contexts") or [],
        "rag_queries": rag_output.get("rag_queries") or [],
        "rag_sufficiency": rag_output.get("rag_sufficiency") or {},
        "tool_results": tool_output.get("tool_results") or [],
        "tool_context": str(tool_output.get("tool_context") or ""),
        "agent_trace": [
            *state["agent_trace"],
            *ordered_trace,
            {
                "agent": "gather_agent",
                "receivedAgents": [item.get("agent") for item in outputs],
            },
        ],
    }
    logger.info(
        "context_graph timing gather_done. user_id=%s conversation_id=%s elapsed=%.3fs received_agents=%s",
        state["user_id"],
        state["conversation_id"],
        perf_counter() - started,
        [item.get("agent") for item in outputs],
    )
    return next_state


def reviewer_agent_node(state: MultiAgentState) -> MultiAgentState:
    started = perf_counter()
    try:
        review = run_reviewer_agent(
            state["question"],
            state["answer"],
            state["contexts"],
            state["memory_context"],
            memory=state["memory"],
            rag_sufficiency=state["rag_sufficiency"],
            tool_results=state["tool_results"],
        )
    except TypeError:
        review = run_reviewer_agent(
            state["question"],
            state["answer"],
            state["contexts"],
            state["memory_context"],
        )
    return {
        **state,
        "review": review,
        "review_attempts": state["review_attempts"] + (0 if review.get("approved") else 1),
        "review_feedback": str(review.get("rewrite_instruction") or review.get("reason") or ""),
        "agent_trace": [
            *state["agent_trace"],
            {
                "agent": "reviewer_agent",
                "approved": review.get("approved"),
                "score": review.get("score"),
                "skipped": review.get("skipped", False),
                "elapsedMs": _elapsed_ms(started),
            },
        ],
    }


# 记忆写入不再作为图节点串行执行，由 API 层在响应返回后异步完成（见 app/api/chat.py）。
def route_after_review(state: MultiAgentState) -> str:
    if state["review"].get("approved", True):
        return END
    if state["review_attempts"] <= settings.reviewer_max_retry:
        return "writer_agent"
    return END


def build_multi_agent_graph():
    graph = StateGraph(MultiAgentState)
    graph.add_node("supervisor_agent", supervisor_node)
    graph.add_node("memory_agent", memory_agent_node)
    graph.add_node("rag_agent", rag_agent_node)
    graph.add_node("tool_agent", tool_agent_node)
    graph.add_node("gather_agent", gather_agent_node)
    graph.add_node("writer_agent", writer_agent_node)
    graph.add_node("reviewer_agent", reviewer_agent_node)
    graph.add_edge(START, "supervisor_agent")
    graph.add_conditional_edges("supervisor_agent", route_after_supervisor)
    graph.add_edge("memory_agent", "gather_agent")
    graph.add_edge("rag_agent", "gather_agent")
    graph.add_edge("tool_agent", "gather_agent")
    graph.add_edge("gather_agent", "writer_agent")
    graph.add_edge("writer_agent", "reviewer_agent")
    graph.add_conditional_edges("reviewer_agent", route_after_review)
    return graph.compile()


def build_context_graph():
    graph = StateGraph(MultiAgentState)
    graph.add_node("supervisor_agent", supervisor_node)
    graph.add_node("memory_agent", memory_agent_node)
    graph.add_node("rag_agent", rag_agent_node)
    graph.add_node("tool_agent", tool_agent_node)
    graph.add_node("gather_agent", gather_agent_node)
    graph.add_edge(START, "supervisor_agent")
    graph.add_conditional_edges("supervisor_agent", route_after_supervisor)
    graph.add_edge("memory_agent", "gather_agent")
    graph.add_edge("rag_agent", "gather_agent")
    graph.add_edge("tool_agent", "gather_agent")
    graph.add_edge("gather_agent", END)
    return graph.compile()


multi_agent_graph = build_multi_agent_graph()
context_graph = build_context_graph()


def run_multi_agent_graph(
    user_id: str,
    conversation_id: str | None,
    question: str,
    knowledge_base_ids: list[str],
) -> dict:
    return multi_agent_graph.invoke(_initial_state(user_id, conversation_id, question, knowledge_base_ids))


def prepare_multi_agent_context(
    user_id: str,
    conversation_id: str | None,
    question: str,
    knowledge_base_ids: list[str],
) -> dict:
    return context_graph.invoke(_initial_state(user_id, conversation_id, question, knowledge_base_ids))


def _initial_state(
    user_id: str,
    conversation_id: str | None,
    question: str,
    knowledge_base_ids: list[str],
) -> MultiAgentState:
    return {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "question": question,
        "knowledge_base_ids": knowledge_base_ids,
        "supervisor": {},
        "memory": {},
        "memory_context": "",
        "contexts": [],
        "rag_queries": [],
        "rag_sufficiency": {},
        "tool_results": [],
        "tool_context": "",
        "candidate_memories": [],
        "answer": "",
        "intent": "chat",
        "review": {},
        "review_attempts": 0,
        "review_feedback": "",
        "agent_trace": [],
        "agent_outputs": [],
    }


def _elapsed_ms(started: float) -> int:
    return int((perf_counter() - started) * 1000)


def _agent_order(agent_name: str) -> int:
    return {
        "memory_agent": 10,
        "rag_agent": 20,
        "tool_agent": 30,
    }.get(agent_name, 99)


def _memory_trace_summary(memory: dict[str, object]) -> dict[str, object]:
    summary = str(memory.get("summary") or "")
    short_term = _as_list(memory.get("short_term"))
    structured = _as_list(memory.get("structured"))
    semantic = _as_list(memory.get("semantic"))
    vector = _as_list(memory.get("vector"))
    retrieved = _as_list(memory.get("retrieved"))
    return {
        "summaryChars": len(summary),
        "summaryPreview": _preview(summary),
        "shortTermCount": len(short_term),
        "shortTermPreview": _preview_turns(short_term),
        "structuredCount": len(structured),
        "structuredPreview": _preview_structured(structured),
        "semanticCount": len(semantic),
        "vectorCount": len(vector),
        "retrievedCount": len(retrieved),
        "retrievedPreview": _preview_vector_items(retrieved),
    }


def _as_list(value: object) -> list:
    return value if isinstance(value, list) else []


def _preview(text: str, max_chars: int = 180) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 1].rstrip() + "…"


def _preview_turns(items: list, max_items: int = 3) -> list[dict[str, str]]:
    preview = []
    for item in items[-max_items:]:
        if isinstance(item, dict):
            preview.append({
                "role": str(item.get("role") or ""),
                "content": _preview(str(item.get("content") or ""), 80),
            })
    return preview


def _preview_structured(items: list, max_items: int = 8) -> list[str]:
    preview = []
    for item in items[:max_items]:
        if isinstance(item, dict):
            preview.append(f"{item.get('key')}={item.get('value')}")
    return preview


def _preview_vector_items(items: list, max_items: int = 3) -> list[str]:
    preview = []
    for item in items[:max_items]:
        if isinstance(item, dict):
            preview.append(_preview(str(item.get("text") or ""), 120))
    return preview


def _writer_input_debug(
    memory_context: str,
    tool_context: str,
    contexts: list[dict[str, str]],
) -> dict[str, object]:
    combined_context = "\n\n".join(item for item in [memory_context, tool_context] if item)
    return {
        "memoryContextChars": len(memory_context),
        "memoryContext": _preview(memory_context, 6000),
        "toolContextChars": len(tool_context),
        "toolContext": _preview(tool_context, 6000),
        "ragContextCount": len(contexts),
        "ragContexts": [
            {
                "title": item.get("title", ""),
                "chunkPreview": _preview(item.get("chunk", ""), 500),
            }
            for item in contexts[:5]
        ],
        "combinedContextChars": len(combined_context),
        "combinedContext": _preview(combined_context, 10000),
    }
