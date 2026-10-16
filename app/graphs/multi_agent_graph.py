import operator
import logging
from functools import wraps
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
from app.chat_progress import emit_step, sanitize_timings
from app.chat_attachments import current_attachments
from app.memory.overview import build_memory_overview_answer
from app.memory.conversation_context import load_conversation_context
from app.rag.query_context import query_context_scope


logger = logging.getLogger(__name__)

_RUNNING_FEEDBACK = {
    "supervisor_agent": "正在识别请求类型并选择处理步骤",
    "memory_agent": "正在查找与本轮对话相关的历史上下文",
    "rag_agent": "正在查找知识库中的相关资料",
    "tool_agent": "正在选择工具并等待调用结果",
    "gather_agent": "正在汇总各步骤返回的上下文",
    "writer_agent": "正在整理上下文并生成回答",
    "reviewer_agent": "正在检查回答内容",
}
_SKIPPED_FEEDBACK = {
    "memory_agent": "本轮未执行记忆召回",
    "rag_agent": "本轮未执行知识库检索",
    "tool_agent": "本轮未调用工具",
}


def _tool_feedback_counts(results: list[dict[str, object]]) -> tuple[int, int]:
    failed = pending = 0
    for result in results:
        if not isinstance(result, dict) or result.get("needsClarification"):
            continue
        operation = result.get("operation") if isinstance(result.get("operation"), dict) else {}
        if result.get("ok") is False or result.get("policyBlocked") or operation.get("status") == "failed":
            failed += 1
        elif result.get("ok") and result.get("requiresConfirmation") and operation.get("status") == "pending":
            # requiresConfirmation also marks failed write requests. Only an
            # actual pending operation can offer confirmation to the user.
            pending += 1
    return failed, pending


def _step_outcome(agent: str, trace: dict[str, object]) -> tuple[str, str]:
    # Reviewer skipped means the optional model review was disabled; local
    # content rules still ran. Do not describe that real check as unexecuted.
    if agent == "reviewer_agent":
        if trace.get("approved") is False:
            return "failed", "回答内容检查未通过"
        if trace.get("modelReviewUsed") is False or ("modelReviewUsed" not in trace and trace.get("skipped")):
            return "skipped", "本地规则检查完成，本轮无需模型复核"
        return "completed", "模型复核完成"
    if trace.get("skipped"):
        return "skipped", _SKIPPED_FEEDBACK.get(agent, "本轮未执行此步骤")
    if agent == "supervisor_agent":
        return "completed", "已确定本轮需要执行的处理步骤"
    if agent == "memory_agent":
        return "completed", (
            "已整理历史上下文供本轮参考" if trace.get("memoryContextChars")
            else "未召回可用的历史上下文"
        )
    if agent == "rag_agent":
        queries = int(trace.get("queryCount") or 0)
        contexts = int(trace.get("contextCount") or 0)
        if not contexts:
            return "completed", (
                f"已检索 {queries} 次，未命中相关资料片段" if queries
                else "本次未返回相关资料片段"
            )
        return "completed", f"已检索 {queries} 次，取得 {contexts} 个相关资料片段"
    if agent == "tool_agent":
        count = int(trace.get("toolCount") or 0)
        if not count:
            return "completed", "本次未返回工具结果"
        details = []
        if trace.get("failedToolCount"):
            details.append(f"{int(trace['failedToolCount'])} 个失败")
        if trace.get("pendingToolCount"):
            details.append(f"{int(trace['pendingToolCount'])} 个待确认")
        return "completed", f"已返回 {count} 个工具结果" + (f"，其中 {'、'.join(details)}" if details else "")
    if agent == "gather_agent":
        return "completed", "已汇总各步骤返回的上下文"
    if agent == "writer_agent":
        return "completed", "回答生成完成" if trace.get("answerChars") else "本轮未生成回答内容"
    return "completed", "此步骤已结束"


def _progress_node(agent: str):
    """Observe real node boundaries, without publishing input/output contents."""
    def decorate(function):
        @wraps(function)
        def observed(state):
            started = perf_counter()
            emit_step(agent, "running", summary=_RUNNING_FEEDBACK.get(agent))
            try:
                result = function(state)
            except Exception:
                emit_step(agent, "failed", duration_ms=_elapsed_ms(started), summary="步骤执行失败")
                raise
            traces = [
                *(result.get("agent_trace") or []),
                *(item.get("trace") or {} for item in result.get("agent_outputs") or []),
            ]
            trace = next((item for item in reversed(traces) if item.get("agent") == agent), {})
            status, summary = _step_outcome(agent, trace)
            emit_step(
                agent, status,
                duration_ms=_elapsed_ms(started), summary=summary,
                timings=trace.get("timings"),
            )
            return result
        return observed
    return decorate


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
    supplemental_context: str
    candidate_memories: list[dict[str, object]]
    answer: str
    intent: str
    review: dict[str, object]
    review_attempts: int
    review_feedback: str
    agent_trace: list[dict[str, object]]
    agent_outputs: Annotated[list[dict[str, object]], operator.add]


@_progress_node("supervisor_agent")
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


@_progress_node("memory_agent")
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


@_progress_node("rag_agent")
def rag_agent_node(state: MultiAgentState) -> MultiAgentState:
    started = perf_counter()
    with query_context_scope(state["user_id"], state["conversation_id"], state["question"]):
        result = run_rag_agent(
            state["question"],
            state["knowledge_base_ids"],
            bool(state["supervisor"].get("use_rag", False)),
        )
    contexts = result.get("contexts") or []
    queries = result.get("queries") or []
    sufficiency = result.get("sufficiency") or {}
    decision = result.get("decision") or {}
    retrieval_plan = result.get("retrievalPlan") or {}
    logger.info(
        "context_graph timing rag_done. user_id=%s conversation_id=%s elapsed=%.3fs skipped=%s contexts=%s queries=%s mode=%s candidates=%s rerank_selected=%s",
        state["user_id"],
        state["conversation_id"],
        perf_counter() - started,
        result.get("skipped"),
        len(contexts),
        len(queries),
        retrieval_plan.get("mode", "skipped"),
        retrieval_plan.get("candidateCount", 0),
        retrieval_plan.get("selectedCount", 0),
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
                "timings": sanitize_timings(result.get("timings")),
                "retrievalPlan": retrieval_plan,
                "queryCount": len(queries),
                "sufficient": sufficiency.get("sufficient"),
                "skipped": result.get("skipped", False),
                "decision": decision,
                "elapsedMs": _elapsed_ms(started),
            },
        }],
    }


@_progress_node("writer_agent")
def writer_agent_node(state: MultiAgentState) -> MultiAgentState:
    started = perf_counter()
    from app.memory.fact_store import enabled as facts_enabled
    has_work_summary_tool = _has_successful_tool(state["tool_results"], "work_summary") and not facts_enabled()
    memory_context = "" if has_work_summary_tool else state["memory_context"]
    attachments = current_attachments(state["user_id"])
    writer_tool_context = "\n\n".join(
        item for item in [state["tool_context"], state["supplemental_context"],
                          attachments.context() if attachments else ""] if item
    )
    writer_input = _writer_input_debug(
        memory_context,
        writer_tool_context,
        state["contexts"],
        excluded_memory_context_chars=len(state["memory_context"]) if has_work_summary_tool else 0,
        memory_exclusion_reason="work_summary_tool_takes_priority" if has_work_summary_tool else "",
    )
    overview_answer = build_memory_overview_answer(
        state["question"],
        state["memory"],
        state["tool_results"],
    )
    answer = overview_answer or run_writer_agent(
        state["question"],
        state["contexts"],
        memory_context,
        writer_tool_context,
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
                "deterministicMemoryOverview": bool(overview_answer),
                "elapsedMs": _elapsed_ms(started),
            },
        ],
    }


@_progress_node("tool_agent")
def tool_agent_node(state: MultiAgentState) -> MultiAgentState:
    started = perf_counter()
    result = run_tool_agent(
        state["user_id"],
        state["conversation_id"],
        state["question"],
        bool(state["supervisor"].get("use_tools", False)),
        knowledge_base_ids=state["knowledge_base_ids"],
    )
    tool_results = result.get("tool_results") or []
    failed_tool_count, pending_tool_count = _tool_feedback_counts(tool_results)
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
                "failedToolCount": failed_tool_count,
                "pendingToolCount": pending_tool_count,
                "skipped": result.get("skipped", False),
                "decision": decision,
                "elapsedMs": _elapsed_ms(started),
            },
        }],
    }


def route_after_supervisor(state: MultiAgentState) -> list[Send]:
    supervisor = state["supervisor"]
    sends = []
    if supervisor.get("use_memory", False):
        sends.append(Send("memory_agent", state))
    if supervisor.get("use_rag", False):
        sends.append(Send("rag_agent", state))
    if supervisor.get("use_tools", False):
        sends.append(Send("tool_agent", state))
    active = {item.node for item in sends}
    for agent in ("memory_agent", "rag_agent", "tool_agent"):
        if agent not in active:
            emit_step(agent, "skipped", duration_ms=0, summary=_SKIPPED_FEEDBACK[agent])
    # No specialist work is a valid plan. Keep continuity/governance in gather,
    # rather than forcing a memory node solely to connect the graph.
    return sends or [Send("gather_agent", state)]


@_progress_node("gather_agent")
def gather_agent_node(state: MultiAgentState) -> MultiAgentState:
    started = perf_counter()
    outputs = state.get("agent_outputs", [])
    by_agent = {str(item.get("agent")): item for item in outputs}
    memory_output = by_agent.get("memory_agent")
    if memory_output is None or (memory_output.get("trace") or {}).get("skipped"):
        memory_output = load_conversation_context(state["user_id"], state["conversation_id"])
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
                "elapsedMs": _elapsed_ms(started),
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


@_progress_node("reviewer_agent")
def reviewer_agent_node(state: MultiAgentState) -> MultiAgentState:
    started = perf_counter()
    review_context = "\n\n".join(
        item for item in [state["memory_context"], state["supplemental_context"]] if item
    )
    try:
        review = run_reviewer_agent(
            state["question"],
            state["answer"],
            state["contexts"],
            review_context,
            memory=state["memory"],
            rag_sufficiency=state["rag_sufficiency"],
            tool_results=state["tool_results"],
        )
    except TypeError:
        review = run_reviewer_agent(
            state["question"],
            state["answer"],
            state["contexts"],
            review_context,
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
                **({"modelReviewUsed": review["modelReviewUsed"]} if "modelReviewUsed" in review else {}),
                "elapsedMs": _elapsed_ms(started),
            },
        ],
    }


# 记忆写入不再作为图节点串行执行，由 API 层在响应返回后异步完成（见 app/api/chat.py）。
def route_after_review(state: MultiAgentState) -> str:
    if state["review"].get("approved", True):
        return END
    from app.core.runtime_config import current_review_config
    if state["review_attempts"] <= current_review_config()[1]:
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
    supplemental_context: str = "",
) -> dict:
    return multi_agent_graph.invoke(
        _initial_state(user_id, conversation_id, question, knowledge_base_ids, supplemental_context)
    )


def prepare_multi_agent_context(
    user_id: str,
    conversation_id: str | None,
    question: str,
    knowledge_base_ids: list[str],
    supplemental_context: str = "",
) -> dict:
    return context_graph.invoke(
        _initial_state(user_id, conversation_id, question, knowledge_base_ids, supplemental_context)
    )


def _initial_state(
    user_id: str,
    conversation_id: str | None,
    question: str,
    knowledge_base_ids: list[str],
    supplemental_context: str = "",
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
        "supplemental_context": supplemental_context,
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
    excluded_memory_context_chars: int = 0,
    memory_exclusion_reason: str = "",
) -> dict[str, object]:
    combined_context = "\n\n".join(item for item in [memory_context, tool_context] if item)
    return {
        "memoryContextChars": len(memory_context),
        "memoryContext": _preview(memory_context, 6000),
        "excludedMemoryContextChars": excluded_memory_context_chars,
        "memoryExclusionReason": memory_exclusion_reason,
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


def _has_successful_tool(tool_results: list[dict[str, object]], tool_name: str) -> bool:
    return any(
        isinstance(item, dict) and item.get("tool") == tool_name and item.get("ok")
        for item in tool_results
    )
