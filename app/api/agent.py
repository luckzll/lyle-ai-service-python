import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.api.chat import _memory_usage_payload, write_memory_in_background
from app.core.runtime_config import temporary_ai_config
from app.rag.request_scope import request_retrieval_scope
from app.graphs.multi_agent_graph import run_multi_agent_graph
from app.schemas.chat import ChatRequest, ChatResponse
from app.security.chat_debug import build_safe_chat_debug
from app.usage.context import activate_model_call_context, usage_totals

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/agent/chat", response_model=ChatResponse)
def agent_chat(request: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    usage_context = activate_model_call_context(request)
    try:
        with temporary_ai_config(request.ai_config), request_retrieval_scope(
            request.user_id, request.knowledge_base_ids, request.knowledgeScope, request.conversation_id
        ):
            result = run_multi_agent_graph(
                request.user_id,
                request.conversation_id,
                request.message,
                request.knowledge_base_ids,
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Agent chat failed. user_id=%s conversation_id=%s", request.user_id, request.conversation_id)
        raise HTTPException(status_code=502, detail=f"LangGraph 调用失败：{exc}") from exc

    background_tasks.add_task(
        write_memory_in_background,
        request.user_id,
        request.conversation_id,
        request.message,
        result.get("answer") or "",
        request.ai_config,
        False,
        usage_context,
    )
    contexts = result.get("contexts") or []
    usage_payload = {
        "promptTokens": int(usage_totals(usage_context)["prompt_tokens"]),
        "completionTokens": int(usage_totals(usage_context)["completion_tokens"]),
        "callCount": int(usage_totals(usage_context)["call_count"]),
        "estimatedCostMicros": int(usage_totals(usage_context)["estimated_cost_micros"]),
        "measurementSource": str(usage_totals(usage_context)["measurement_source"]),
    }
    memory_usage = _memory_usage_payload(
        result.get("memory") if isinstance(result.get("memory"), dict) else {},
        included=True,
    )
    safe_debug = build_safe_chat_debug(
        knowledge_base_ids=request.knowledge_base_ids,
        contexts=contexts,
        agent_trace=result.get("agent_trace") or [],
        tool_results=result.get("tool_results") or [],
        review=result.get("review") or {},
        usage=usage_payload,
        used_memories=memory_usage["usedMemories"],
        memory_exclusions=memory_usage["memoryExclusions"],
        rag_queries=result.get("rag_queries") or [],
        rag_sufficiency=result.get("rag_sufficiency") or {},
    )
    return ChatResponse(
        answer=result.get("answer") or "",
        **safe_debug,
    )
