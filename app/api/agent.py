import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.api.chat import write_memory_in_background
from app.graphs.multi_agent_graph import run_multi_agent_graph
from app.schemas.chat import ChatRequest, ChatResponse, Source, Usage

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/agent/chat", response_model=ChatResponse)
def agent_chat(request: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    try:
        result = run_multi_agent_graph(
            request.user_id,
            request.conversation_id,
            request.message,
            request.knowledge_base_ids,
        )
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
    )
    contexts = result.get("contexts") or []
    return ChatResponse(
        answer=result.get("answer") or "",
        sources=[
            Source(docId=item["doc_id"], title=item["title"], chunk=item["chunk"])
            for item in contexts
        ],
        usage=Usage(promptTokens=0, completionTokens=0),
        agentTrace=result.get("agent_trace") or [],
        toolResults=result.get("tool_results") or [],
        review=result.get("review") or {},
    )
