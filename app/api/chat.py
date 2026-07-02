import json
import logging
import re
from time import perf_counter

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from app.core.runtime_config import temporary_ai_config
from app.core.config import settings
from app.agents.reviewer_agent import run_reviewer_agent
from app.graphs.multi_agent_graph import prepare_multi_agent_context, run_multi_agent_graph
from app.llm.client import stream_answer
from app.memory.store import remember_interaction
from app.schemas.chat import ChatRequest, ChatResponse, Source, Usage
from app.tools.task_tool import extract_candidate_tasks
from app.tools.tool_registry import select_tools

router = APIRouter()
logger = logging.getLogger(__name__)


def write_memory_in_background(
    user_id: str,
    conversation_id: str | None,
    question: str,
    answer: str,
    ai_config: dict[str, str] | None = None,
) -> None:
    """在 HTTP 响应返回后执行记忆写入（结构化抽取 / 摘要压缩 / 向量写入），不阻塞用户。"""
    started = perf_counter()
    try:
        with temporary_ai_config(ai_config):
            write_result = remember_interaction(user_id, conversation_id, question, answer)
        logger.info(
            "Background memory write done. user_id=%s conversation_id=%s elapsed=%.3fs candidate_memories=%s",
            user_id,
            conversation_id,
            perf_counter() - started,
            len(write_result.get("candidateMemories") or []),
        )
    except Exception:
        logger.exception(
            "Background memory write failed. user_id=%s conversation_id=%s",
            user_id,
            conversation_id,
        )


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    try:
        with temporary_ai_config(request.ai_config):
            result = run_multi_agent_graph(
                request.user_id,
                request.conversation_id,
                request.message,
                request.knowledge_base_ids,
            )
    except Exception as exc:
        logger.exception("LangGraph chat failed. user_id=%s conversation_id=%s", request.user_id, request.conversation_id)
        raise HTTPException(status_code=502, detail=f"LangGraph 调用失败：{exc}") from exc

    contexts = result.get("contexts") or []
    tool_results = result.get("tool_results") or []
    answer = _image_generate_answer(request.message, tool_results)
    if answer is None:
        answer = _append_tool_artifacts(
            _strip_unbacked_image_markdown(result.get("answer") or "", tool_results),
            tool_results,
        )
    background_tasks.add_task(
        write_memory_in_background,
        request.user_id,
        request.conversation_id,
        request.message,
        answer,
        request.ai_config,
    )
    agent_trace = [
        *(result.get("agent_trace") or []),
        {"agent": "memory_agent", "action": "write", "deferred": True},
    ]
    return ChatResponse(
        answer=answer,
        sources=[
            Source(docId=item["doc_id"], title=item["title"], chunk=item["chunk"])
            for item in contexts
        ],
        usage=Usage(promptTokens=0, completionTokens=0),
        agentTrace=agent_trace,
        toolResults=tool_results,
        candidateMemories=[],
        candidateTasks=extract_candidate_tasks(request.message),
        review=result.get("review") or {},
    )


@router.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    request_started = perf_counter()
    logger.info(
        "chat_stream timing start. user_id=%s conversation_id=%s message_chars=%s knowledge_bases=%s",
        request.user_id,
        request.conversation_id,
        len(request.message or ""),
        len(request.knowledge_base_ids or []),
    )
    prepare_started = perf_counter()
    with temporary_ai_config(request.ai_config):
        state = prepare_multi_agent_context(
            request.user_id,
            request.conversation_id,
            request.message,
            request.knowledge_base_ids,
        )
    logger.info(
        "chat_stream timing prepare_context_done. user_id=%s conversation_id=%s elapsed=%.3fs total=%.3fs trace_agents=%s",
        request.user_id,
        request.conversation_id,
        perf_counter() - prepare_started,
        perf_counter() - request_started,
        [item.get("agent") for item in state.get("agent_trace", []) if isinstance(item, dict)],
    )
    contexts = state.get("contexts") or []
    memory_context = str(state.get("memory_context") or "")
    tool_context = str(state.get("tool_context") or "")
    combined_context = "\n\n".join(
        item for item in [
            memory_context,
            tool_context,
        ] if item
    )
    writer_input = _writer_input_debug(memory_context, tool_context, contexts, combined_context)

    final_answer_holder: dict[str, str] = {}

    def generate():
        stream_started = perf_counter()
        answer_parts = []
        agent_trace = list(state.get("agent_trace") or [])
        review: dict[str, object] = {"approved": True, "reason": "streaming mode"}
        try:
            with temporary_ai_config(request.ai_config):
                image_answer = _image_generate_answer(request.message, state.get("tool_results") or [])
                if image_answer is not None:
                    answer = image_answer
                    answer_parts.append(answer)
                    logger.info(
                        "chat_stream timing first_delta. user_id=%s conversation_id=%s elapsed=%.3fs total=%.3fs source=image_answer",
                        request.user_id,
                        request.conversation_id,
                        perf_counter() - stream_started,
                        perf_counter() - request_started,
                    )
                    yield _sse("delta", {"text": answer})
                else:
                    writer_started = perf_counter()
                    first_delta_sent = False
                    for chunk in stream_answer(request.message, contexts, combined_context):
                        answer_parts.append(chunk)
                        if not first_delta_sent:
                            first_delta_sent = True
                            logger.info(
                                "chat_stream timing first_delta. user_id=%s conversation_id=%s elapsed=%.3fs total=%.3fs source=writer",
                                request.user_id,
                                request.conversation_id,
                                perf_counter() - writer_started,
                                perf_counter() - request_started,
                            )
                        yield _sse("delta", {"text": chunk})
                    logger.info(
                        "chat_stream timing writer_done. user_id=%s conversation_id=%s elapsed=%.3fs total=%.3fs chunks=%s chars=%s",
                        request.user_id,
                        request.conversation_id,
                        perf_counter() - writer_started,
                        perf_counter() - request_started,
                        len(answer_parts),
                        sum(len(part) for part in answer_parts),
                    )
                    answer = "".join(answer_parts)
                    answer = _strip_unbacked_image_markdown(answer, state.get("tool_results") or [])
                    final_answer = _append_tool_artifacts(answer, state.get("tool_results") or [])
                    artifact_text = final_answer[len(answer):]
                    if artifact_text:
                        answer = final_answer
                        yield _sse("delta", {"text": artifact_text})
                agent_trace.append({
                    "agent": "writer_agent",
                    "answerChars": len(answer),
                    "rewriteAttempt": 0,
                    "streaming": True,
                    "writerInput": writer_input,
                })
                review_started = perf_counter()
                review = run_reviewer_agent(
                    request.message,
                    answer,
                    contexts,
                    memory_context,
                )
                logger.info(
                    "chat_stream timing review_done. user_id=%s conversation_id=%s elapsed=%.3fs total=%.3fs skipped=%s approved=%s",
                    request.user_id,
                    request.conversation_id,
                    perf_counter() - review_started,
                    perf_counter() - request_started,
                    review.get("skipped"),
                    review.get("approved"),
                )
                agent_trace.append({
                    "agent": "reviewer_agent",
                    "approved": review.get("approved"),
                    "score": review.get("score"),
                    "skipped": review.get("skipped", False),
                    "streaming": True,
                })

                if not review.get("approved", True) and settings.reviewer_max_retry >= 1:
                    rewritten_answer = []
                    rewrite_prompt = (
                        f"{request.message}\n\n"
                        "上一次回答没有通过审核，请根据以下审核意见重新回答：\n"
                        f"{review.get('rewrite_instruction') or review.get('reason') or ''}"
                    )
                    yield _sse("delta", {"text": "\n\n"})
                    for chunk in stream_answer(rewrite_prompt, contexts, combined_context):
                        rewritten_answer.append(chunk)
                        yield _sse("delta", {"text": chunk})
                    answer = "".join(rewritten_answer)
                    final_answer = _append_tool_artifacts(answer, state.get("tool_results") or [])
                    artifact_text = final_answer[len(answer):]
                    if artifact_text:
                        answer = final_answer
                        yield _sse("delta", {"text": artifact_text})
                    agent_trace.append({
                        "agent": "writer_agent",
                        "answerChars": len(answer),
                        "rewriteAttempt": 1,
                        "streaming": True,
                        "writerInput": writer_input,
                    })
                    review = run_reviewer_agent(
                        request.message,
                        answer,
                        contexts,
                        memory_context,
                    )
                    agent_trace.append({
                        "agent": "reviewer_agent",
                        "approved": review.get("approved"),
                        "score": review.get("score"),
                        "skipped": review.get("skipped", False),
                        "streaming": True,
                    })

                # 记忆写入延迟到响应结束后执行（见 StreamingResponse 的 background 参数）。
                final_answer_holder["answer"] = answer
                agent_trace.append({
                    "agent": "memory_agent",
                    "action": "write",
                    "deferred": True,
                })
                meta = {
                    "sources": [
                        {"docId": item["doc_id"], "title": item["title"], "chunk": item["chunk"]}
                        for item in contexts
                    ],
                    "agentTrace": agent_trace,
                    "toolResults": state.get("tool_results") or [],
                    "candidateMemories": [],
                    "candidateTasks": extract_candidate_tasks(request.message),
                    "review": review,
                }
                meta_started = perf_counter()
                yield _sse("meta", meta)
                yield _sse("done", {})
                logger.info(
                    "chat_stream timing done. user_id=%s conversation_id=%s elapsed=%.3fs total=%.3fs",
                    request.user_id,
                    request.conversation_id,
                    perf_counter() - meta_started,
                    perf_counter() - request_started,
                )
        except Exception as exc:
            logger.exception("Streaming chat failed. user_id=%s conversation_id=%s", request.user_id, request.conversation_id)
            yield _sse("error", {"message": f"DeepSeek API 调用失败：{exc}"})

    def write_memory_after_stream() -> None:
        answer = final_answer_holder.get("answer")
        if not answer:
            return
        write_memory_in_background(
            request.user_id,
            request.conversation_id,
            request.message,
            answer,
            request.ai_config,
        )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream; charset=utf-8",
        background=BackgroundTask(write_memory_after_stream),
    )


def _sse(event: str, data: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _writer_input_debug(
    memory_context: str,
    tool_context: str,
    contexts: list[dict[str, str]],
    combined_context: str,
) -> dict[str, object]:
    return {
        "memoryContextChars": len(memory_context),
        "memoryContext": _clip(memory_context, 6000),
        "toolContextChars": len(tool_context),
        "toolContext": _clip(tool_context, 6000),
        "ragContextCount": len(contexts),
        "ragContexts": [
            {
                "title": item.get("title", ""),
                "chunkPreview": _clip(item.get("chunk", ""), 500),
            }
            for item in contexts[:5]
        ],
        "combinedContextChars": len(combined_context),
        "combinedContext": _clip(combined_context, 10000),
    }


IMAGE_MARKDOWN_RE = re.compile(r"!\[[^\]]*]\([^)\s]+(?:\s+\"[^\"]*\")?\)")
IMAGE_ARTIFACT_TOOLS = {"image_generate", "image_edit"}


def _append_tool_artifacts(answer: str, tool_results: list[dict[str, object]]) -> str:
    artifacts: list[str] = []
    for item in tool_results:
        if item.get("tool") not in IMAGE_ARTIFACT_TOOLS or not item.get("ok"):
            continue
        content = str(item.get("content") or "")
        for match in IMAGE_MARKDOWN_RE.findall(content):
            if match not in artifacts:
                artifacts.append(match)
    if not artifacts:
        return answer

    cleaned = IMAGE_MARKDOWN_RE.sub("", answer).rstrip()
    suffix = "\n\n" + "\n\n".join(artifacts)
    return f"{cleaned}{suffix}"


def _image_generate_answer(question: str, tool_results: list[dict[str, object]]) -> str | None:
    edit_results = [item for item in tool_results if item.get("tool") == "image_edit"]
    if edit_results:
        for item in edit_results:
            content = str(item.get("content") or "").strip()
            if item.get("ok") and IMAGE_MARKDOWN_RE.search(content):
                return content
        return "\n\n".join(
            str(item.get("content") or "图片编辑失败，请查看工具调试信息。").strip()
            for item in edit_results
        ).strip() or "图片编辑失败，请查看工具调试信息。"

    image_results = [item for item in tool_results if item.get("tool") in IMAGE_ARTIFACT_TOOLS]
    if image_results:
        for item in image_results:
            content = str(item.get("content") or "").strip()
            if item.get("ok") and IMAGE_MARKDOWN_RE.search(content):
                return content
        return "\n\n".join(
            str(item.get("content") or "图片生成失败，请查看工具调试信息。").strip()
            for item in image_results
        ).strip() or "图片生成失败，请查看工具调试信息。"

    selected_tools = select_tools(question)
    if "image_generate" in selected_tools or "image_edit" in selected_tools:
        return "图片工具没有执行成功，所以我不会伪造图片链接。请查看 Agent Trace 里的 tool_agent 是否被跳过，或确认图片工具已启用。"
    return None


def _strip_unbacked_image_markdown(answer: str, tool_results: list[dict[str, object]]) -> str:
    has_image_tool_success = any(
        item.get("tool") in IMAGE_ARTIFACT_TOOLS and item.get("ok")
        for item in tool_results
    )
    if has_image_tool_success:
        return answer
    return IMAGE_MARKDOWN_RE.sub("", answer).strip()


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"
