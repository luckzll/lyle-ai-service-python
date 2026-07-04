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
from app.plans.plan_builder import build_plan_candidate
from app.projects.candidate_extractor import extract_project_candidates
from app.projects.project_context_builder import build_project_context
from app.projects.project_resolver import resolve_project
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
    project_context = _project_context_for_response(request.user_id, request.conversation_id, request.message, request.project_id)
    if project_context:
        agent_trace.append({
            "agent": "project_resolver",
            "resolver": project_context.get("resolver") or {},
            "contextSummary": (project_context.get("context") or {}).get("summary") if isinstance(project_context.get("context"), dict) else "",
        })
    project_candidates = extract_project_candidates(request.message, project_context)
    plan_candidate = build_plan_candidate(request.message, project_context)
    return ChatResponse(
        answer=answer,
        sources=[
            Source(
                docId=item["doc_id"],
                title=item["title"],
                chunk=item["chunk"],
                score=str(item.get("score") or ""),
                backend=str(item.get("backend") or ""),
                hitModes=str(item.get("hitModes") or ""),
                chunkIndex=str(item.get("chunkIndex") or ""),
                parentTitle=str(item.get("parentTitle") or ""),
                startChar=str(item.get("startChar") or ""),
                endChar=str(item.get("endChar") or ""),
            )
            for item in contexts
        ],
        usage=Usage(promptTokens=0, completionTokens=0),
        agentTrace=agent_trace,
        toolResults=tool_results,
        candidateMemories=[],
        candidateTasks=_candidate_tasks_for_response(request.message, tool_results),
        candidateGoals=project_candidates["candidateGoals"],
        candidateDecisions=project_candidates["candidateDecisions"],
        candidateOpenQuestions=project_candidates["candidateOpenQuestions"],
        candidatePlans=[plan_candidate] if plan_candidate else [],
        projectContext=project_context,
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
    project_context = _project_context_for_response(request.user_id, request.conversation_id, request.message, request.project_id)
    project_candidates = extract_project_candidates(request.message, project_context)
    plan_candidate = build_plan_candidate(request.message, project_context)
    candidate_plans = [plan_candidate] if plan_candidate else []
    memory_context = str(state.get("memory_context") or "")
    tool_context = str(state.get("tool_context") or "")
    tool_results = state.get("tool_results") if isinstance(state.get("tool_results"), list) else []
    has_work_summary_tool = any(
        isinstance(item, dict) and item.get("tool") == "work_summary" and item.get("ok")
        for item in tool_results
    )
    confirmation_guidance = _v3_confirmation_guidance(
        project_candidates["candidateGoals"],
        candidate_plans,
        tool_results,
    )
    combined_context = "\n\n".join(
        item for item in [
            "" if has_work_summary_tool else memory_context,
            tool_context,
            confirmation_guidance,
        ] if item
    )
    writer_input = _writer_input_debug(memory_context, tool_context, contexts, combined_context)

    final_answer_holder: dict[str, str] = {}

    def generate():
        stream_started = perf_counter()
        answer_parts = []
        used_v3_preview = False
        agent_trace = list(state.get("agent_trace") or [])
        if project_context:
            agent_trace.append({
                "agent": "project_resolver",
                "resolver": project_context.get("resolver") or {},
                "contextSummary": (project_context.get("context") or {}).get("summary") if isinstance(project_context.get("context"), dict) else "",
                "streaming": True,
            })
        review: dict[str, object] = {"approved": True, "reason": "streaming mode"}
        try:
            with temporary_ai_config(request.ai_config):
                image_answer = _image_generate_answer(request.message, state.get("tool_results") or [])
                if image_answer is not None:
                    answer = image_answer
                    answer_parts.append(answer)
                    logger.info(
                        "chat_stream timing answer_prepared. user_id=%s conversation_id=%s elapsed=%.3fs total=%.3fs source=image_answer",
                        request.user_id,
                        request.conversation_id,
                        perf_counter() - stream_started,
                        perf_counter() - request_started,
                    )
                else:
                    preview_answer = _v3_preview_answer(
                        project_candidates["candidateGoals"],
                        candidate_plans,
                        state.get("tool_results") if isinstance(state.get("tool_results"), list) else [],
                        project_context,
                    )
                    if preview_answer:
                        used_v3_preview = True
                        answer = preview_answer
                        answer_parts.append(answer)
                        logger.info(
                            "chat_stream timing answer_prepared. user_id=%s conversation_id=%s elapsed=%.3fs total=%.3fs source=v3_preview",
                            request.user_id,
                            request.conversation_id,
                            perf_counter() - stream_started,
                            perf_counter() - request_started,
                        )
                    else:
                        writer_started = perf_counter()
                        for chunk in stream_answer(request.message, contexts, combined_context):
                            answer_parts.append(chunk)
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
                    memory=state.get("memory") if isinstance(state.get("memory"), dict) else {},
                    rag_sufficiency=state.get("rag_sufficiency") if isinstance(state.get("rag_sufficiency"), dict) else {},
                    tool_results=state.get("tool_results") if isinstance(state.get("tool_results"), list) else [],
                    project_context=project_context,
                    candidate_goals=project_candidates["candidateGoals"],
                    candidate_plans=candidate_plans,
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

                if not used_v3_preview and not review.get("approved", True) and settings.reviewer_max_retry >= 1:
                    rewritten_answer = []
                    rewrite_prompt = (
                        f"{request.message}\n\n"
                        "上一次回答没有通过审核，请根据以下审核意见重新回答：\n"
                        f"{review.get('rewrite_instruction') or review.get('reason') or ''}"
                    )
                    rewrite_started = perf_counter()
                    for chunk in stream_answer(rewrite_prompt, contexts, combined_context):
                        rewritten_answer.append(chunk)
                    answer = "".join(rewritten_answer)
                    final_answer = _append_tool_artifacts(answer, state.get("tool_results") or [])
                    if final_answer != answer:
                        answer = final_answer
                    agent_trace.append({
                        "agent": "writer_agent",
                        "answerChars": len(answer),
                        "rewriteAttempt": 1,
                        "streaming": True,
                        "bufferedUntilReview": True,
                        "writerInput": writer_input,
                    })
                    review = run_reviewer_agent(
                        request.message,
                        answer,
                        contexts,
                        memory_context,
                        memory=state.get("memory") if isinstance(state.get("memory"), dict) else {},
                        rag_sufficiency=state.get("rag_sufficiency") if isinstance(state.get("rag_sufficiency"), dict) else {},
                        tool_results=state.get("tool_results") if isinstance(state.get("tool_results"), list) else [],
                        project_context=project_context,
                        candidate_goals=project_candidates["candidateGoals"],
                        candidate_plans=candidate_plans,
                    )
                    logger.info(
                        "chat_stream timing rewrite_review_done. user_id=%s conversation_id=%s elapsed=%.3fs total=%.3fs approved=%s",
                        request.user_id,
                        request.conversation_id,
                        perf_counter() - rewrite_started,
                        perf_counter() - request_started,
                        review.get("approved"),
                    )
                    agent_trace.append({
                        "agent": "reviewer_agent",
                        "approved": review.get("approved"),
                        "score": review.get("score"),
                        "skipped": review.get("skipped", False),
                        "streaming": True,
                    })

                if not review.get("approved", True):
                    answer = _review_blocked_answer(review)
                    agent_trace.append({
                        "agent": "writer_agent",
                        "action": "quality_blocked",
                        "answerChars": len(answer),
                        "streaming": True,
                    })

                logger.info(
                    "chat_stream timing first_delta. user_id=%s conversation_id=%s elapsed=%.3fs total=%.3fs source=review_gated",
                    request.user_id,
                    request.conversation_id,
                    perf_counter() - stream_started,
                    perf_counter() - request_started,
                )
                yield _sse("delta", {"text": answer})

                # 记忆写入延迟到响应结束后执行（见 StreamingResponse 的 background 参数）。
                final_answer_holder["answer"] = answer
                agent_trace.append({
                    "agent": "memory_agent",
                    "action": "write",
                    "deferred": True,
                })
                meta = {
                    "sources": [
                        {
                            "docId": item["doc_id"],
                            "title": item["title"],
                            "chunk": item["chunk"],
                            "score": str(item.get("score") or ""),
                            "backend": str(item.get("backend") or ""),
                            "hitModes": str(item.get("hitModes") or ""),
                            "chunkIndex": str(item.get("chunkIndex") or ""),
                            "parentTitle": str(item.get("parentTitle") or ""),
                            "startChar": str(item.get("startChar") or ""),
                            "endChar": str(item.get("endChar") or ""),
                        }
                        for item in contexts
                    ],
                    "agentTrace": agent_trace,
                    "toolResults": state.get("tool_results") or [],
                    "candidateMemories": [],
                    "candidateTasks": _candidate_tasks_for_response(request.message, state.get("tool_results") or []),
                    "candidateGoals": project_candidates["candidateGoals"],
                    "candidateDecisions": project_candidates["candidateDecisions"],
                    "candidateOpenQuestions": project_candidates["candidateOpenQuestions"],
                    "candidatePlans": candidate_plans,
                    "projectContext": project_context,
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
        except (GeneratorExit, BrokenPipeError, ConnectionResetError):
            logger.info(
                "Streaming chat cancelled by client. user_id=%s conversation_id=%s",
                request.user_id,
                request.conversation_id,
            )
            raise
        except Exception as exc:
            logger.exception("Streaming chat failed. user_id=%s conversation_id=%s", request.user_id, request.conversation_id)
            yield _sse("error", {"message": _friendly_stream_error(exc)})

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


def _friendly_stream_error(exc: Exception) -> str:
    text = str(exc)
    lowered = text.lower()
    if "timeout" in lowered or "timed out" in lowered:
        return "模型响应超时了，请稍后重试，或换一个更短的问题。"
    if "api key" in lowered or "unauthorized" in lowered or "401" in lowered:
        return "模型服务鉴权失败，请检查后端模型 API Key 配置。"
    if "rate" in lowered or "429" in lowered:
        return "模型服务当前限流了，请稍后再试。"
    return "模型服务暂时不可用，请稍后重试。"


def _review_blocked_answer(review: dict[str, object]) -> str:
    reason = str(review.get("reason") or "").strip()
    if not reason:
        reason = "回答没有通过质量检查"
    return (
        "这次回答被质量检查拦截了，所以我没有把未通过的版本发出来。\n\n"
        f"原因：{reason}\n\n"
        "你可以换个更明确的问题再试一次，或者打开调试信息查看 reviewer 的具体判断。"
    )


def _project_context_for_response(user_id: str, conversation_id: str | None, message: str, project_id: int | None = None) -> dict[str, object]:
    resolver = resolve_project(user_id, conversation_id, message, project_id)
    project_id = resolver.get("projectId")
    context = build_project_context(user_id, int(project_id)) if project_id else {}
    return {
        "resolver": resolver,
        "context": context,
    }


def _candidate_tasks_for_response(question: str, tool_results: list[dict[str, object]]) -> list[dict[str, object]]:
    has_pending_task_operation = any(
        item.get("requiresConfirmation")
        and isinstance(item.get("operation"), dict)
        and str(item["operation"].get("operationType") or "").startswith("todo.")
        for item in tool_results
    )
    if has_pending_task_operation:
        return []
    return extract_candidate_tasks(question)


def _v3_confirmation_guidance(
    candidate_goals: list[dict[str, object]],
    candidate_plans: list[dict[str, object]],
    tool_results: list[dict[str, object]],
) -> str:
    pending_operations = [
        item for item in tool_results
        if isinstance(item, dict)
        and item.get("requiresConfirmation")
        and isinstance(item.get("operation"), dict)
        and (item.get("operation") or {}).get("status") == "pending"
    ]
    lines: list[str] = []
    if candidate_goals:
        lines.append(
            "本轮将生成待确认目标卡。回答必须说明目标仍需用户在卡片中确认；"
            "禁止使用“已新增、已记录、已安排、已就位、已写入、为你新增、直接记录到项目”等已完成写入表述。"
        )
    if candidate_plans:
        lines.append(
            "本轮将生成待确认计划卡。回答必须说明计划草案需要用户点击“创建计划”；"
            "禁止声称计划已经创建到项目或已经开始执行。"
        )
    if pending_operations:
        lines.append(
            "本轮已有待确认操作卡。回答只提示用户在卡片中确认或拒绝；"
            "不要自问自答，不要说已经正式创建或已经执行。"
        )
    return "\n".join(f"[V3确认规则] {line}" for line in lines)


def _v3_preview_answer(
    candidate_goals: list[dict[str, object]],
    candidate_plans: list[dict[str, object]],
    tool_results: list[dict[str, object]],
    project_context: dict[str, object] | None,
) -> str:
    resolver = project_context.get("resolver") if isinstance(project_context, dict) else {}
    project_name = str(resolver.get("projectName") or "") if isinstance(resolver, dict) else ""
    pending_operations = [
        item for item in tool_results
        if isinstance(item, dict)
        and item.get("requiresConfirmation")
        and isinstance(item.get("operation"), dict)
        and (item.get("operation") or {}).get("status") == "pending"
    ]
    parts: list[str] = []
    project_text = f"（{project_name}）" if project_name else ""
    if candidate_goals:
        goal = candidate_goals[0]
        title = str(goal.get("title") or "目标")
        horizon = str(goal.get("horizon") or "")
        horizon_text = {
            "day": "今日",
            "week": "本周",
            "month": "本月",
            "quarter": "季度",
            "long_term": "长期",
        }.get(horizon, horizon or "未设置")
        parts.append(
            f"已生成待确认目标{project_text}：{title}。周期：{horizon_text}。"
            "请在下方目标卡片中确认或忽略，确认后才会写入项目。"
        )
    if candidate_plans:
        plan = candidate_plans[0]
        title = str(plan.get("title") or "推进计划")
        step_count = len(plan.get("steps") or [])
        parts.append(
            f"已生成待确认计划{project_text}：{title}，共 {step_count} 步。"
            "请在下方计划卡片中点击“创建计划”或忽略，创建后才会进入项目计划列表。"
        )
    if pending_operations:
        operation = pending_operations[0].get("operation") or {}
        summary = str(operation.get("summary") or "待确认操作")
        parts.append(
            f"已生成待确认操作：{summary}。"
            "请在下方操作卡片中点击“确认执行”或“拒绝”，确认前不会创建或执行。"
        )
    return "\n\n".join(parts)


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
                "score": str(item.get("score") or ""),
                "backend": str(item.get("backend") or ""),
                "hitModes": str(item.get("hitModes") or ""),
                "chunkIndex": str(item.get("chunkIndex") or ""),
                "parentTitle": str(item.get("parentTitle") or ""),
                "startChar": str(item.get("startChar") or ""),
                "endChar": str(item.get("endChar") or ""),
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
