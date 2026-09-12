from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.security.secret_masker import mask_secrets
from app.chat_progress import sanitize_timings


DEBUG_VERSION = "chat-debug-v1"
MAX_TOTAL_TEXT_CHARS = 12_000
MAX_KNOWLEDGE_BASES = 20
MAX_SOURCES = 5
MAX_AGENT_TRACE_ITEMS = 16
MAX_TOOL_RESULTS = 8
MAX_USED_MEMORIES = 8
MAX_RAG_QUERIES = 4


@dataclass
class _DebugBudget:
    remaining: int = MAX_TOTAL_TEXT_CHARS
    truncated: bool = False

    def text(self, value: object, limit: int) -> str:
        if value is None:
            raw = ""
        elif isinstance(value, (str, int, float, bool)):
            raw = str(value)
        else:
            # Never stringify nested extensible objects into a supposedly safe
            # display field. Their values may be raw provider/tool payloads.
            self.truncated = True
            return ""
        masked = mask_secrets(raw)
        allowed = max(0, min(limit, self.remaining))
        if len(masked) > allowed:
            self.truncated = True
            masked = masked[:allowed]
        self.remaining -= len(masked)
        return masked

    def limited(self, values: object, limit: int) -> list[Any]:
        items = values if isinstance(values, list) else []
        if len(items) > limit:
            self.truncated = True
        return items[:limit]


def build_safe_chat_debug(
    *,
    knowledge_base_ids: object,
    contexts: object,
    agent_trace: object,
    tool_results: object,
    review: object,
    usage: object,
    used_memories: object,
    memory_exclusions: object,
    rag_queries: object = None,
    rag_sufficiency: object = None,
) -> dict[str, object]:
    """Build the only chat diagnostics allowed to cross the Python API boundary.

    The caller supplies knowledge-base identifiers already validated by the Java
    API. This function still treats every nested trace/tool value as untrusted and
    reconstructs the response from a small allowlist. In particular, writer input,
    prompts, raw tool payloads/inputs/results, model reasoning and credentials are
    never copied into the returned object.
    """
    budget = _DebugBudget()
    safe_ids = _sanitize_ids(knowledge_base_ids, budget)
    raw_contexts = contexts if isinstance(contexts, list) else []
    raw_trace = agent_trace if isinstance(agent_trace, list) else []
    retrieval_scope = _retrieval_scope(
        safe_ids,
        raw_contexts,
        raw_trace,
        rag_queries,
        rag_sufficiency,
        budget,
    )
    # Preserve the high-level decision, actual injected-memory list and usage
    # before spending the remaining budget on previews and source excerpts.
    safe_used_memories = _sanitize_used_memories(used_memories, budget)
    safe_usage = _sanitize_usage(usage, budget)
    safe_memory_exclusions = _sanitize_memory_exclusions(memory_exclusions)
    safe_trace = _sanitize_agent_trace(
        raw_trace,
        raw_contexts,
        rag_queries,
        rag_sufficiency,
        budget,
    )
    safe_tools = _sanitize_tool_results(tool_results, budget)
    safe_review = _sanitize_review(review, budget)
    safe_sources = _sanitize_sources(raw_contexts, budget)
    payload = {
        "debugVersion": DEBUG_VERSION,
        "knowledgeBaseIds": safe_ids,
        "retrievalScope": retrieval_scope,
        "sources": safe_sources,
        "agentTrace": safe_trace,
        "toolResults": safe_tools,
        "review": safe_review,
        "usage": safe_usage,
        "usedMemories": safe_used_memories,
        "memoryExclusions": safe_memory_exclusions,
    }
    payload["debugTruncated"] = budget.truncated
    return payload


def _sanitize_ids(value: object, budget: _DebugBudget) -> list[str]:
    values = value if isinstance(value, list) else []
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        identifier = budget.text(item, 128).strip()
        if not identifier or identifier in seen:
            continue
        seen.add(identifier)
        result.append(identifier)
        if len(result) >= MAX_KNOWLEDGE_BASES:
            if len(values) > len(result):
                budget.truncated = True
            break
    return result


def _sanitize_sources(contexts: object, budget: _DebugBudget) -> list[dict[str, object]]:
    sources: list[dict[str, object]] = []
    for raw in budget.limited(contexts, MAX_SOURCES):
        if not isinstance(raw, dict):
            continue
        sources.append({
            "docId": budget.text(raw.get("doc_id", raw.get("docId", "")), 128),
            "title": budget.text(raw.get("title", ""), 240),
            "chunk": budget.text(raw.get("chunk", ""), 1_200),
            "score": budget.text(raw.get("score", ""), 32),
            "backend": budget.text(raw.get("backend", ""), 48),
            "hitModes": budget.text(raw.get("hitModes", ""), 96),
            "chunkIndex": budget.text(raw.get("chunkIndex", ""), 32),
            "parentTitle": budget.text(raw.get("parentTitle", ""), 240),
            "startChar": budget.text(raw.get("startChar", ""), 32),
            "endChar": budget.text(raw.get("endChar", ""), 32),
        })
    return sources


def _sanitize_agent_trace(
    traces: object,
    contexts: list[object],
    rag_queries: object,
    rag_sufficiency: object,
    budget: _DebugBudget,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    raw_traces = budget.limited(traces, MAX_AGENT_TRACE_ITEMS)
    for raw in raw_traces:
        if not isinstance(raw, dict):
            continue
        agent = budget.text(raw.get("agent", "unknown"), 64)
        item: dict[str, object] = {"agent": agent}
        _copy_bool(raw, item, "skipped")
        _copy_int(raw, item, "elapsedMs")
        if agent in {"rag_agent", "writer_agent", "reviewer_agent", "gather_agent"} and isinstance(raw.get("timings"), dict):
            item["timings"] = sanitize_timings(raw["timings"])

        if agent == "supervisor_agent":
            _copy_text(raw, item, "intent", 48, budget)
            _copy_string_list(raw, item, "nextAgents", 8, 64, budget)
        elif agent == "memory_agent":
            _copy_text(raw, item, "action", 64, budget)
            _copy_int(raw, item, "candidateCount")
            _copy_int(raw, item, "memoryContextChars")
            if isinstance(raw.get("memorySummary"), dict):
                item["memorySummary"] = _sanitize_memory_summary(raw["memorySummary"], budget)
            if isinstance(raw.get("decision"), dict):
                item["decision"] = _sanitize_decision(raw["decision"], budget, memory=True)
        elif agent == "rag_agent":
            _copy_int(raw, item, "contextCount")
            _copy_int(raw, item, "queryCount")
            _copy_bool(raw, item, "sufficient")
            if isinstance(raw.get("retrievalPlan"), dict):
                item["retrievalPlan"] = _sanitize_retrieval_plan(raw["retrievalPlan"], budget)
            if isinstance(raw.get("decision"), dict):
                item["decision"] = _sanitize_decision(raw["decision"], budget)
            item["queries"] = _sanitize_queries(rag_queries, budget)
            item["sufficiency"] = _sanitize_sufficiency(rag_sufficiency, budget)
            item["sources"] = _source_summaries(contexts, budget)
        elif agent == "tool_agent":
            _copy_text(raw, item, "tool", 96, budget)
            _copy_int(raw, item, "toolCount")
            _copy_int(raw, item, "imageCount")
            _copy_string_list(raw, item, "selectedTools", 8, 96, budget)
            if isinstance(raw.get("decision"), dict):
                item["decision"] = _sanitize_decision(raw["decision"], budget, tool=True)
        elif agent == "gather_agent":
            _copy_string_list(raw, item, "receivedAgents", 8, 64, budget)
        elif agent == "writer_agent":
            # Keep only aggregate injection counts; prompt/context text never crosses.
            _copy_int(raw, item, "answerChars")
            _copy_int(raw, item, "rewriteAttempt")
            _copy_bool(raw, item, "streaming")
            _copy_bool(raw, item, "deterministicMemoryOverview")
            _copy_bool(raw, item, "bufferedUntilReview")
            if isinstance(raw.get("writerInput"), dict):
                item["writerInput"] = _sanitize_writer_input(raw["writerInput"], budget)
        elif agent == "reviewer_agent":
            _copy_bool(raw, item, "approved")
            _copy_float(raw, item, "score")
        else:
            # Unknown agents expose identity and coarse status only.
            _copy_text(raw, item, "action", 64, budget)
            _copy_text(raw, item, "status", 48, budget)
        result.append(item)
    return result


def _sanitize_retrieval_plan(raw: dict[str, object], budget: _DebugBudget) -> dict[str, object]:
    result: dict[str, object] = {}
    for key in ("mode", "fusionMode"):
        _copy_text(raw, result, key, 32, budget)
    for key in ("actualQueryCount", "candidateCount", "rerankSelectedCount", "parallelWorkers"):
        value = raw.get(key, raw.get("selectedCount") if key == "rerankSelectedCount" else None)
        if value is not None and not isinstance(value, bool):
            try:
                result[key] = max(0, int(value))
            except (TypeError, ValueError, OverflowError):
                pass
    if isinstance(raw.get("adaptiveUsed"), bool):
        result["adaptiveUsed"] = raw["adaptiveUsed"]
    cache = raw.get("candidateCache")
    if isinstance(cache, dict):
        safe_cache: dict[str, int] = {}
        for key in ("hit", "miss", "shared", "bypass"):
            value = cache.get(key)
            if value is None or isinstance(value, bool):
                continue
            try:
                safe_cache[key] = max(0, int(value))
            except (TypeError, ValueError, OverflowError):
                pass
        result["candidateCache"] = safe_cache
    resolution = raw.get("queryResolution")
    if isinstance(resolution, dict):
        safe_resolution: dict[str, object] = {}
        for key, limit in (("method", 64), ("reason", 160),
                           ("originalQuestion", 500), ("effectiveQuestion", 500)):
            _copy_text(resolution, safe_resolution, key, limit, budget)
        result["queryResolution"] = safe_resolution
    return result


def _sanitize_writer_input(raw: dict[str, object], budget: _DebugBudget) -> dict[str, object]:
    result: dict[str, object] = {}
    for key in (
        "memoryContextChars",
        "excludedMemoryContextChars",
        "toolContextChars",
        "ragContextCount",
        "combinedContextChars",
    ):
        _copy_int(raw, result, key)
    _copy_text(raw, result, "memoryExclusionReason", 160, budget)
    return result


def _sanitize_memory_summary(raw: dict[str, object], budget: _DebugBudget) -> dict[str, object]:
    result: dict[str, object] = {}
    for key in (
        "summaryChars",
        "shortTermCount",
        "structuredCount",
        "semanticCount",
        "vectorCount",
        "retrievedCount",
    ):
        _copy_int(raw, result, key)
    _copy_text(raw, result, "summaryPreview", 240, budget)
    result["shortTermPreview"] = [
        {
            "role": budget.text(item.get("role", ""), 24),
            "content": budget.text(item.get("content", ""), 160),
        }
        for item in budget.limited(raw.get("shortTermPreview"), 3)
        if isinstance(item, dict)
    ]
    result["structuredPreview"] = [
        budget.text(item, 160)
        for item in budget.limited(raw.get("structuredPreview"), 8)
    ]
    result["retrievedPreview"] = [
        budget.text(item, 180)
        for item in budget.limited(raw.get("retrievedPreview"), 3)
    ]
    return result


def _sanitize_decision(
    raw: dict[str, object],
    budget: _DebugBudget,
    *,
    memory: bool = False,
    tool: bool = False,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key in ("should_retrieve", "should_recall", "should_call_tools", "needs_clarification"):
        _copy_bool(raw, result, key)
    _copy_text(raw, result, "depth", 32, budget)
    _copy_text(raw, result, "reason", 240, budget)
    _copy_text(raw, result, "router", 32, budget)
    _copy_float(raw, result, "confidence")
    if tool:
        _copy_string_list(raw, result, "selected_tools", 8, 96, budget)
    if memory:
        _copy_text(raw, result, "mode", 32, budget)
    # tool_inputs, payload, arguments and similar raw values are intentionally absent.
    return result


def _sanitize_queries(value: object, budget: _DebugBudget) -> list[str]:
    return [budget.text(item, 240) for item in budget.limited(value, MAX_RAG_QUERIES)]


def _sanitize_sufficiency(value: object, budget: _DebugBudget) -> dict[str, object]:
    raw = value if isinstance(value, dict) else {}
    result: dict[str, object] = {}
    _copy_bool(raw, result, "sufficient")
    _copy_float(raw, result, "score")
    _copy_text(raw, result, "reason", 240, budget)
    return result


def _source_summaries(contexts: list[object], budget: _DebugBudget) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for raw in budget.limited(contexts, MAX_SOURCES):
        if not isinstance(raw, dict):
            continue
        result.append({
            "docId": budget.text(raw.get("doc_id", raw.get("docId", "")), 128),
            "title": budget.text(raw.get("title", ""), 200),
            "score": budget.text(raw.get("score", ""), 32),
            "backend": budget.text(raw.get("backend", ""), 48),
            "hitModes": budget.text(raw.get("hitModes", ""), 96),
        })
    return result


def _retrieval_scope(
    knowledge_base_ids: list[str],
    contexts: list[object],
    traces: list[object],
    rag_queries: object,
    rag_sufficiency: object,
    budget: _DebugBudget,
) -> dict[str, object]:
    rag_trace = next(
        (
            item for item in traces
            if isinstance(item, dict) and str(item.get("agent") or "") == "rag_agent"
        ),
        None,
    )
    requested = bool(knowledge_base_ids)
    executed = bool(rag_trace is not None and not bool(rag_trace.get("skipped", False)))
    query_count = _safe_int(rag_trace.get("queryCount")) if rag_trace else 0
    if query_count <= 0 and isinstance(rag_queries, list):
        query_count = len(rag_queries)
    context_count = len(contexts)
    reason = ""
    if rag_trace and isinstance(rag_trace.get("decision"), dict):
        reason = str(rag_trace["decision"].get("reason") or "")
    if not reason and isinstance(rag_sufficiency, dict):
        reason = str(rag_sufficiency.get("reason") or "")
    if not reason:
        if not requested:
            reason = "未选择知识库。"
        elif not executed:
            reason = "知识库检索未执行。"
        elif context_count == 0:
            reason = "知识库检索已执行，但未命中文档片段。"
        else:
            reason = "知识库检索已执行。"
    return {
        "requested": requested,
        "executed": executed,
        "skipped": not executed,
        "knowledgeBaseCount": len(knowledge_base_ids),
        "queryCount": max(0, query_count),
        "contextCount": context_count,
        "reason": budget.text(reason, 240),
    }


def _sanitize_tool_results(value: object, budget: _DebugBudget) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for raw in budget.limited(value, MAX_TOOL_RESULTS):
        if not isinstance(raw, dict):
            continue
        item: dict[str, object] = {
            "tool": budget.text(raw.get("tool", "unknown"), 96),
            "ok": bool(raw.get("ok", False)),
        }
        for key in (
            "elapsedMs",
            "extractedChars",
            "imageCount",
            "requestedCount",
            "completedCount",
            "failedCount",
            "count",
        ):
            _copy_int(raw, item, key)
        for key, limit in (
            ("status", 48),
            ("batchStatus", 48),
            ("attachmentId", 128),
            ("batchId", 128),
            ("operationType", 96),
            ("riskLevel", 48),
        ):
            _copy_text(raw, item, key, limit, budget)
        _copy_bool(raw, item, "requiresConfirmation")
        # Never promote an extensible tool body/error into durable diagnostics.
        # Tool implementations may put whole third-party responses in those
        # fields.  Only an explicit, intentionally display-safe summary crosses
        # this boundary; otherwise the UI derives a generic status from ``ok``.
        summary = raw.get("summary")
        if isinstance(summary, str) and summary.strip():
            item["summary"] = budget.text(summary, 320)
        operation = raw.get("operation")
        if isinstance(operation, dict):
            safe_operation: dict[str, object] = {}
            _copy_text(operation, safe_operation, "toolName", 96, budget)
            _copy_text(operation, safe_operation, "operationType", 96, budget)
            _copy_text(operation, safe_operation, "riskLevel", 48, budget)
            _copy_text(operation, safe_operation, "status", 48, budget)
            operation_summary = operation.get("summary")
            if isinstance(operation_summary, str) and operation_summary.strip():
                safe_operation["summary"] = budget.text(operation_summary, 240)
            for timestamp_key in ("createdAt", "updatedAt", "confirmedAt", "rejectedAt"):
                _copy_text(operation, safe_operation, timestamp_key, 80, budget)
            _copy_int(operation, safe_operation, "id")
            safe_payload = _sanitize_operation_payload(operation, budget)
            if safe_payload:
                safe_operation["payload"] = safe_payload
            if safe_operation:
                item["operation"] = safe_operation
        result.append(item)
    return result


def _sanitize_operation_payload(operation: dict[str, object], budget: _DebugBudget) -> dict[str, str]:
    raw = operation.get("payload") if isinstance(operation.get("payload"), dict) else {}
    operation_type = str(operation.get("operationType") or "")
    if operation_type == "todo.create":
        allowed = ("title", "dueAt")
    elif operation_type == "calendar.create":
        allowed = ("title", "startsAt")
    else:
        return {}
    return {
        key: budget.text(raw.get(key), 200 if key == "title" else 80)
        for key in allowed
        if raw.get(key) not in (None, "")
    }


def _sanitize_review(value: object, budget: _DebugBudget) -> dict[str, object]:
    raw = value if isinstance(value, dict) else {}
    result: dict[str, object] = {}
    _copy_bool(raw, result, "approved")
    _copy_bool(raw, result, "skipped")
    _copy_bool(raw, result, "ruleBlocked")
    _copy_float(raw, result, "score")
    _copy_text(raw, result, "reason", 320, budget)
    _copy_text(raw, result, "v3Rule", 96, budget)
    result["recalledMemoryRefs"] = _sanitize_memory_refs(raw.get("recalledMemoryRefs"), budget)
    result["objectStatuses"] = _sanitize_object_statuses(raw.get("objectStatuses"), budget)
    blocked = raw.get("blockedMemoryRef")
    if isinstance(blocked, dict):
        refs = _sanitize_memory_refs([blocked], budget)
        if refs:
            result["blockedMemoryRef"] = refs[0]
    return result


def _sanitize_memory_refs(value: object, budget: _DebugBudget) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for raw in budget.limited(value, MAX_USED_MEMORIES):
        if not isinstance(raw, dict):
            continue
        item: dict[str, object] = {}
        _copy_text(raw, item, "kind", 48, budget)
        _copy_int(raw, item, "id")
        _copy_text(raw, item, "schemaPath", 160, budget)
        _copy_text(raw, item, "status", 48, budget)
        if item:
            result.append(item)
    return result


def _sanitize_object_statuses(value: object, budget: _DebugBudget) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for raw in budget.limited(value, MAX_USED_MEMORIES):
        if not isinstance(raw, dict):
            continue
        item: dict[str, object] = {}
        _copy_text(raw, item, "kind", 48, budget)
        _copy_int(raw, item, "id")
        _copy_text(raw, item, "status", 48, budget)
        _copy_text(raw, item, "title", 160, budget)
        if item:
            result.append(item)
    return result


def _sanitize_usage(value: object, budget: _DebugBudget) -> dict[str, object]:
    raw = value.model_dump(by_alias=True) if hasattr(value, "model_dump") else value
    raw = raw if isinstance(raw, dict) else {}
    result: dict[str, object] = {
        "promptTokens": 0,
        "completionTokens": 0,
        "callCount": 0,
        "estimatedCostMicros": 0,
    }
    for key in ("promptTokens", "completionTokens", "callCount", "estimatedCostMicros"):
        _copy_int(raw, result, key)
    source = raw.get("measurementSource", "PENDING")
    result["measurementSource"] = budget.text(source, 48)
    return result


def _sanitize_used_memories(value: object, budget: _DebugBudget) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for raw in budget.limited(value, MAX_USED_MEMORIES):
        if not isinstance(raw, dict):
            continue
        item: dict[str, object] = {}
        for key, limit in (
            ("memoryId", 128),
            ("title", 160),
            ("valueSummary", 240),
            ("reasonCode", 64),
            ("reliability", 48),
        ):
            _copy_text(raw, item, key, limit, budget)
        _copy_int(raw, item, "sourceCount")
        result.append(item)
    return result


def _sanitize_memory_exclusions(value: object) -> dict[str, int]:
    raw = value if isinstance(value, dict) else {}
    return {
        "conflicted": max(0, _safe_int(raw.get("conflicted"))),
        "sensitive": max(0, _safe_int(raw.get("sensitive"))),
        "governed": max(0, _safe_int(raw.get("governed"))),
    }


def _copy_text(raw: dict[str, object], target: dict[str, object], key: str, limit: int, budget: _DebugBudget) -> None:
    if key in raw and raw.get(key) is not None:
        target[key] = budget.text(raw.get(key), limit)


def _copy_bool(raw: dict[str, object], target: dict[str, object], key: str) -> None:
    if key in raw:
        target[key] = bool(raw.get(key))


def _copy_int(raw: dict[str, object], target: dict[str, object], key: str) -> None:
    if key in raw and raw.get(key) is not None:
        target[key] = _safe_int(raw.get(key))


def _copy_float(raw: dict[str, object], target: dict[str, object], key: str) -> None:
    if key in raw and raw.get(key) is not None:
        target[key] = _safe_float(raw.get(key))


def _copy_string_list(
    raw: dict[str, object],
    target: dict[str, object],
    key: str,
    item_limit: int,
    text_limit: int,
    budget: _DebugBudget,
) -> None:
    if isinstance(raw.get(key), list):
        target[key] = [budget.text(item, text_limit) for item in budget.limited(raw[key], item_limit)]


def _safe_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
