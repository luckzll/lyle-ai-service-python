from __future__ import annotations

import asyncio
import json

from fastapi import BackgroundTasks

from app.api.chat import chat, chat_stream
from app.schemas.chat import ChatRequest
from app.security.chat_debug import DEBUG_VERSION, build_safe_chat_debug
from app.security.secret_masker import mask_secrets


def _collect_sse(response) -> dict[str, list[dict[str, object]]]:
    async def read() -> str:
        parts: list[str] = []
        async for chunk in response.body_iterator:
            parts.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return "".join(parts)

    events: dict[str, list[dict[str, object]]] = {}
    for block in asyncio.run(read()).split("\n\n"):
        name = ""
        data = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line.removeprefix("event: ").strip()
            elif line.startswith("data: "):
                data += line.removeprefix("data: ")
        if name and data:
            events.setdefault(name, []).append(json.loads(data))
    return events


def _assert_forbidden_keys_absent(value: object) -> None:
    forbidden = {
        "memoryContext",
        "toolContext",
        "combinedContext",
        "ragContexts",
        "inputs",
        "tool_inputs",
        "result",
        "reasoning",
        "chainOfThought",
    }
    if isinstance(value, dict):
        assert forbidden.isdisjoint(value)
        for item in value.values():
            _assert_forbidden_keys_absent(item)
    elif isinstance(value, list):
        for item in value:
            _assert_forbidden_keys_absent(item)


def _diagnostic_state() -> dict[str, object]:
    return {
        "answer": "安全回答",
        "contexts": [{
            "doc_id": "doc-1",
            "title": "架构说明 token=top-secret-token",
            "chunk": "Q 是查询向量；api_key=should-not-leak。",
            "score": "0.92",
            "backend": "qdrant",
            "hitModes": "vector",
            "chunkIndex": "3",
            "parentTitle": "Self-Attention",
            "startChar": "80",
            "endChar": "180",
        }],
        "agent_trace": [
            {"agent": "supervisor_agent", "intent": "knowledge", "nextAgents": ["rag_agent", "writer_agent"]},
            {
                "agent": "memory_agent",
                "memoryContextChars": 5000,
                "memorySummary": {
                    "summaryChars": 120,
                    "summaryPreview": "用户偏好 token=memory-secret",
                    "shortTermCount": 2,
                    "shortTermPreview": [{"role": "user", "content": "password=hunter2"}],
                    "structuredCount": 1,
                    "structuredPreview": ["preferences.answer=简洁"],
                    "semanticCount": 1,
                    "vectorCount": 1,
                    "retrievedCount": 1,
                    "retrievedPreview": ["Bearer secret-bearer-value"],
                },
            },
            {
                "agent": "rag_agent",
                "contextCount": 1,
                "queryCount": 2,
                "sufficient": True,
                "skipped": False,
                "decision": {"should_retrieve": True, "reason": "已选择知识库"},
            },
            {
                "agent": "writer_agent",
                "answerChars": 4,
                "writerInput": {
                    "memoryContextChars": 5000,
                    "excludedMemoryContextChars": 0,
                    "memoryExclusionReason": "",
                    "toolContextChars": 3000,
                    "ragContextCount": 1,
                    "combinedContextChars": 8000,
                    "memoryContext": "private memory",
                    "toolContext": "private tool output",
                    "combinedContext": "all private context",
                    "ragContexts": [{"chunkPreview": "private chunk"}],
                },
                "reasoning": "hidden chain of thought",
            },
        ],
        "rag_queries": ["Transformer QKV", "自注意力查询键值"],
        "rag_sufficiency": {"sufficient": True, "score": 0.9, "reason": "资料充分"},
        "tool_results": [{
            "tool": "todo",
            "ok": True,
            "content": "创建任务 password=secret-password",
            "payload": {"title": "raw payload"},
            "inputs": {"token": "raw-token"},
            "result": {"private": "raw result"},
            "requiresConfirmation": True,
            "operation": {
                "id": 42,
                "toolName": "todo",
                "operationType": "todo.create",
                "riskLevel": "write",
                "summary": "创建安全验收任务",
                "status": "pending",
                "payload": {"title": "验收任务", "dueAt": "2026-09-05T10:00:00Z", "secret": "must-not-cross-boundary"},
            },
        }],
        "review": {"approved": True, "score": 0.95, "reason": "grounded", "rewrite_instruction": "private prompt"},
        "memory": {
            "profile_context": {
                "usedMemories": [{
                    "memoryId": "profile:1",
                    "title": "回答偏好",
                    "valueSummary": "简洁 token=memory-token",
                    "reasonCode": "QUESTION_RELEVANT",
                    "reliability": "high",
                    "sourceCount": 2,
                    "raw": "not allowed",
                }],
                "memoryExclusions": {"conflicted": 1, "sensitive": 2, "governed": 3, "raw": 99},
            },
        },
    }


def test_debug_builder_is_bounded_and_allowlist_only() -> None:
    state = _diagnostic_state()
    payload = build_safe_chat_debug(
        knowledge_base_ids=[f"kb-{index}" for index in range(30)],
        contexts=[*state["contexts"], *state["contexts"], *state["contexts"], *state["contexts"], *state["contexts"], *state["contexts"]],
        agent_trace=state["agent_trace"],
        tool_results=state["tool_results"],
        review=state["review"],
        usage={"promptTokens": 12, "completionTokens": 8, "callCount": 1, "estimatedCostMicros": 4, "measurementSource": "provider"},
        used_memories=state["memory"]["profile_context"]["usedMemories"],
        memory_exclusions=state["memory"]["profile_context"]["memoryExclusions"],
        rag_queries=state["rag_queries"],
        rag_sufficiency=state["rag_sufficiency"],
    )

    assert payload["debugVersion"] == DEBUG_VERSION
    assert len(payload["knowledgeBaseIds"]) == 20
    assert len(payload["sources"]) == 5
    assert payload["debugTruncated"] is True
    assert payload["retrievalScope"] == {
        "requested": True,
        "executed": True,
        "skipped": False,
        "knowledgeBaseCount": 20,
        "queryCount": 2,
        "contextCount": 6,
        "reason": "已选择知识库",
    }
    assert payload["toolResults"][0]["operation"] == {
        "toolName": "todo",
        "operationType": "todo.create",
        "riskLevel": "write",
        "status": "pending",
        "summary": "创建安全验收任务",
        "payload": {"title": "验收任务", "dueAt": "2026-09-05T10:00:00Z"},
        "id": 42,
    }
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "top-secret-token" not in serialized
    assert "should-not-leak" not in serialized
    assert "secret-password" not in serialized
    assert "private context" not in serialized
    _assert_forbidden_keys_absent(payload)


def test_retrieval_scope_distinguishes_not_selected_from_supervisor_skip() -> None:
    common = {
        "contexts": [],
        "agent_trace": [],
        "tool_results": [],
        "review": {},
        "usage": {},
        "used_memories": [],
        "memory_exclusions": {},
    }
    not_selected = build_safe_chat_debug(knowledge_base_ids=[], **common)
    supervisor_skipped = build_safe_chat_debug(knowledge_base_ids=["kb-1"], **common)

    assert not_selected["retrievalScope"] == {
        "requested": False,
        "executed": False,
        "skipped": True,
        "knowledgeBaseCount": 0,
        "queryCount": 0,
        "contextCount": 0,
        "reason": "未选择知识库。",
    }
    assert supervisor_skipped["retrievalScope"] == {
        "requested": True,
        "executed": False,
        "skipped": True,
        "knowledgeBaseCount": 1,
        "queryCount": 0,
        "contextCount": 0,
        "reason": "知识库检索未执行。",
    }


def test_tool_body_is_not_used_as_debug_summary() -> None:
    payload = build_safe_chat_debug(
        knowledge_base_ids=[],
        contexts=[],
        agent_trace=[],
        tool_results=[{
            "tool": "third_party_lookup",
            "ok": True,
            "content": "ordinary private response body that must not be persisted",
            "error": "ordinary private error that must not be persisted",
        }],
        review={},
        usage={},
        used_memories=[],
        memory_exclusions={},
    )

    tool = payload["toolResults"][0]
    assert tool == {"tool": "third_party_lookup", "ok": True}
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "ordinary private response body" not in serialized
    assert "ordinary private error" not in serialized


def test_structured_tool_summary_and_extra_preview_fields_are_rejected() -> None:
    payload = build_safe_chat_debug(
        knowledge_base_ids=[],
        contexts=[],
        agent_trace=[],
        tool_results=[{
            "tool": "work_summary",
            "ok": True,
            "summary": {"customer": "ordinary private tool data", "api_key": "nested-secret"},
            "title": "top-level private title",
            "dueAt": "2026-09-06T10:00:00Z",
            "startAt": "2026-09-06T11:00:00Z",
            "endAt": "2026-09-06T12:00:00Z",
            "time": "private time",
            "operation": {
                "id": 7,
                "toolName": "todo",
                "operationType": "todo.create",
                "riskLevel": "write",
                "summary": "创建安全待办",
                "status": "pending",
                "title": "duplicate private title",
                "dueAt": "duplicate private due date",
                "time": "duplicate private time",
                "payload": {
                    "title": "允许的待办标题",
                    "dueAt": "2026-09-06T10:00:00Z",
                    "private": "must-not-survive",
                },
            },
        }],
        review={},
        usage={},
        used_memories=[],
        memory_exclusions={},
    )

    tool = payload["toolResults"][0]
    assert "summary" not in tool
    assert "title" not in tool
    assert "dueAt" not in tool
    assert "startAt" not in tool
    assert "endAt" not in tool
    assert "time" not in tool
    assert tool["operation"] == {
        "id": 7,
        "toolName": "todo",
        "operationType": "todo.create",
        "riskLevel": "write",
        "summary": "创建安全待办",
        "status": "pending",
        "payload": {"title": "允许的待办标题", "dueAt": "2026-09-06T10:00:00Z"},
    }
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "ordinary private tool data" not in serialized
    assert "nested-secret" not in serialized
    assert "duplicate private" not in serialized
    assert "must-not-survive" not in serialized


def test_secret_masker_covers_private_keys_jwt_and_aws_credentials() -> None:
    private_key = "-----BEGIN PRIVATE KEY-----\nprivate-key-material\n-----END PRIVATE KEY-----"
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.signature123456"
    aws_access_key = "AKIAABCDEFGHIJKLMNOP"
    aws_secret = "A" * 40
    raw = (
        f"{private_key}\n{jwt}\n{aws_access_key}\n"
        f'{{"aws_secret_access_key": "{aws_secret}"}}'
    )

    masked = mask_secrets(raw)

    assert "private-key-material" not in masked
    assert jwt not in masked
    assert aws_access_key not in masked
    assert aws_secret not in masked


def test_nested_values_are_never_stringified_into_debug_text() -> None:
    payload = build_safe_chat_debug(
        knowledge_base_ids=[],
        contexts=[{
            "doc_id": "doc-1",
            "title": {"private": "nested title data"},
            "chunk": ["nested chunk data"],
        }],
        agent_trace=[],
        tool_results=[],
        review={},
        usage={},
        used_memories=[],
        memory_exclusions={},
    )

    serialized = json.dumps(payload, ensure_ascii=False)
    assert payload["debugTruncated"] is True
    assert "nested title data" not in serialized
    assert "nested chunk data" not in serialized


def test_sync_chat_response_uses_safe_debug_boundary(monkeypatch) -> None:
    state = _diagnostic_state()
    monkeypatch.setattr("app.api.chat.run_multi_agent_graph", lambda *args, **kwargs: state)
    monkeypatch.setattr("app.api.chat._prepare_candidate_memories_for_response", lambda *args, **kwargs: [])
    monkeypatch.setattr("app.api.chat._project_context_for_response", lambda *args, **kwargs: {})
    monkeypatch.setattr("app.api.chat.extract_project_candidates", lambda *args, **kwargs: {
        "candidateGoals": [], "candidateDecisions": [], "candidateOpenQuestions": [],
    })
    monkeypatch.setattr("app.api.chat.build_plan_candidate", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.api.chat.build_execution_suggestions", lambda *args, **kwargs: [])
    monkeypatch.setattr("app.api.chat._candidate_tasks_for_response", lambda *args, **kwargs: [])

    response = chat(ChatRequest(
        userId="user-1",
        conversationId="conversation-1",
        message="解释 QKV",
        knowledgeBaseIds=["kb-1"],
        knowledgeScope={"userId": "user-1", "authorizationRevision": "auth", "knowledgeBaseRevisions": {"kb-1": "v1"},
                        "allowedDocumentIds": {"kb-1": ["doc-1"]}, "documentRevisions": {"kb-1": {"doc-1": "legacy"}}},
    ), BackgroundTasks())
    payload = response.model_dump(by_alias=True)

    assert payload["debugVersion"] == DEBUG_VERSION
    assert payload["knowledgeBaseIds"] == ["kb-1"]
    assert payload["retrievalScope"]["executed"] is True
    assert payload["sources"][0]["docId"] == "doc-1"
    writer_trace = next(item for item in payload["agentTrace"] if item["agent"] == "writer_agent")
    assert writer_trace["writerInput"] == {
        "memoryContextChars": 5000,
        "excludedMemoryContextChars": 0,
        "toolContextChars": 3000,
        "ragContextCount": 1,
        "combinedContextChars": 8000,
        "memoryExclusionReason": "",
    }
    assert "payload" not in payload["toolResults"][0]
    assert payload["toolResults"][0]["operation"]["id"] == 42
    _assert_forbidden_keys_absent({
        key: payload[key]
        for key in ("agentTrace", "sources", "toolResults", "review", "usedMemories", "memoryExclusions")
    })


def test_stream_meta_uses_same_safe_debug_boundary(monkeypatch) -> None:
    state = _diagnostic_state()
    monkeypatch.setattr("app.api.chat.prepare_multi_agent_context", lambda *args, **kwargs: state)
    monkeypatch.setattr("app.api.chat.stream_answer", lambda *args, **kwargs: iter(["安全回答"]))
    monkeypatch.setattr("app.api.chat.run_reviewer_agent", lambda *args, **kwargs: state["review"])
    monkeypatch.setattr("app.api.chat._prepare_candidate_memories_for_response", lambda *args, **kwargs: [])
    monkeypatch.setattr("app.api.chat._project_context_for_response", lambda *args, **kwargs: {})
    monkeypatch.setattr("app.api.chat.extract_project_candidates", lambda *args, **kwargs: {
        "candidateGoals": [], "candidateDecisions": [], "candidateOpenQuestions": [],
    })
    monkeypatch.setattr("app.api.chat.build_plan_candidate", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.api.chat.build_execution_suggestions", lambda *args, **kwargs: [])
    monkeypatch.setattr("app.api.chat._candidate_tasks_for_response", lambda *args, **kwargs: [])

    events = _collect_sse(chat_stream(ChatRequest(
        userId="user-1",
        conversationId="conversation-1",
        message="解释 QKV",
        knowledgeBaseIds=["kb-1"],
        knowledgeScope={"userId": "user-1", "authorizationRevision": "auth", "knowledgeBaseRevisions": {"kb-1": "v1"},
                        "allowedDocumentIds": {"kb-1": ["doc-1"]}, "documentRevisions": {"kb-1": {"doc-1": "legacy"}}},
        stream=True,
    )))
    meta = events["meta"][0]

    assert meta["debugVersion"] == DEBUG_VERSION
    assert meta["knowledgeBaseIds"] == ["kb-1"]
    assert meta["retrievalScope"]["executed"] is True
    assert meta["retrievalScope"]["queryCount"] == 2
    assert meta["agentTrace"][2]["queries"] == state["rag_queries"]
    assert meta["agentTrace"][2]["sufficiency"]["sufficient"] is True
    assert meta["agentTrace"][3]["writerInput"]["ragContextCount"] == 1
    assert "payload" not in meta["toolResults"][0]
    _assert_forbidden_keys_absent({
        key: meta[key]
        for key in ("agentTrace", "sources", "toolResults", "review", "usedMemories", "memoryExclusions")
    })
