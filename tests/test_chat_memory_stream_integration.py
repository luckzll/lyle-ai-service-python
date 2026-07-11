from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.memory.store import format_memory_context, recall_memory


def _sse_events(body: str) -> dict[str, list[dict[str, object]]]:
    events: dict[str, list[dict[str, object]]] = {}
    for block in body.split("\n\n"):
        event = ""
        data = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line.removeprefix("event: ").strip()
            elif line.startswith("data: "):
                data += line.removeprefix("data: ")
        if event and data:
            events.setdefault(event, []).append(json.loads(data))
    return events


def test_stream_memory_candidate_confirm_builds_profile_fragment_and_evidence(monkeypatch):
    user_id = "stream-memory-integration-user"
    conversation_id = "stream-memory-conversation"
    internal_token = "stream-memory-integration-token-123456"

    monkeypatch.setattr(settings, "internal_api_token", internal_token)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "hybrid")
    monkeypatch.setattr(
        "app.api.chat.prepare_multi_agent_context",
        lambda *args, **kwargs: {
            "contexts": [],
            "memory_context": "",
            "tool_context": "",
            "tool_results": [],
            "agent_trace": [],
            "memory": {},
            "rag_sufficiency": {},
        },
    )
    monkeypatch.setattr("app.api.chat.stream_answer", lambda *args, **kwargs: iter(["已记住，我不喜欢帽子。"]))
    monkeypatch.setattr(
        "app.api.chat.run_reviewer_agent",
        lambda *args, **kwargs: {"approved": True, "score": 1.0, "reason": "ok", "skipped": True},
    )
    monkeypatch.setattr("app.api.chat._project_context_for_response", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        "app.api.chat.extract_project_candidates",
        lambda *args, **kwargs: {
            "candidateGoals": [],
            "candidateDecisions": [],
            "candidateOpenQuestions": [],
        },
    )
    monkeypatch.setattr("app.api.chat.build_plan_candidate", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.api.chat.build_execution_suggestions", lambda *args, **kwargs: [])

    client = TestClient(app)
    headers = {"X-Internal-Token": internal_token}
    response = client.post(
        "/ai/chat/stream",
        headers=headers,
        json={
            "userId": user_id,
            "conversationId": conversation_id,
            "message": "请记住：我不喜欢帽子。",
            "knowledgeBaseIds": [],
            "stream": True,
            "aiConfig": {},
        },
    )

    assert response.status_code == 200
    events = _sse_events(response.text)
    delta = str(events["delta"][0]["text"])
    assert "待确认记忆" in delta
    assert "确认后才会写入长期记忆" in delta
    assert "已记住" not in delta

    candidate = events["meta"][0]["candidateMemories"][0]
    assert candidate["status"] == "pending"
    assert candidate["targetKey"] == "negative_preference"
    assert candidate["schemaPath"] == "preferences.visual.dislikes"
    assert candidate["value"] == "帽子"
    assert candidate["source"].startswith(f"source_ref:conversation:{conversation_id}")

    pending_snapshot = client.get(
        f"/ai/memory/{user_id}",
        headers=headers,
        params={"conversationId": conversation_id},
    ).json()
    pending = next(item for item in pending_snapshot["structured_all"] if item["key"] == candidate["key"])
    assert pending["status"] == "pending"
    assert pending["schemaPath"] == "preferences.visual.dislikes"

    confirmed = client.put(
        f"/ai/memory/{user_id}/structured",
        headers=headers,
        json={
            "key": candidate["schemaPath"],
            "value": candidate["value"],
            "confidence": candidate["confidence"],
            "status": "active",
            "source": candidate["source"],
        },
    )
    assert confirmed.status_code == 200
    disabled = client.put(
        f"/ai/memory/{user_id}/structured/{candidate['key']}/status",
        headers=headers,
        json={"status": "disabled"},
    )
    assert disabled.status_code == 200

    snapshot = client.get(
        f"/ai/memory/{user_id}",
        headers=headers,
        params={"conversationId": conversation_id},
    ).json()
    profile = next(item for item in snapshot["profiles"] if item["schemaPath"] == "preferences.visual.dislikes")
    fragment = next(item for item in snapshot["fragments"] if item["category"] == "preferences.visual.dislikes")
    assert profile["value"] == "帽子"
    assert profile["evidenceCount"] == 1
    assert fragment["normalizedText"] == "帽子"
    assert fragment["sourceRef"] == f"conversation:{conversation_id}"


def test_memory_claim_guard_reports_extraction_failure_truthfully():
    from app.api.chat import _guard_unconfirmed_memory_claim

    answer = _guard_unconfirmed_memory_claim("请记住这件事", "好的，已经记住了。", [])

    assert "不会写入长期记忆" in answer
    assert "已经记住" not in answer


def test_memory_claim_guard_keeps_reviewer_rewrites_pending():
    from app.api.chat import _guard_unconfirmed_memory_claim

    answer = _guard_unconfirmed_memory_claim(
        "请记住：我不喜欢帽子。",
        "好的，已经记住了。",
        [{"schemaPath": "preferences.visual.dislikes", "value": "帽子", "status": "pending"}],
    )

    assert "待确认记忆" in answer
    assert "确认后才会写入长期记忆" in answer
    assert "已经记住" not in answer


def _configure_cross_session_stream(monkeypatch, internal_token: str, answer_factory):
    monkeypatch.setattr(settings, "internal_api_token", internal_token)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "rule")
    monkeypatch.setattr(settings, "reviewer_enabled", False)
    monkeypatch.setattr(settings, "reviewer_max_retry", 0)

    def prepare_context(user_id, conversation_id, question, knowledge_base_ids):
        memory = recall_memory(user_id, conversation_id, question, include_vector=False)
        return {
            "contexts": [],
            "memory_context": format_memory_context(memory),
            "tool_context": "",
            "tool_results": [],
            "agent_trace": [{"agent": "memory_agent", "recalledRefs": memory.get("recalled_refs") or []}],
            "memory": memory,
            "rag_sufficiency": {},
        }

    monkeypatch.setattr("app.api.chat.prepare_multi_agent_context", prepare_context)
    monkeypatch.setattr("app.api.chat.stream_answer", answer_factory)
    monkeypatch.setattr("app.api.chat._project_context_for_response", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        "app.api.chat.extract_project_candidates",
        lambda *args, **kwargs: {
            "candidateGoals": [],
            "candidateDecisions": [],
            "candidateOpenQuestions": [],
        },
    )
    monkeypatch.setattr("app.api.chat.build_plan_candidate", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.api.chat.build_execution_suggestions", lambda *args, **kwargs: [])


def _post_stream(client, headers, user_id: str, conversation_id: str, message: str):
    response = client.post(
        "/ai/chat/stream",
        headers=headers,
        json={
            "userId": user_id,
            "conversationId": conversation_id,
            "message": message,
            "knowledgeBaseIds": [],
            "stream": True,
            "aiConfig": {},
        },
    )
    assert response.status_code == 200
    return _sse_events(response.text)


def test_confirmed_active_memory_is_recalled_in_new_conversation_and_reviewer_allows(monkeypatch):
    user_id = "cross-session-active-reviewer-user"
    internal_token = "cross-session-active-reviewer-token-123456"

    def answer_factory(question, contexts, memory_context):
        if "请记住" in question:
            return iter(["已记住，我不喜欢帽子。"])
        assert "视觉负向偏好" in memory_context
        assert "帽子" in memory_context
        return iter(["你不喜欢帽子。"])

    _configure_cross_session_stream(monkeypatch, internal_token, answer_factory)
    client = TestClient(app)
    headers = {"X-Internal-Token": internal_token}

    first_events = _post_stream(
        client,
        headers,
        user_id,
        "cross-session-source",
        "请记住：我不喜欢帽子。",
    )
    candidate = first_events["meta"][0]["candidateMemories"][0]
    confirmed = client.put(
        f"/ai/memory/{user_id}/structured",
        headers=headers,
        json={
            "key": candidate["schemaPath"],
            "value": candidate["value"],
            "confidence": candidate["confidence"],
            "status": "active",
            "source": candidate["source"],
        },
    )
    assert confirmed.status_code == 200
    # Reproduce the legacy frontend's two-step confirmation that caused the bug.
    disabled_candidate = client.put(
        f"/ai/memory/{user_id}/structured/{candidate['key']}/status",
        headers=headers,
        json={"status": "disabled"},
    )
    assert disabled_candidate.status_code == 200

    snapshot = client.get(f"/ai/memory/{user_id}", headers=headers).json()
    profile = next(item for item in snapshot["profiles"] if item["schemaPath"] == "preferences.visual.dislikes")
    fragment = next(item for item in snapshot["fragments"] if item["category"] == "preferences.visual.dislikes")
    assert profile["status"] == "active"
    assert fragment["status"] == "active"
    assert not any(item["value"] == "帽子" for item in snapshot["governance"]["blockedTerms"])

    recall_events = _post_stream(
        client,
        headers,
        user_id,
        "cross-session-new",
        "你记得我什么？",
    )
    answer = str(recall_events["delta"][0]["text"])
    review = recall_events["meta"][0]["review"]
    recalled_refs = review["recalledMemoryRefs"]

    assert "帽子" in answer
    assert "已确认记忆" in answer
    assert "已治理记忆" not in answer
    assert "确认后才会正式生效" not in answer
    assert review["approved"] is True
    assert any(ref["kind"] == "profile" and ref["id"] == profile["id"] for ref in recalled_refs)
    assert any(ref["kind"] == "fragment" and ref["id"] == fragment["id"] for ref in recalled_refs)


@pytest.mark.parametrize("governed_status", ["disabled", "forgotten"])
def test_governed_memory_is_not_recalled_and_reviewer_blocks(monkeypatch, governed_status):
    user_id = f"cross-session-{governed_status}-reviewer-user"
    internal_token = f"cross-session-{governed_status}-reviewer-token-123456"

    def answer_factory(question, contexts, memory_context):
        assert "视觉负向偏好" not in memory_context
        return iter(["你不喜欢帽子。"])

    _configure_cross_session_stream(monkeypatch, internal_token, answer_factory)
    client = TestClient(app)
    headers = {"X-Internal-Token": internal_token}
    created = client.put(
        f"/ai/memory/{user_id}/structured",
        headers=headers,
        json={
            "key": "preferences.visual.dislikes",
            "value": "帽子",
            "confidence": 0.95,
            "status": "active",
            "source": "source_ref:conversation:governed-source\n我不喜欢帽子。",
        },
    )
    assert created.status_code == 200

    governed = client.patch(
        f"/ai/memory/{user_id}/profiles/preferences.visual.dislikes/status",
        headers=headers,
        json={"status": governed_status},
    )
    assert governed.status_code == 200
    snapshot = client.get(f"/ai/memory/{user_id}", headers=headers).json()
    profile = next(item for item in snapshot["profiles"] if item["schemaPath"] == "preferences.visual.dislikes")
    fragment = next(item for item in snapshot["fragments"] if item["category"] == "preferences.visual.dislikes")
    structured = next(item for item in snapshot["structured_all"] if item["key"] == "preferences.visual.dislikes")
    assert profile["status"] == governed_status
    assert fragment["status"] == governed_status
    assert structured["status"] == governed_status

    recall_events = _post_stream(
        client,
        headers,
        user_id,
        f"cross-session-{governed_status}-new",
        "我不喜欢什么？",
    )
    answer = str(recall_events["delta"][0]["text"])
    review = recall_events["meta"][0]["review"]

    assert review["approved"] is False
    assert review["ruleBlocked"] is True
    assert governed_status in review["reason"]
    assert review["blockedMemoryRef"] == {"kind": "profile", "id": profile["id"]}
    assert not any(ref["schemaPath"] == "preferences.visual.dislikes" for ref in review["recalledMemoryRefs"])
    assert "你不喜欢帽子" not in answer


def test_memory_overview_stream_uses_each_objects_actual_status(monkeypatch):
    internal_token = "mixed-memory-overview-token-123456"
    model_calls: list[str] = []

    def wrong_model_answer(question, contexts, memory_context):
        model_calls.append(question)
        return iter(["这些记录目前都是待确认状态，需要你明确确认后才会正式生效。"])

    _configure_cross_session_stream(monkeypatch, internal_token, wrong_model_answer)
    memory = {
        "profiles": [{
            "id": 11,
            "schemaPath": "preferences.visual.dislikes",
            "label": "视觉负向偏好",
            "value": "帽子",
            "summary": "用户在视觉形象上不喜欢帽子。",
            "status": "active",
            "confidence": 0.95,
        }],
        "fragments": [],
        "structured": [],
        "recalled_refs": [{
            "kind": "profile",
            "id": 11,
            "schemaPath": "preferences.visual.dislikes",
            "value": "帽子",
            "status": "active",
        }],
        "governance": {"blockedTerms": [], "lowConfidenceTerms": []},
    }
    tool_results = [{
        "tool": "work_summary",
        "ok": True,
        "summary": {
            "todos": {"items": [{"id": 22, "title": "进行中的验收任务", "status": "open"}]},
            "operations": {"items": [{"id": 33, "summary": "创建验收日历", "status": "rejected"}]},
        },
    }]
    monkeypatch.setattr(
        "app.api.chat.prepare_multi_agent_context",
        lambda *args, **kwargs: {
            "contexts": [],
            "memory_context": "用户画像：视觉负向偏好：不喜欢帽子",
            "tool_context": "",
            "tool_results": tool_results,
            "agent_trace": [],
            "memory": memory,
            "rag_sufficiency": {},
        },
    )
    client = TestClient(app)
    events = _post_stream(
        client,
        {"X-Internal-Token": internal_token},
        "mixed-memory-overview-user",
        "mixed-memory-overview-conversation",
        "你记得我什么？",
    )
    answer = str(events["delta"][0]["text"])
    review = events["meta"][0]["review"]

    assert "帽子" in answer and "已确认记忆" in answer
    assert "进行中的验收任务（进行中）" in answer
    assert "创建验收日历（已拒绝，未执行）" in answer
    assert "确认后才会正式生效" not in answer
    assert review["approved"] is True
    assert {item["status"] for item in review["objectStatuses"]} == {"active", "open", "rejected"}
    assert events["meta"][0]["candidateMemories"] == []
    assert model_calls == []
