import pytest

import app.memory.governance as governance_module

from app.core.config import settings
from app.memory.fragments import create_memory_fragment, list_memory_fragments
from app.memory.governance import (
    MemoryGovernanceConflict,
    create_memory_conflict,
    govern_memory_record,
    list_governance_events,
    list_memory_inbox,
    list_memory_records,
    list_memory_sources,
    process_pending_purge_jobs,
    request_memory_purge,
    resolve_memory_conflict,
)
from app.memory.profile_store import get_user_profile
from app.memory.store import upsert_structured_memory
from app.memory.store import extract_structured_memory


def configure_memory_fallback(monkeypatch):
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "memory_conflict_shadow_enabled", False)
    monkeypatch.setattr(settings, "memory_source_validation_enabled", False)


def test_record_source_correction_and_idempotency(monkeypatch):
    configure_memory_fallback(monkeypatch)
    user_id = "v66-correction-user"
    upsert_structured_memory(
        user_id,
        "negative_preference",
        "人物戴帽子",
        source="source_ref:conversation:v66-conversation\n生成图片时不要给人物戴帽子。",
        confidence=0.9,
        status="active",
    )

    record = list_memory_records(user_id)["items"][0]
    assert record["title"] == "视觉负向偏好"
    assert record["sourceCount"] == 1
    assert record["reliability"] == "RELIABLE"

    first = govern_memory_record(
        user_id,
        record["memoryId"],
        action="CORRECT",
        value="不喜欢人物戴帽子",
        client_request_id="v66-correct-request-001",
    )
    repeated = govern_memory_record(
        user_id,
        record["memoryId"],
        action="CORRECT",
        value="这次重复请求不应再次写入",
        client_request_id="v66-correct-request-001",
    )

    assert first["memory"]["value"] == "不喜欢人物戴帽子"
    assert repeated["memory"]["value"] == "不喜欢人物戴帽子"
    events = list_governance_events(user_id)["items"]
    assert [item["action"] for item in events].count("CORRECT") == 1


def test_conflict_requires_version_and_resolves_to_candidate(monkeypatch):
    configure_memory_fallback(monkeypatch)
    user_id = "v66-conflict-user"
    upsert_structured_memory(user_id, "career", "产品经理", confidence=0.9, status="active")
    fragment = create_memory_fragment(
        user_id,
        source_text="我现在转岗做设计师了",
        normalized_text="设计师",
        category="work.current_role",
        conversation_id="conflict-source",
        confidence=0.9,
    )
    conflict = create_memory_conflict(user_id, "work.current_role", int(fragment["id"]))

    assert list_memory_inbox(user_id)["items"][0]["type"] == "CONFLICT"
    with pytest.raises(MemoryGovernanceConflict):
        resolve_memory_conflict(
            user_id,
            int(conflict["id"]),
            resolution="USE_CANDIDATE",
            merged_value=None,
            expected_version=99,
            client_request_id="v66-conflict-wrong-version",
        )

    result = resolve_memory_conflict(
        user_id,
        int(conflict["id"]),
        resolution="USE_CANDIDATE",
        merged_value=None,
        expected_version=int(conflict["version"]),
        client_request_id="v66-conflict-resolution-001",
    )
    assert result["memory"]["value"] == "设计师"
    assert get_user_profile(user_id, "work.current_role")["status"] == "active"
    assert list_memory_inbox(user_id)["items"] == []


def test_same_semantic_conflict_from_different_fragments_is_deduplicated(monkeypatch):
    configure_memory_fallback(monkeypatch)
    user_id = "v66-conflict-dedupe-user"
    upsert_structured_memory(user_id, "career", "产品经理", confidence=0.9, status="active")
    first_fragment = create_memory_fragment(
        user_id,
        source_text="我现在转岗做产品设计师了",
        normalized_text="产品设计师",
        category="work.current_role",
        conversation_id="conversation-a",
        source_message_id="message-a",
        confidence=0.9,
    )
    second_fragment = create_memory_fragment(
        user_id,
        source_text="  我现在转岗做产品设计师了  ",
        normalized_text="产品设计师",
        category="work.current_role",
        conversation_id="conversation-b",
        source_message_id="message-b",
        confidence=0.9,
    )

    first = create_memory_conflict(user_id, "work.current_role", int(first_fragment["id"]))
    second = create_memory_conflict(user_id, "work.current_role", int(second_fragment["id"]))
    open_items = [
        item for item in list_memory_inbox(user_id)["items"] if item["type"] == "CONFLICT"
    ]

    assert second["id"] == first["id"]
    assert second["conflictFingerprint"] == first["conflictFingerprint"]
    assert len(open_items) == 1


def test_async_purge_removes_profile_sources_and_structured_value(monkeypatch):
    configure_memory_fallback(monkeypatch)
    user_id = "v66-purge-user"
    upsert_structured_memory(user_id, "food_preference", "西瓜", confidence=0.9, status="active")
    record = list_memory_records(user_id)["items"][0]

    job = request_memory_purge(
        user_id,
        record["memoryId"],
        client_request_id="v66-purge-request-001",
    )
    outcome = process_pending_purge_jobs(limit=5)

    assert job["status"] == "PENDING"
    assert outcome == {"claimed": 1, "succeeded": 1, "failed": 0}
    assert list_memory_records(user_id)["items"] == []
    assert list_memory_fragments(user_id, category="preferences.food.likes") == []


def test_new_memory_preserves_message_level_source_anchor(monkeypatch):
    configure_memory_fallback(monkeypatch)
    user_id = "v66-source-anchor-user"
    candidate = extract_structured_memory(
        user_id,
        "conversation-66",
        "我不喜欢帽子",
        extraction_mode="rule",
        source_message_id="message-8801",
    )[0]
    upsert_structured_memory(
        user_id,
        "negative_preference",
        str(candidate["value"]),
        source=str(candidate["source"]),
        confidence=float(candidate["confidence"]),
        status="active",
    )
    record = list_memory_records(user_id)["items"][0]
    source = list_memory_sources(user_id, record["memoryId"])["items"][0]

    assert source["conversationId"] == "conversation-66"
    assert source["messageId"] == "message-8801"
    assert source["link"] == "/?conversationId=conversation-66&messageId=message-8801"


def test_unavailable_source_is_revalidated_and_cached_excerpt_is_hidden(monkeypatch):
    configure_memory_fallback(monkeypatch)
    monkeypatch.setattr(settings, "memory_source_validation_enabled", True)
    monkeypatch.setattr(settings, "memory_source_validation_fail_closed", True)
    monkeypatch.setattr(settings, "internal_api_token", "test-internal-token")
    user_id = "v66-unavailable-source-user"
    candidate = extract_structured_memory(
        user_id,
        "deleted-conversation",
        "我不喜欢帽子",
        extraction_mode="rule",
        source_message_id="deleted-message",
    )[0]
    upsert_structured_memory(
        user_id,
        "negative_preference",
        str(candidate["value"]),
        source=str(candidate["source"]),
        confidence=float(candidate["confidence"]),
        status="active",
    )

    class FakeResponse:
        def __init__(self, source_id):
            self.source_id = source_id

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "items": [{
                    "id": self.source_id,
                    "available": False,
                    "reason": "原消息已删除或当前用户无权访问",
                }]
            }

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def post(self, *_args, **kwargs):
            return FakeResponse(kwargs["json"]["sources"][0]["id"])

    monkeypatch.setattr(governance_module.httpx, "Client", FakeClient)
    record = list_memory_records(user_id)["items"][0]
    source = list_memory_sources(user_id, record["memoryId"])["items"][0]

    assert source["availability"] == "UNAVAILABLE"
    assert source["excerpt"] == ""
    assert source["link"] is None
    assert source["unavailableReason"] == "原消息已删除或当前用户无权访问"
