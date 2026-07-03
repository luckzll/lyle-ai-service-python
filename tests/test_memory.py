from app.core.config import settings
from app.memory import store as memory_store
from app.memory.store import (
    delete_structured_memory,
    export_memory,
    format_memory_context,
    _merge_vector_memories,
    recall_memory,
    remember_interaction,
    set_structured_memory_status,
    upsert_structured_memory,
    get_structured_memory_merge_suggestions,
)
from app.memory.evidence_store import list_profile_evidence
from app.memory.profile_schema import schema_path_for_memory


def assert_candidate(result: dict[str, object], target_key: str, value: str):
    candidates = result.get("candidateMemories") or []
    assert any(
        item.get("targetKey") == target_key
        and item.get("value") == value
        and item.get("status") == "pending"
        for item in candidates
    )


def test_memory_uses_in_memory_fallback(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "rule")

    result = remember_interaction("unit-user", "unit-conversation", "我喜欢西瓜", "记住了")
    memory = export_memory("unit-user", "unit-conversation")

    assert len(memory["short_term"]) == 2
    assert memory["structured"] == []
    assert_candidate(result, "food_preference", "西瓜")


def test_negative_preference_extracts_as_dislike_candidate(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "rule")

    result = remember_interaction("negative-preference-user", "negative-preference-conversation", "我不喜欢帽子", "我记下了")

    assert_candidate(result, "negative_preference", "帽子")
    assert result["candidateMemories"][0]["schemaPath"] == "preferences.visual.dislikes"


def test_schema_path_maps_legacy_memory_keys_to_profile_paths():
    assert schema_path_for_memory("career", "产品经理方向") == "work.current_role"
    assert schema_path_for_memory("negative_preference", "帽子") == "preferences.visual.dislikes"
    assert schema_path_for_memory("food_preference", "西瓜") == "preferences.food.likes"


def test_candidate_memory_records_source_reference(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "rule")

    result = remember_interaction("source-ref-user", "conversation-123", "我喜欢西瓜", "我记下了")

    candidate = result["candidateMemories"][0]
    assert candidate["source"].startswith("source_ref:conversation:conversation-123\n")


def test_structured_memory_confidence_adjusts_with_confirmation_and_disable(monkeypatch):
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)

    created = upsert_structured_memory("confidence-user", "preference", "西瓜", confidence=0.8, status="active")
    refreshed = upsert_structured_memory("confidence-user", "preference", "西瓜", confidence=0.8, status="active")
    disabled = set_structured_memory_status("confidence-user", "preference", "disabled")

    assert created["confidence"] == 0.8
    assert refreshed["confidence"] > created["confidence"]
    assert disabled["confidence"] < refreshed["confidence"]


def test_low_confidence_structured_memory_is_marked_cautious_in_context():
    context = format_memory_context({
        "summary": "",
        "structured": [{"key": "preference", "value": "帽子", "confidence": 0.62}],
        "retrieved": [],
        "short_term": [],
    })

    assert "低置信度，仅作参考" in context


def test_duplicate_pending_structured_memory_reuses_existing_candidate(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "rule")

    first = remember_interaction("pending-dedupe-user", "pending-dedupe-conversation", "我不喜欢帽子", "我记下了")
    second = remember_interaction("pending-dedupe-user", "pending-dedupe-conversation", "我不喜欢帽子", "我记下了")
    memory = export_memory("pending-dedupe-user", "pending-dedupe-conversation")

    first_candidate = first["candidateMemories"][0]
    second_candidate = second["candidateMemories"][0]
    pending_items = [
        item
        for item in memory["structured_all"]
        if item["status"] == "pending" and item["key"].startswith("pending:") and item["key"].endswith(":negative_preference")
    ]

    assert first_candidate["key"] == second_candidate["key"]
    assert len(pending_items) == 1


def test_merge_suggestions_include_duplicate_pending_memories(monkeypatch):
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)

    upsert_structured_memory("merge-duplicate-user", "pending:one:negative_preference", "帽子", status="pending")
    upsert_structured_memory("merge-duplicate-user", "pending:two:negative_preference", "帽子", status="pending")

    suggestions = get_structured_memory_merge_suggestions("merge-duplicate-user")

    duplicate = next(item for item in suggestions if item["group"] == "duplicate:negative_preference:帽子")
    assert "重复" in duplicate["reason"]
    assert len(duplicate["items"]) == 2


def test_merge_suggestions_group_positive_and_negative_preferences(monkeypatch):
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)

    upsert_structured_memory("merge-conflict-user", "preference", "帽子", status="active")
    upsert_structured_memory("merge-conflict-user", "negative_preference", "帽子", status="active")

    suggestions = get_structured_memory_merge_suggestions("merge-conflict-user")

    conflict = next(item for item in suggestions if item["group"] == "conflict:preference")
    assert "同类" in conflict["reason"]
    assert len(conflict["items"]) == 2


def test_confirmed_candidate_memory_becomes_recallable(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "rule")

    result = remember_interaction("confirm-user", "confirm-conversation", "我喜欢西瓜", "记住了")
    assert_candidate(result, "food_preference", "西瓜")

    memory_before = recall_memory("confirm-user", "confirm-conversation", "我喜欢什么")
    assert memory_before["structured"] == []

    upsert_structured_memory("confirm-user", "food_preference", "西瓜", source="confirmed", status="active")
    memory_after = recall_memory("confirm-user", "confirm-conversation", "我喜欢什么")
    assert any(item["key"] == "food_preference" and item["value"] == "西瓜" for item in memory_after["structured"])


def test_confirmed_structured_memory_projects_to_profile_fragment_and_evidence(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)

    upsert_structured_memory(
        "profile-user",
        "negative_preference",
        "帽子",
        source="source_ref:conversation:profile-conversation\n我不喜欢帽子",
        confidence=0.9,
        status="active",
    )

    memory = export_memory("profile-user", "profile-conversation")
    profile = next(item for item in memory["profiles"] if item["schemaPath"] == "preferences.visual.dislikes")
    fragment = next(item for item in memory["fragments"] if item["category"] == "preferences.visual.dislikes")
    evidence = list_profile_evidence("profile-user", int(profile["id"]))

    assert profile["value"] == "帽子"
    assert profile["evidenceCount"] == 1
    assert fragment["normalizedText"] == "帽子"
    assert evidence[0]["fragmentId"] == fragment["id"]


def test_recall_memory_includes_schema_aware_profile_context(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)

    upsert_structured_memory("profile-recall-user", "career", "产品经理方向", status="active")
    upsert_structured_memory("profile-recall-user", "food_preference", "西瓜", status="active")

    memory = recall_memory("profile-recall-user", None, "帮我规划一下这周产品定位文档怎么做")
    context = format_memory_context(memory)

    assert any(item["schemaPath"] == "work.current_role" for item in memory["profiles"])
    assert all(item["schemaPath"] != "preferences.food.likes" for item in memory["profiles"])
    assert "当前角色：产品经理方向" in context


def test_manual_structured_memory_upsert_and_delete(monkeypatch):
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "rule")

    upsert_structured_memory("manual-user", "role", "developer")
    memory = export_memory("manual-user", None)
    assert any(item["key"] == "role" and item["value"] == "developer" for item in memory["structured"])

    delete_structured_memory("manual-user", "role")
    memory = export_memory("manual-user", None)
    assert all(item["key"] != "role" for item in memory["structured"])
    assert any(item["key"] == "role" and item["status"] == "disabled" for item in memory["structured_all"])

    set_structured_memory_status("manual-user", "role", "active")
    memory = export_memory("manual-user", None)
    assert any(item["key"] == "role" and item["status"] == "active" for item in memory["structured"])


def test_export_memory_returns_latest_structured_when_over_recall_limit(monkeypatch):
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "memory_recall_limit", 2)

    upsert_structured_memory("overflow-user", "name", "Lyle")
    upsert_structured_memory("overflow-user", "career", "工程师")
    upsert_structured_memory("overflow-user", "preference", "西瓜")

    memory = export_memory("overflow-user", None)

    assert len(memory["structured"]) == 2
    assert [item["key"] for item in memory["structured"]] == ["preference", "career"]
    assert len(memory["structured_all"]) == 3


def test_structured_recall_falls_back_to_latest_when_question_misses(monkeypatch):
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "memory_recall_limit", 1)

    upsert_structured_memory("fallback-user", "name", "Lyle")
    upsert_structured_memory("fallback-user", "career", "工程师")

    memory = recall_memory("fallback-user", None, "今天天气怎么样")

    assert len(memory["structured"]) == 1
    assert memory["structured"][0]["key"] == "career"


def test_structured_memory_conflict_ignores_lower_confidence(monkeypatch):
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_conflict_confidence_margin", 0.15)

    created = upsert_structured_memory(
        "conflict-user",
        "career",
        "后端开发",
        confidence=0.95,
    )
    ignored = upsert_structured_memory(
        "conflict-user",
        "career",
        "产品经理",
        confidence=0.6,
    )

    memory = export_memory("conflict-user", None)
    assert created["conflict_action"] == "created"
    assert ignored["conflict_action"] == "ignored"
    assert any(item["key"] == "career" and item["value"] == "后端开发" for item in memory["structured"])


def test_structured_memory_conflict_replaces_with_high_confidence(monkeypatch):
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_conflict_confidence_margin", 0.15)

    upsert_structured_memory("replace-user", "career", "学生", confidence=0.7)
    replaced = upsert_structured_memory("replace-user", "career", "后端开发", confidence=0.9)

    memory = export_memory("replace-user", None)
    assert replaced["conflict_action"] == "replaced"
    assert any(item["key"] == "career" and item["value"] == "后端开发" for item in memory["structured"])


def test_structured_memory_conflict_reactivates_disabled_memory(monkeypatch):
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)

    upsert_structured_memory("reactivate-user", "preference", "西瓜", confidence=0.9)
    delete_structured_memory("reactivate-user", "preference")
    reactivated = upsert_structured_memory("reactivate-user", "preference", "西瓜", confidence=0.7)

    memory = export_memory("reactivate-user", None)
    assert reactivated["conflict_action"] == "reactivated"
    assert any(item["key"] == "preference" and item["status"] == "active" for item in memory["structured"])


def test_memory_compresses_only_after_trigger_limit(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "rule")
    monkeypatch.setattr(settings, "deepseek_api_key", "")
    monkeypatch.setattr(settings, "memory_short_term_limit", 10)
    monkeypatch.setattr(settings, "memory_compression_trigger_limit", 20)
    monkeypatch.setattr(settings, "memory_recall_limit", 30)

    user_id = "compression-user"
    conversation_id = "compression-conversation"

    for index in range(10):
        remember_interaction(user_id, conversation_id, f"问题{index}", f"回答{index}")

    memory = export_memory(user_id, conversation_id)
    assert len(memory["short_term"]) == 20
    assert memory["summary"] == ""

    remember_interaction(user_id, conversation_id, "问题10", "回答10")

    memory = export_memory(user_id, conversation_id)
    assert len(memory["short_term"]) == 10
    assert "问题0" in memory["summary"]


def test_memory_summary_falls_back_to_latest_user_summary(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "rule")
    monkeypatch.setattr(settings, "deepseek_api_key", "")
    monkeypatch.setattr(settings, "memory_short_term_limit", 2)
    monkeypatch.setattr(settings, "memory_compression_trigger_limit", 4)
    monkeypatch.setattr(settings, "memory_recall_limit", 10)

    user_id = "summary-fallback-user"
    source_conversation = "summary-source-conversation"
    target_conversation = "summary-target-conversation"

    for index in range(3):
        remember_interaction(user_id, source_conversation, f"摘要问题{index}", f"摘要回答{index}")

    source_memory = export_memory(user_id, source_conversation)
    target_memory = export_memory(user_id, target_conversation)

    assert "摘要问题0" in source_memory["summary"]
    assert target_memory["summary"] == source_memory["summary"]
    assert target_memory["summary_meta"]["fallback"] is True
    assert target_memory["summary_meta"]["sourceConversationId"] == source_conversation


def test_memory_compression_does_not_merge_fallback_summary(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "rule")
    monkeypatch.setattr(settings, "deepseek_api_key", "")
    monkeypatch.setattr(settings, "memory_short_term_limit", 1)
    monkeypatch.setattr(settings, "memory_compression_trigger_limit", 2)

    user_id = "summary-isolation-user"
    source_conversation = "source-summary-conversation"
    target_conversation = "target-summary-conversation"

    for index in range(3):
        remember_interaction(user_id, source_conversation, f"来源问题{index}", f"来源回答{index}")

    for index in range(3):
        remember_interaction(user_id, target_conversation, f"目标问题{index}", f"目标回答{index}")

    target_memory = export_memory(user_id, target_conversation)

    assert "目标问题0" in target_memory["summary"]
    assert "来源问题0" not in target_memory["summary"]
    assert target_memory["summary_meta"]["fallback"] is False


def test_llm_structured_memory_extractor(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "llm")
    monkeypatch.setattr(settings, "structured_memory_confidence_threshold", 0.8)
    monkeypatch.setattr(settings, "structured_memory_recent_turns", 6)

    def fake_extract(question: str, recent_turns_text: str):
        assert "最近在做AI智能体" in question
        assert "user:" in recent_turns_text
        return [
            {"key": "current_project", "value": "AI智能体", "confidence": 0.92},
            {"key": "low_confidence", "value": "不应该保存", "confidence": 0.5},
        ]

    monkeypatch.setattr("app.memory.store.invoke_structured_memory_chain", fake_extract)

    result = remember_interaction("llm-user", "llm-conversation", "最近在做AI智能体", "听起来很棒")
    memory = export_memory("llm-user", "llm-conversation")

    assert_candidate(result, "current_project", "AI智能体")
    assert all(item.get("targetKey") != "low_confidence" for item in result["candidateMemories"])
    assert memory["structured"] == []


def test_structured_memory_does_not_extract_questions(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "rule")

    remember_interaction("question-user", "question-conversation", "我的职业是什么", "你还没有告诉我")
    memory = export_memory("question-user", "question-conversation")

    assert all(item["key"] != "职业" for item in memory["structured"])


def test_structured_memory_does_not_extract_how_questions(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "hybrid")

    def fake_extract(question: str, recent_turns_text: str):
        raise AssertionError("how-question input should not call the LLM extractor")

    monkeypatch.setattr("app.memory.store.invoke_structured_memory_chain", fake_extract)

    remember_interaction("how-question-user", "how-question-conversation", "你怎么没回我", "来了")
    memory = export_memory("how-question-user", "how-question-conversation")

    assert memory["structured"] == []


def test_rule_structured_memory_extracts_career_from_self_intro(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "rule")

    result = remember_interaction("career-user", "career-conversation", "我是一个后端开发工程师", "记住了")

    assert_candidate(result, "career", "后端开发工程师")


def test_rule_structured_memory_normalises_chinese_career_key(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "rule")

    result = remember_interaction("career-key-user", "career-key-conversation", "我的职业是算法工程师", "记住了")

    assert_candidate(result, "career", "算法工程师")
    assert all(item.get("targetKey") != "职业" for item in result["candidateMemories"])


def test_rule_structured_memory_does_not_mix_preference_into_identity(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "rule")

    result = remember_interaction("mixed-identity-user", "mixed-identity-conversation", "我是周乐乐喜欢吃西瓜天天敲代码", "记住了")

    assert_candidate(result, "name", "周乐乐")
    assert all(item.get("targetKey") != "identity" for item in result["candidateMemories"])
    assert all("西瓜" not in item.get("value", "") for item in result["candidateMemories"] if item.get("targetKey") == "identity")


def test_rule_structured_memory_extracts_zodiac_separately(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "rule")

    result = remember_interaction("zodiac-user", "zodiac-conversation", "我是天秤星座", "记住了")

    assert_candidate(result, "zodiac", "天秤座")
    assert all(item.get("targetKey") != "identity" for item in result["candidateMemories"])


def test_llm_structured_memory_normalises_zodiac_identity(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "llm")
    monkeypatch.setattr(settings, "structured_memory_confidence_threshold", 0.8)

    def fake_extract(question: str, recent_turns_text: str):
        return [{"key": "identity", "value": "天秤星座", "confidence": 0.95}]

    monkeypatch.setattr("app.memory.store.invoke_structured_memory_chain", fake_extract)

    result = remember_interaction("llm-zodiac-user", "llm-zodiac-conversation", "我是天秤星座", "记住了")

    assert_candidate(result, "zodiac", "天秤座")
    assert all(item.get("targetKey") != "identity" for item in result["candidateMemories"])


def test_hybrid_structured_memory_skips_llm_for_questions(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "hybrid")

    def fake_extract(question: str, recent_turns_text: str):
        raise AssertionError("question-like input should not call the LLM extractor")

    monkeypatch.setattr("app.memory.store.invoke_structured_memory_chain", fake_extract)

    remember_interaction("hybrid-question-user", "hybrid-question-conversation", "我的职业是什么", "你还没有告诉我")
    memory = export_memory("hybrid-question-user", "hybrid-question-conversation")

    assert all(item["key"] != "职业" for item in memory["structured"])


def test_hybrid_structured_memory_merges_rule_and_llm(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "hybrid")
    monkeypatch.setattr(settings, "structured_memory_confidence_threshold", 0.8)

    def fake_extract(question: str, recent_turns_text: str):
        return [{"key": "career", "value": "后端开发", "confidence": 0.9}]

    monkeypatch.setattr("app.memory.store.invoke_structured_memory_chain", fake_extract)

    result = remember_interaction("hybrid-user", "hybrid-conversation", "我喜欢西瓜，也做后端服务", "记住了")

    assert_candidate(result, "food_preference", "西瓜")
    assert_candidate(result, "career", "后端开发")


def test_structured_memory_extracts_travel_interest_candidate(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "rule")

    result = remember_interaction("travel-user", "travel-conversation", "我想去杭州旅行", "杭州不错")

    assert_candidate(result, "travel_interest", "杭州旅行")


def test_structured_memory_extracts_goal_candidate(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "rule")

    result = remember_interaction("goal-user", "goal-conversation", "我想做一个个人AI助理", "可以拆计划")

    assert_candidate(result, "goal", "一个个人AI助理")


def test_task_commitment_does_not_extract_structured_memory_candidates(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "rule")

    result = remember_interaction("task-memory-user", "task-memory-conversation", "这周我要完成产品定位文档", "收到")

    assert result["candidateMemories"] == []


def test_structured_memory_splits_preference_categories(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "rule")

    first_result = remember_interaction("split-pref-user", "split-pref-conversation", "我喜欢香蕉", "记住了")
    second_result = remember_interaction("split-pref-user", "split-pref-conversation", "相比于小猫我更喜欢小狗", "记住了")

    assert_candidate(first_result, "food_preference", "香蕉")
    assert_candidate(second_result, "pet_preference", "小狗")


def test_structured_memory_merges_multi_value_preferences(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "rule")

    first_result = remember_interaction("multi-pref-user", "multi-pref-conversation", "我喜欢小狗", "记住了")
    second_result = remember_interaction("multi-pref-user", "multi-pref-conversation", "我喜欢猫", "记住了")

    assert_candidate(first_result, "pet_preference", "小狗")
    assert_candidate(second_result, "pet_preference", "猫")


def test_structured_memory_replaces_multi_value_when_source_says_change(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "rule")

    remember_interaction("replace-pref-user", "replace-pref-conversation", "我喜欢西瓜", "记住了")
    result = remember_interaction("replace-pref-user", "replace-pref-conversation", "我不喜欢西瓜了我喜欢香蕉", "记住了")

    assert_candidate(result, "food_preference", "香蕉")


def test_memory_writes_semantic_and_vector_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", True)
    monkeypatch.setattr(settings, "milvus_enabled", True)
    monkeypatch.setattr(settings, "structured_memory_extractor", "rule")
    monkeypatch.setattr(settings, "memory_vector_text_max_chars", 500)

    writes: list[tuple[str, str]] = []
    monkeypatch.setattr(
        memory_store,
        "_upsert_qdrant_memory",
        lambda user_id, conversation_id, text: writes.append(("qdrant", text)),
    )
    monkeypatch.setattr(
        memory_store,
        "_upsert_milvus_memory",
        lambda user_id, conversation_id, text: writes.append(("milvus", text)),
    )

    result = remember_interaction("vector-user", "vector-conversation", "我喜欢西瓜", "这是一段不应该进入向量记忆的助手回复")

    assert writes == [
        ("qdrant", "用户：我喜欢西瓜"),
        ("milvus", "用户：我喜欢西瓜"),
    ]
    assert_candidate(result, "food_preference", "西瓜")
    assert result["vectorMemoryWrite"]["shouldWrite"] is True
    assert result["vectorMemoryWrite"]["reason"] == "value_signal"
    assert result["vectorMemoryWrite"]["writtenBackends"] == ["qdrant", "milvus"]


def test_memory_skips_noisy_vector_writes(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", True)
    monkeypatch.setattr(settings, "milvus_enabled", True)
    monkeypatch.setattr(settings, "structured_memory_extractor", "rule")
    monkeypatch.setattr(settings, "memory_vector_write_policy_enabled", True)

    writes: list[tuple[str, str]] = []
    monkeypatch.setattr(
        memory_store,
        "_upsert_qdrant_memory",
        lambda user_id, conversation_id, text: writes.append(("qdrant", text)),
    )
    monkeypatch.setattr(
        memory_store,
        "_upsert_milvus_memory",
        lambda user_id, conversation_id, text: writes.append(("milvus", text)),
    )

    first_result = remember_interaction("vector-noise-user", "vector-noise-conversation", "你好", "你好呀")
    second_result = remember_interaction("vector-noise-user", "vector-noise-conversation", "继续", "好的")

    assert writes == []
    assert first_result["vectorMemoryWrite"]["shouldWrite"] is False
    assert first_result["vectorMemoryWrite"]["reason"] == "noise_phrase"
    assert second_result["vectorMemoryWrite"]["shouldWrite"] is False
    assert second_result["vectorMemoryWrite"]["reason"] == "noise_phrase"


def test_memory_keeps_valuable_vector_writes(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", True)
    monkeypatch.setattr(settings, "milvus_enabled", True)
    monkeypatch.setattr(settings, "structured_memory_extractor", "rule")
    monkeypatch.setattr(settings, "memory_vector_write_policy_enabled", True)

    writes: list[tuple[str, str]] = []
    monkeypatch.setattr(
        memory_store,
        "_upsert_qdrant_memory",
        lambda user_id, conversation_id, text: writes.append(("qdrant", text)),
    )
    monkeypatch.setattr(
        memory_store,
        "_upsert_milvus_memory",
        lambda user_id, conversation_id, text: writes.append(("milvus", text)),
    )

    result = remember_interaction("vector-valuable-user", "vector-valuable-conversation", "如何优化向量记忆写入策略", "可以加策略层")

    assert writes == [
        ("qdrant", "用户：如何优化向量记忆写入策略"),
        ("milvus", "用户：如何优化向量记忆写入策略"),
    ]
    assert result["vectorMemoryWrite"]["shouldWrite"] is True
    assert result["vectorMemoryWrite"]["reason"] == "value_signal"


def test_memory_dedupes_high_similarity_vector_writes(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", True)
    monkeypatch.setattr(settings, "milvus_enabled", True)
    monkeypatch.setattr(settings, "structured_memory_extractor", "rule")
    monkeypatch.setattr(settings, "memory_vector_write_policy_enabled", True)
    monkeypatch.setattr(settings, "memory_vector_dedupe_enabled", True)
    monkeypatch.setattr(settings, "memory_vector_dedupe_score_threshold", 0.92)
    memory_store._vector_backend_disabled_until.clear()

    writes: list[tuple[str, str]] = []
    monkeypatch.setattr(
        memory_store,
        "_search_qdrant_memory",
        lambda user_id, question, top_k: [{"text": "用户：我去吃饭了", "score": 0.96, "backend": "qdrant"}],
    )
    monkeypatch.setattr(
        memory_store,
        "_upsert_qdrant_memory",
        lambda user_id, conversation_id, text: writes.append(("qdrant", text)),
    )
    monkeypatch.setattr(
        memory_store,
        "_upsert_milvus_memory",
        lambda user_id, conversation_id, text: writes.append(("milvus", text)),
    )

    result = remember_interaction("vector-dedupe-user", "vector-dedupe-conversation", "我去干饭了", "吃好")

    assert writes == []
    assert result["vectorMemoryWrite"]["shouldWrite"] is False
    assert result["vectorMemoryWrite"]["reason"] == "duplicate_vector_memory"
    assert result["vectorMemoryWrite"]["duplicateSkipped"] is True
    assert result["vectorMemoryWrite"]["duplicateScore"] == 0.96


def test_memory_keeps_low_similarity_vector_writes(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", True)
    monkeypatch.setattr(settings, "milvus_enabled", True)
    monkeypatch.setattr(settings, "structured_memory_extractor", "rule")
    monkeypatch.setattr(settings, "memory_vector_write_policy_enabled", True)
    monkeypatch.setattr(settings, "memory_vector_dedupe_enabled", True)
    monkeypatch.setattr(settings, "memory_vector_dedupe_score_threshold", 0.92)
    memory_store._vector_backend_disabled_until.clear()

    writes: list[tuple[str, str]] = []
    monkeypatch.setattr(
        memory_store,
        "_search_qdrant_memory",
        lambda user_id, question, top_k: [{"text": "用户：喜欢香蕉", "score": 0.61, "backend": "qdrant"}],
    )
    monkeypatch.setattr(
        memory_store,
        "_upsert_qdrant_memory",
        lambda user_id, conversation_id, text: writes.append(("qdrant", text)),
    )
    monkeypatch.setattr(
        memory_store,
        "_upsert_milvus_memory",
        lambda user_id, conversation_id, text: writes.append(("milvus", text)),
    )

    result = remember_interaction("vector-dedupe-user", "vector-dedupe-conversation", "如何优化工具调用策略", "可以加置信度")

    assert writes == [
        ("qdrant", "用户：如何优化工具调用策略"),
        ("milvus", "用户：如何优化工具调用策略"),
    ]
    assert result["vectorMemoryWrite"]["shouldWrite"] is True
    assert result["vectorMemoryWrite"]["duplicateSkipped"] is False


def test_memory_recall_collects_semantic_and_vector(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", True)
    monkeypatch.setattr(settings, "milvus_enabled", True)
    monkeypatch.setattr(settings, "rerank_enabled", False)
    monkeypatch.setattr(settings, "memory_recall_limit", 3)
    monkeypatch.setattr(
        memory_store,
        "_search_qdrant_memory",
        lambda user_id, question, top_k: [{"text": "用户喜欢西瓜", "score": 0.8, "backend": "qdrant"}],
    )
    monkeypatch.setattr(
        memory_store,
        "_search_milvus_memory",
        lambda user_id, question, top_k: [{"text": "用户使用Python", "score": 0.7, "backend": "milvus"}],
    )

    memory = recall_memory("recall-user", None, "我喜欢什么")

    assert [item["backend"] for item in memory["semantic"]] == ["qdrant"]
    assert [item["backend"] for item in memory["vector"]] == ["milvus"]
    assert [item["text"] for item in memory["retrieved"]] == ["用户喜欢西瓜", "用户使用Python"]


def test_merge_vector_memories_dedupes_and_sorts(monkeypatch):
    monkeypatch.setattr(settings, "rerank_enabled", False)
    monkeypatch.setattr(settings, "memory_vector_score_threshold", 0.5)
    merged = _merge_vector_memories(
        "用户喜欢什么",
        [
            {"text": "用户喜欢西瓜", "score": 0.72, "backend": "qdrant"},
            {"text": "用户正在做AI项目", "score": 0.88, "backend": "qdrant"},
        ],
        [
            {"text": "用户喜欢西瓜", "score": 0.91, "backend": "milvus"},
            {"text": "用户使用Java", "score": 0.65, "backend": "milvus"},
        ],
        top_k=2,
    )

    assert [item["text"] for item in merged] == ["用户喜欢西瓜", "用户正在做AI项目"]
    assert merged[0]["score"] == 0.91
    assert merged[0]["backend"] == "qdrant,milvus"


def test_merge_vector_memories_filters_low_scores(monkeypatch):
    monkeypatch.setattr(settings, "rerank_enabled", False)
    monkeypatch.setattr(settings, "memory_vector_score_threshold", 0.5)

    merged = _merge_vector_memories(
        "用户喜欢什么",
        [
            {"text": "低相关闲聊", "score": 0.37, "backend": "qdrant"},
            {"text": "用户喜欢香蕉", "score": 0.86, "backend": "qdrant"},
        ],
        [],
        top_k=3,
    )

    assert [item["text"] for item in merged] == ["用户喜欢香蕉"]


def test_vector_backend_failure_cooldown_skips_repeated_qdrant_attempts(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", True)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "vector_backend_failure_cooldown_seconds", 60)
    monkeypatch.setattr(memory_store, "_qdrant_memory_ready", False)
    memory_store._vector_backend_disabled_until.clear()

    calls = {"count": 0}

    def fail_ensure():
        calls["count"] += 1
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(memory_store, "_ensure_qdrant_memory_collection", fail_ensure)

    recall_memory("cooldown-user", None, "我喜欢什么")
    recall_memory("cooldown-user", None, "我喜欢什么")

    assert calls["count"] == 1


def test_milvus_search_loads_collection_before_query(monkeypatch):
    monkeypatch.setattr(settings, "milvus_enabled", True)
    monkeypatch.setattr(settings, "milvus_collection", "unit_milvus_memory")
    monkeypatch.setattr(memory_store, "_ensure_milvus_collection", lambda: None)
    monkeypatch.setattr(memory_store, "_vector_backend_available", lambda name: True)
    monkeypatch.setattr(memory_store, "embed_text", lambda text: [0.1, 0.2, 0.3])

    calls: list[tuple[str, str]] = []

    class FakeMilvusClient:
        def load_collection(self, collection_name: str) -> None:
            calls.append(("load", collection_name))

        def search(self, collection_name: str, **kwargs):
            calls.append(("search", collection_name))
            return [[]]

    monkeypatch.setattr(memory_store, "_milvus_client", lambda: FakeMilvusClient())

    assert memory_store._search_milvus_memory("milvus-user", "我喜欢什么", 3) == []
    assert calls == [("load", "unit_milvus_memory"), ("search", "unit_milvus_memory")]


def test_merge_vector_memories_can_rerank(monkeypatch):
    monkeypatch.setattr(settings, "rerank_enabled", True)
    monkeypatch.setattr(settings, "rerank_provider", "lexical")
    monkeypatch.setattr(settings, "memory_vector_score_threshold", 0.5)

    merged = _merge_vector_memories(
        "用户使用什么语言",
        [
            {"text": "用户喜欢西瓜", "score": 0.95, "backend": "qdrant"},
            {"text": "用户使用Java开发后端", "score": 0.6, "backend": "qdrant"},
        ],
        [],
        top_k=1,
    )

    assert merged[0]["text"] == "用户使用Java开发后端"
    assert merged[0]["rerank_backend"] == "lexical"
