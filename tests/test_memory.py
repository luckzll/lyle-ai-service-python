from app.core.config import settings
from app.memory import store as memory_store
from app.memory import profile_store
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
from app.memory.profile_store import get_user_profile


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


def test_operational_plan_message_does_not_create_structured_memory(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "hybrid")

    result = remember_interaction(
        "operation-noise-user",
        "operation-noise-conversation",
        "基于 V3测试项目 的目标「完成测试报告」拆一个三步推进计划。",
        "已生成待确认计划，请在下方计划卡片中点击创建计划。",
    )

    assert result["candidateMemories"] == []


def test_todo_operation_message_does_not_create_structured_memory(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "hybrid")

    result = remember_interaction(
        "todo-noise-user",
        "todo-noise-conversation",
        "创建待办：明天检查V3测试报告。",
        "已生成待确认操作，请在下方操作卡片中点击确认执行。",
    )

    assert result["candidateMemories"] == []


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


def test_uncertain_preference_creates_low_confidence_candidate(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "rule")

    result = remember_interaction(
        "uncertain-pref-user",
        "uncertain-pref-conversation",
        "我可能喜欢极简风格，但也不太确定。",
        "好的，我会先轻轻记一笔。",
    )

    assert_candidate(result, "preference", "极简风格")
    candidate = result["candidateMemories"][0]
    assert candidate["confidence"] == 0.65
    assert candidate["schemaPath"] == "preferences.general.likes"


def test_clear_and_strong_preference_confidence_tiers(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "rule")

    clear = remember_interaction("tier-user", "tier-conversation", "我喜欢西瓜", "记住了")
    strong = remember_interaction("tier-user", "tier-conversation", "我非常喜欢香蕉", "记住了")

    assert_candidate(clear, "food_preference", "西瓜")
    assert clear["candidateMemories"][0]["confidence"] == 0.85
    assert_candidate(strong, "food_preference", "香蕉")
    assert strong["candidateMemories"][0]["confidence"] == 0.9


def test_confirmed_candidate_boosts_but_keeps_low_confidence_tier(monkeypatch):
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)

    confirmed = upsert_structured_memory(
        "confirm-boost-user",
        "preference",
        "极简风格",
        source="source_ref:conversation:confirm-boost\n我可能喜欢极简风格，但也不太确定。",
        confidence=0.65,
        status="active",
    )

    assert confirmed["confidence"] == 0.75


def test_repeated_memory_confidence_caps_at_high_confidence(monkeypatch):
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)

    item = upsert_structured_memory("repeat-cap-user", "preference", "西瓜", confidence=0.9, status="active")
    for _ in range(5):
        item = upsert_structured_memory("repeat-cap-user", "preference", "西瓜", confidence=0.9, status="active")

    assert item["confidence"] == 0.95


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


def test_profile_summary_synthesizes_multiple_evidence_fragments(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)

    upsert_structured_memory(
        "synthesis-user",
        "negative_preference",
        "帽子",
        source="source_ref:conversation:synthesis-1\n我不喜欢帽子",
        confidence=0.88,
        status="active",
    )
    upsert_structured_memory(
        "synthesis-user",
        "negative_preference",
        "帽衫",
        source="source_ref:conversation:synthesis-2\n我也不喜欢帽衫",
        confidence=0.9,
        status="active",
    )

    memory = recall_memory("synthesis-user", None, "图片风格要注意什么")
    profile = next(item for item in memory["profiles"] if item["schemaPath"] == "preferences.visual.dislikes")

    assert profile["value"] == "帽子、帽衫"
    assert profile["summary"] == "用户在视觉形象上不喜欢帽子、帽衫。"
    assert "视觉负向偏好：用户在视觉形象上不喜欢帽子、帽衫。" in format_memory_context(memory)


def test_forget_request_candidate_uses_memory_control_schema(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "rule")

    result = remember_interaction("forget-candidate-user", "forget-candidate-conversation", "请忘掉帽子好吗", "好的")

    assert_candidate(result, "forget_request", "帽子")
    assert result["candidateMemories"][0]["schemaPath"] == "memory_control.forget_requests"


def test_confirmed_forget_request_marks_matching_memory_forgotten(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)

    upsert_structured_memory("forget-user", "negative_preference", "帽子", status="active")
    before = recall_memory("forget-user", None, "图片风格要注意什么")
    assert any(item["schemaPath"] == "preferences.visual.dislikes" for item in before["profiles"])

    upsert_structured_memory("forget-user", "forget_request", "帽子", status="active")

    memory = export_memory("forget-user", None)
    assert any(item["key"] == "negative_preference" and item["status"] == "forgotten" for item in memory["structured_all"])
    assert any(item["schemaPath"] == "preferences.visual.dislikes" and item["status"] == "forgotten" for item in memory["profiles"])
    after = recall_memory("forget-user", None, "图片风格要注意什么")
    assert all(item["schemaPath"] != "preferences.visual.dislikes" for item in after["profiles"])
    assert any(item["value"] == "帽子" for item in after["governance"]["blockedTerms"])


def test_confirmed_sensitive_request_disables_matching_profile_context(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)

    upsert_structured_memory("sensitive-user", "career", "后端开发工程师", status="active")
    upsert_structured_memory("sensitive-user", "sensitive_info", "后端开发工程师", status="active")

    memory = export_memory("sensitive-user", None)
    assert any(item["key"] == "career" and item["status"] == "disabled" for item in memory["structured_all"])
    assert any(item["schemaPath"] == "work.current_role" and item["status"] == "disabled" for item in memory["profiles"])
    context = format_memory_context(recall_memory("sensitive-user", None, "我的职业是什么"))
    assert "后端开发工程师" not in context


def test_sensitive_candidate_dedupes_full_sentence_fallback(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "rule")

    result = remember_interaction(
        "sensitive-dedupe-user",
        "sensitive-dedupe-conversation",
        "后端开发工程师这个职业信息是隐私，不要默认提。",
        "好的",
    )
    sensitive_candidates = [
        item for item in result["candidateMemories"]
        if item.get("targetKey") == "sensitive_info"
    ]

    assert len(sensitive_candidates) == 1
    assert sensitive_candidates[0]["value"] == "后端开发工程师这个职业信息"


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


def test_recall_planner_selects_visual_profile_without_unrelated_food(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)

    upsert_structured_memory("visual-recall-user", "food_preference", "西瓜", status="active")
    upsert_structured_memory("visual-recall-user", "negative_preference", "帽子", status="active")

    memory = recall_memory("visual-recall-user", None, "图片风格要注意什么")

    assert memory["recall_plan"]["intent"] == "visual"
    assert any(item["schemaPath"] == "preferences.visual.dislikes" for item in memory["profiles"])
    assert all(item["schemaPath"] != "preferences.food.likes" for item in memory["profiles"])


def test_recall_planner_combines_project_and_communication_paths(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)

    upsert_structured_memory("planner-combo-user", "career", "产品经理方向", status="active")
    upsert_structured_memory("planner-combo-user", "current_project", "Lyle AI", status="active")
    upsert_structured_memory("planner-combo-user", "preferences.output_style", "清楚直接", status="active")

    memory = recall_memory("planner-combo-user", None, "帮我规划产品定位文档，输出写法要清楚")
    paths = {item["schemaPath"] for item in memory["profiles"]}

    assert "project" in memory["recall_plan"]["intent"]
    assert "communication" in memory["recall_plan"]["intent"]
    assert {"work.current_role", "projects.current", "preferences.output_style"} <= paths


def test_profile_recall_usage_updates_hit_count_and_confidence(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)

    upsert_structured_memory("profile-usage-user", "career", "产品经理方向", confidence=0.8, status="active")

    before = get_user_profile("profile-usage-user", "work.current_role")
    memory = recall_memory("profile-usage-user", None, "我的职业是什么")
    after = get_user_profile("profile-usage-user", "work.current_role")

    assert any(item["schemaPath"] == "work.current_role" for item in memory["profiles"])
    assert before["hitCount"] == 0
    assert after["hitCount"] == 1
    assert after["confidence"] > before["confidence"]


def test_low_effective_confidence_profile_is_not_injected(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)

    upsert_structured_memory("profile-decay-user", "career", "过期角色", confidence=0.51, status="active")
    profile = profile_store._profiles["profile-decay-user"]["work.current_role"]
    profile.updated_at -= 90 * 24 * 60 * 60

    memory = recall_memory("profile-decay-user", None, "我的职业是什么")

    assert get_user_profile("profile-decay-user", "work.current_role")["confidence"] == 0.48
    assert all(item["schemaPath"] != "work.current_role" for item in memory["profiles"])


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


def test_llm_soft_preference_confidence_is_calibrated(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "llm")
    monkeypatch.setattr(settings, "structured_memory_confidence_threshold", 0.8)

    def fake_extract(question: str, recent_turns_text: str):
        assert "大概率还是会选更克制一点的设计" in question
        return [
            {"key": "preference", "value": "更克制一点的设计风格", "confidence": 0.9},
        ]

    monkeypatch.setattr("app.memory.store.invoke_structured_memory_chain", fake_extract)

    result = remember_interaction(
        "llm-soft-pref-user",
        "llm-soft-pref-conversation",
        "我不是完全排斥活泼风格，但如果要长期使用，我大概率还是会选更克制一点的设计。",
        "收到",
    )

    assert_candidate(result, "preference", "更克制一点的设计风格")
    assert result["candidateMemories"][0]["confidence"] == 0.72


def test_llm_low_confidence_candidate_is_kept_for_confirmation(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "llm")
    monkeypatch.setattr(settings, "structured_memory_confidence_threshold", 0.8)

    def fake_extract(question: str, recent_turns_text: str):
        return [
            {"key": "preference", "value": "低饱和配色", "confidence": 0.65},
            {"key": "preference", "value": "模型脑补内容", "confidence": 0.55},
        ]

    monkeypatch.setattr("app.memory.store.invoke_structured_memory_chain", fake_extract)

    result = remember_interaction(
        "llm-low-pref-user",
        "llm-low-pref-conversation",
        "我可能会更能接受低饱和配色。",
        "收到",
    )

    assert_candidate(result, "preference", "低饱和配色")
    assert result["candidateMemories"][0]["confidence"] == 0.65
    assert all(item.get("value") != "模型脑补内容" for item in result["candidateMemories"])


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


def test_recall_planner_uses_llm_plan_when_available(monkeypatch):
    from app.memory.recall_planner import plan_recall

    monkeypatch.setattr(settings, "recall_planner_llm_enabled", True)
    monkeypatch.setattr("app.chains.recall_planner_chain.is_recall_planner_model_configured", lambda: True)
    monkeypatch.setattr(
        "app.chains.recall_planner_chain.invoke_recall_planner_chain",
        lambda question: {
            "schemaPaths": ["preferences.visual.likes", "memory_control.sensitive_info", "unknown.path"],
            "reason": "用户同时询问视觉偏好和隐私边界。",
            "intent": "visual+sensitive",
            "confidence": 0.91,
        },
    )

    result = plan_recall("我喜欢什么设计风格，哪些隐私不要提？")

    assert result["planner"] == "llm"
    assert result["schemaPaths"] == ["preferences.visual.likes", "memory_control.sensitive_info"]
    assert result["confidence"] == 0.91


def test_profile_decay_worker_marks_expired_memory(monkeypatch):
    monkeypatch.setattr(settings, "postgres_enabled", False)
    profile_store._profiles.clear()
    profile_store.upsert_user_profile("decay-worker-user", "work.current_role", "过期角色", confidence=0.51)
    profile = profile_store._profiles["decay-worker-user"]["work.current_role"]
    profile.updated_at -= profile_store.PROFILE_DECAY_INTERVAL_SECONDS * 3

    result = profile_store.apply_profile_decay(user_id="decay-worker-user")

    assert result == {"updated": 1, "expired": 1}
    assert profile_store.get_user_profile("decay-worker-user", "work.current_role")["status"] == "expired"


def test_user_denial_marks_matching_structured_memory_conflicted(monkeypatch):
    monkeypatch.setattr(settings, "postgres_enabled", False)
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    upsert_structured_memory("denial-user", "career", "产品经理", confidence=0.9, status="active")

    remember_interaction("denial-user", "c1", "我不是产品经理了，你之前记错了。", "好的")

    memory = export_memory("denial-user", None)
    item = next(item for item in memory["structured_all"] if item["key"] == "career")
    assert item["status"] == "conflicted"
