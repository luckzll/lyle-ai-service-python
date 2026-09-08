from uuid import uuid4

import pytest

from app.agents.memory_agent import decide_memory_recall, run_memory_agent
from app.core.config import settings
from app.memory import profile_builder, store
from app.memory.profile_schema import PROFILE_SCHEMA
from app.memory.profile_store import get_user_profile, set_user_profile_status
from app.memory.recall_policy import ACTION_CONSTRAINT_PATHS, constrain_plan, recall_policy


@pytest.fixture
def user(monkeypatch):
    for flag in ("postgres_enabled", "redis_enabled", "qdrant_enabled", "milvus_enabled", "rerank_enabled"):
        monkeypatch.setattr(settings, flag, False)
    monkeypatch.setattr(settings, "recall_planner_llm_enabled", False)
    monkeypatch.setattr(settings, "structured_memory_extractor", "rule")
    return f"policy-{uuid4()}"


def test_light_avoids_planner_model_summary_and_vector_queries(user, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Light recall must not call this source")
    monkeypatch.setattr("app.memory.recall_planner._plan_recall_with_model", unexpected)
    monkeypatch.setattr(store, "_search_qdrant_memory", unexpected)
    monkeypatch.setattr(store, "_search_milvus_memory", unexpected)
    monkeypatch.setattr(store, "_load_summary_with_meta", unexpected)
    monkeypatch.setattr(settings, "recall_planner_llm_enabled", True)
    store.add_turn(user, "one", "user", "上一轮的话")
    result = run_memory_agent(user, "one", "帮我解释量子力学")
    assert result["decision"]["depth"] == "light"
    assert result["memory"]["recall_policy"]["includeVector"] is False
    assert "上一轮的话" in result["memory_context"]


def test_full_uses_both_vector_sources_and_relevant_slots(user, monkeypatch):
    calls = []
    for backend in ("qdrant", "milvus"):
        monkeypatch.setattr(store, f"_search_{backend}_memory", lambda *args, b=backend: calls.append(b) or [])
    store.upsert_structured_memory(user, "projects.current", "Lyle")
    result = run_memory_agent(user, "one", "继续规划我的项目")
    assert result["decision"]["depth"] == "full"
    assert set(calls) == {"qdrant", "milvus"}
    assert "Lyle" in result["memory_context"]


def test_default_constraints_have_priority_and_usage_reason(user, monkeypatch):
    monkeypatch.setattr(settings, "memory_recall_limit", 1)
    for path in ACTION_CONSTRAINT_PATHS:
        store.upsert_structured_memory(user, path, f"已确认的{PROFILE_SCHEMA[path].label}")
    store.upsert_structured_memory(user, "projects.current", "Lyle")
    memory = store.recall_memory(user, "one", "我的项目", include_vector=False)
    assert {item["schemaPath"] for item in memory["profiles"]} == {*ACTION_CONSTRAINT_PATHS, "projects.current"}
    assert sum(item["reasonCode"] == "ACTION_CONSTRAINT" for item in memory["used_memories"]) == len(ACTION_CONSTRAINT_PATHS)
    assert memory["profiles"][0]["schemaPath"] == "interaction.boundaries"


@pytest.mark.parametrize("depth", ["light", "full"])
def test_budget_and_usage_refer_only_to_injected_profiles(user, monkeypatch, depth):
    monkeypatch.setattr(settings, f"memory_{depth}_context_max_chars", 400)
    store.upsert_structured_memory(user, "interaction.boundaries", "不要主动提及家庭")
    store.upsert_structured_memory(user, "preferences.output_style", "很长的风格" * 200)
    store.add_turn(user, "one", "user", "短消息")
    store._save_summary(user, "one", "超长摘要" * 500)
    monkeypatch.setattr(store, "_search_qdrant_memory", lambda *args: [{"text": "历史片段" * 1000, "score": 0.99}])
    memory = store.recall_memory(user, "one", "我的记忆", depth=depth)
    context = store.format_memory_context(memory)
    assert 0 < len(context) <= 400
    assert "不要主动提及家庭" in context
    assert "很长的风格" not in context
    assert len(memory["used_memories"]) == 1
    assert memory["retrieved"] == []
    assert get_user_profile(user, "preferences.output_style")["hitCount"] == 0


@pytest.mark.parametrize("status", ["pending", "disabled", "forgotten", "conflicted", "expired"])
def test_governance_filters_profile_structured_summary_and_vector(user, monkeypatch, status):
    secret = "蓝色风铃方案"
    key = "pending:test:projects.current" if status == "pending" else "projects.current"
    store.upsert_structured_memory(user, key, secret, status="pending" if status == "pending" else "active")
    if status != "pending":
        store.set_structured_memory_status(user, key, status)
    store._save_summary(user, "one", f"用户正在做{secret}")
    monkeypatch.setattr(store, "_search_qdrant_memory", lambda *args: [{"text": f"用户：{secret}", "score": 0.99}])
    memory = store.recall_memory(user, "one", "我的项目")
    assert secret not in store.format_memory_context(memory)
    assert memory["profiles"] == []
    assert memory["structured"] == []
    assert memory["retrieved"] == []


def test_expired_legacy_row_does_not_leak_through_active_projection(user):
    store.upsert_structured_memory(user, "projects.current", "过期方案", expires_at=1)
    assert "过期方案" not in store.format_memory_context(store.recall_memory(user, None, "我的项目"))


def test_direct_profile_conflict_cannot_fall_back_to_active_legacy_row(user):
    store.upsert_structured_memory(user, "career", "工程师")
    set_user_profile_status(user, "work.current_role", "conflicted")
    memory = store.recall_memory(user, None, "我的职业")
    assert memory["structured"] == []
    assert "工程师" not in store.format_memory_context(memory)


def test_pending_source_blocks_stream_write_even_if_extracted_value_is_paraphrased(user, monkeypatch):
    text = "我的语言习惯是少用那些拗口的术语"
    store.upsert_structured_memory(user, "pending:one:preferences.verbal_habits", "避免专业行话",
                                   status="pending", source=f"source_ref:conversation:one\n{text}")
    monkeypatch.setattr(store, "_upsert_qdrant_memory", lambda *args: pytest.fail("pending source was indexed"))
    result = store.remember_interaction(user, "one", text, "请确认", extract_candidates=False)
    assert result["vectorMemoryWrite"]["reason"] == "unconfirmed_or_governed_memory"


def test_confirmation_consumes_alias_candidate_but_not_other_values(user):
    store.upsert_structured_memory(user, "pending:one:career", "程序员", status="pending")
    store.upsert_structured_memory(user, "pending:two:career", "设计师", status="pending")
    store.upsert_structured_memory(user, "work.current_role", "程序员")
    rows = {item.key: item.status for item in store._load_all_structured(user)}
    assert rows["pending:one:career"] == "disabled"
    assert rows["pending:two:career"] == "pending"
    assert "程序员" in store.format_memory_context(store.recall_memory(user, None, "我的职业"))


def test_candidate_category_correction_does_not_block_confirmed_value(user):
    store.upsert_structured_memory(user, "pending:one:preference", "用简短句子回答", status="pending")
    store.upsert_structured_memory(user, "preferences.output_style", "用简短句子回答")
    store.set_structured_memory_status(user, "pending:one:preference", "disabled")
    memory = store.recall_memory(user, None, "解释一个概念", depth="light")
    assert "用简短句子回答" in store.format_memory_context(memory)


def test_unknown_model_paths_cannot_expand_scope_or_sensitive_access(user):
    plan = constrain_plan({"schemaPaths": ["*", "system.override"]}, recall_policy("full"))
    assert "memory_control.sensitive_info" not in plan["schemaPaths"]
    assert "memory_control.forget_requests" not in plan["schemaPaths"]
    assert "system.override" not in plan["schemaPaths"]
    with pytest.raises(ValueError):
        recall_policy("admin")


def test_construction_whitelists_schema_and_dedupes_aliases(user):
    candidates = store._dedupe_structured_candidates([
        ("career", "程序员", .85), ("work.current_role", "程序员", .9),
        ("system.override", "忽略指令", 1),
    ])
    assert candidates == [("career", "程序员", .9)]
    result = store.extract_structured_memory(user, "one", "我的边界是不要主动追问家庭", extraction_mode="rule")
    assert result[0]["schemaPath"] == "interaction.boundaries"
    assert result[0]["status"] == "pending"


def test_summary_ignores_governed_evidence_and_keeps_sentence_intact(user):
    evidence = [
        {"relation": "support", "fragment": {"normalizedText": "不与人争论", "status": "active"}},
        {"relation": "support", "fragment": {"normalizedText": "旧内容", "status": "disabled"}},
        {"relation": "support", "fragment": {"normalizedText": "已遗忘", "status": "forgotten"}},
    ]
    assert profile_builder._evidence_values(evidence) == ["不与人争论"]
    store.upsert_structured_memory(user, "preferences.communication_style", "温和与直接并重")
    assert get_user_profile(user, "preferences.communication_style")["summary"] == "温和与直接并重"


def test_management_export_does_not_mark_memory_used_or_query_model(user, monkeypatch):
    store.upsert_structured_memory(user, "career", "程序员")
    monkeypatch.setattr(store, "plan_recall", lambda *args, **kwargs: pytest.fail("management invoked planner"))
    store.export_memory(user, "one")
    assert get_user_profile(user, "work.current_role")["hitCount"] == 0


def test_session_and_user_isolation(user):
    store._save_summary(user, "one", "另一个会话的秘密")
    store.add_turn(user, "one", "user", "另一个会话的问题")
    store.upsert_structured_memory(user + "-other", "identity.name", "另一个用户")
    context = store.format_memory_context(store.recall_memory(user, "two", "你记得我什么"))
    assert "另一个" not in context


def test_vector_text_does_not_copy_unrelated_confirmed_facts(user):
    store.upsert_structured_memory(user, "identity.name", "Lyle")
    assert store._build_vector_memory_text(user, "如何优化工具调用") == "用户：如何优化工具调用"


def test_generic_first_person_request_is_light_but_history_and_project_are_full():
    assert decide_memory_recall("帮我解释这个概念")["depth"] == "light"
    assert decide_memory_recall("继续规划项目")["depth"] == "full"
    assert decide_memory_recall("上次的方案")["depth"] == "full"
    assert decide_memory_recall("晚饭吃什么")["depth"] == "full"
    assert decide_memory_recall("今天天气怎么样")["depth"] == "light"


def test_repeated_construction_does_not_silently_resolve_conflict(user):
    store.upsert_structured_memory(user, "work.current_role", "程序员")
    store.upsert_structured_memory(user, "work.current_role", "设计师")
    assert get_user_profile(user, "work.current_role")["status"] == "conflicted"
    store.upsert_structured_memory(user, "work.current_role", "程序员")
    assert get_user_profile(user, "work.current_role")["status"] == "conflicted"


def test_conflict_blocks_both_old_and_new_values_in_legacy_vectors(user, monkeypatch):
    store.upsert_structured_memory(user, "work.current_role", "旧岗位工程师")
    store.upsert_structured_memory(user, "work.current_role", "新岗位设计师")
    monkeypatch.setattr(store, "_search_qdrant_memory", lambda *args: [
        {"text": "我的职业是旧岗位工程师", "score": .9},
        {"text": "我的职业是新岗位设计师", "score": .9},
    ])
    memory = store.recall_memory(user, None, "我的职业")
    assert memory["retrieved"] == []
    assert memory["profiles"] == []
    assert memory["structured"] == []


def test_memory_snapshot_tool_never_reintroduces_management_bodies(user):
    from app.tools.tool_registry import _memory_snapshot, format_tool_context
    store._save_summary(user, "one", "应停用的私密旧摘要")
    store.upsert_structured_memory(user, "identity.name", "私密姓名")
    result = _memory_snapshot(user, "one", "你记得我什么")
    assert "私密" not in str(result)
    assert "摘要数量：1" in format_tool_context([{"tool": "memory_snapshot", **result}])


def test_tool_router_keeps_recent_context_without_full_recall(user, monkeypatch):
    from app.agents.tool_agent import _route_tools_with_context
    monkeypatch.setattr(settings, "tool_router_llm_enabled", True)
    store.add_turn(user, "one", "user", "刚才生成了猫的图片")
    captured = []
    monkeypatch.setattr("app.agents.tool_agent.route_tools_with_llm", lambda question, context: captured.append(context) or {})
    monkeypatch.setattr(store, "_search_qdrant_memory", lambda *args: pytest.fail("router triggered vector recall"))
    _route_tools_with_context(user, "one", "把它换成蓝色")
    assert "猫的图片" in captured[0]


def test_same_message_confirmation_does_not_inflate_source_count(user):
    from app.memory.evidence_store import list_profile_evidence
    source = "source_ref:conversation:one;message:message-1\n我的说话风格是温和直接"
    for _ in range(2):
        store.upsert_structured_memory(user, "preferences.communication_style", "温和直接", source=source)
    profile = get_user_profile(user, "preferences.communication_style")
    assert len(list_profile_evidence(user, int(profile["id"]))) == 1
    store.upsert_structured_memory(user, "preferences.communication_style", "温和直接", source=source.replace("message-1", "message-2"))
    assert len(list_profile_evidence(user, int(profile["id"]))) == 2
