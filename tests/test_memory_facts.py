from uuid import uuid4
import pytest

from app.core.config import settings
from app.memory import fact_store, schema_store, schema_builder
from app.memory.profile_store import (get_user_profile, upsert_user_profile, replace_user_profile_value,
                                     set_user_profile_status, update_user_profile_summary)
from app.memory.store import upsert_structured_memory, recall_memory, _legacy_load_all_structured


@pytest.fixture(autouse=True)
def layered_mode(monkeypatch):
    for field in ('postgres_enabled', 'redis_enabled', 'qdrant_enabled', 'milvus_enabled', 'memory_schema_build_with_llm'):
        monkeypatch.setattr(settings, field, False)
    monkeypatch.setattr(settings, 'memory_facts_enabled', True)
    monkeypatch.setattr(settings, 'memory_schema_enabled', True)


@pytest.fixture
def user():
    return 'facts-' + uuid4().hex


def save(user, path, value):
    return upsert_structured_memory(user, path, value, confidence=.95)


def test_confirmation_writes_only_facts_not_legacy_current_values(user):
    save(user, 'career', 'Java 后端工程师')
    facts = fact_store.list_facts(user)
    assert len(facts) == 1
    assert facts[0]['value'] == 'Java 后端工程师'
    assert _legacy_load_all_structured(user) == []
    from app.memory.profile_store import _legacy_list_user_profiles
    assert _legacy_list_user_profiles(user)[0]['value'] == ''
    assert get_user_profile(user, 'work.current_role')['value'] == facts[0]['value']


def test_multi_values_have_distinct_ids_and_duplicate_confirmation_is_idempotent(user):
    save(user, 'food_preference', '苹果、香蕉')
    before = fact_store.list_facts(user)
    save(user, 'food_preference', '苹果')
    after = fact_store.list_facts(user)
    assert len(after) == 2
    assert {fact['factId'] for fact in before} == {fact['factId'] for fact in after}
    assert len({fact['factId'] for fact in after}) == 2


def test_derived_summary_cannot_change_truth(user):
    save(user, 'career', '工程师')
    update_user_profile_summary(user, 'work.current_role', '我是医生', confidence_boost=1)
    assert get_user_profile(user, 'work.current_role')['value'] == '工程师'


def test_correction_replaces_current_fact_preserving_history(user):
    save(user, 'career', '工程师')
    profile = get_user_profile(user, 'work.current_role')
    from app.memory.governance import govern_memory_record
    govern_memory_record(user, f"profile:{profile['id']}", action='CORRECT', value='产品经理', client_request_id=uuid4().hex)
    assert get_user_profile(user, 'work.current_role')['value'] == '产品经理'
    assert [fact['value'] for fact in fact_store.list_facts(user)] == ['产品经理']
    old = [fact for fact in fact_store.list_facts(user, include_history=True) if fact['value'] == '工程师'][0]
    assert old['status'] == 'superseded'
    assert old['revision'] == 2


def test_core_profile_is_present_on_greetings_without_waiting_for_schema(user):
    from app.agents.memory_agent import run_memory_agent
    save(user, 'career', 'Java 后端工程师')
    save(user, 'preferences.communication_style', '先给结论')
    result = run_memory_agent(user, None, '你好')
    assert not result['skipped']
    assert 'Java 后端工程师' in result['memory_context']
    assert '先给结论' in result['memory_context']
    assert any(item['reasonCode'] == 'CORE_PROFILE' for item in result['memory']['used_memories'])


def test_core_revocation_and_correction_do_not_wait_for_worker(user):
    save(user, 'career', '工程师')
    schema_builder.build_user_schema(user)
    replace_user_profile_value(user, 'work.current_role', '产品经理')
    result = recall_memory(user, None, '你好', depth='light')
    assert '产品经理' in result['compiled_context']
    assert '工程师' not in result['compiled_context']
    set_user_profile_status(user, 'work.current_role', 'disabled')
    assert '产品经理' not in recall_memory(user, None, '你好', depth='light')['compiled_context']


def test_scene_filter_precedes_small_schema_full_selection(user):
    save(user, 'career', '工程师')
    save(user, 'food_preference', '苹果')
    save(user, 'travel_interest', '杭州')
    schema_builder.build_user_schema(user)
    result = recall_memory(user, None, '晚餐吃什么')
    assert '苹果' in result['compiled_context']
    assert '杭州' not in result['compiled_context']
    assert result['compiled_context'].count('工程师') == 1


def test_learning_plan_retrieves_role_without_career_keyword_or_short_term(user):
    save(user, 'career', 'Java 后端工程师')
    save(user, 'goals.current_week', '学习数据库')
    schema_builder.build_user_schema(user)
    from app.agents.memory_agent import run_memory_agent
    result = run_memory_agent(user, 'new-conversation', '安排一周学习计划')
    assert 'Java 后端工程师' in result['memory_context']
    assert '学习数据库' in result['memory_context']


def test_incremental_build_reuses_unmodified_slots(user, monkeypatch):
    save(user, 'career', '工程师')
    save(user, 'food_preference', '苹果')
    schema_builder.build_user_schema(user)
    replace_user_profile_value(user, 'work.current_role', '产品经理')
    called = []
    original = schema_builder.build_document
    def capture(inputs):
        called.append(set(inputs))
        return original(inputs)
    monkeypatch.setattr(schema_builder, 'build_document', capture)
    schema_builder.build_user_schema(user)
    assert called == [{'work.current_role'}]
    snapshot = schema_store.read_state(user)['snapshots'][0]
    assert 'preferences.food.likes' in snapshot['reusedPaths']


def test_migration_preserves_ids_and_does_not_resurrect_archived_rows(user, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(settings, 'memory_facts_enabled', False)
        save(user, 'career', '工程师')
        old = get_user_profile(user, 'work.current_role')
    new = get_user_profile(user, 'work.current_role')
    assert old['id'] == new['id']
    replace_user_profile_value(user, 'work.current_role', '产品经理')
    assert get_user_profile(user, 'work.current_role')['value'] == '产品经理'
    assert _legacy_load_all_structured(user)[0].value == '工程师'
    assert fact_store.migration_report(user)['imported'] == 1


def test_allowed_paths_limits_core_and_details(user):
    save(user, 'career', '工程师')
    assert recall_memory(user, None, '你好', allowed_paths=[])['profiles'] == []


def test_users_do_not_share_facts(user):
    save(user, 'career', '工程师')
    assert fact_store.list_facts(user + '-other') == []


def test_expired_fact_not_used_by_core(user):
    upsert_structured_memory(user, 'career', '工程师', expires_at=1)
    assert '工程师' not in recall_memory(user, None, '你好', depth='light')['compiled_context']


def test_individual_disable_does_not_remove_siblings_or_leak_to_builder(user):
    save(user, 'food_preference', '苹果、香蕉')
    banana = next(f for f in fact_store.list_facts(user) if f['value'] == '香蕉')
    fact_store.revise_fact(user, banana['factId'], expected_revision=1, status='disabled')
    assert get_user_profile(user, 'preferences.food.likes')['value'] == '苹果'
    inputs = schema_builder.collect_inputs(user)
    import json
    assert '香蕉' not in json.dumps(inputs, ensure_ascii=False)
    schema_builder.build_user_schema(user)
    assert '香蕉' not in recall_memory(user, None, '晚餐吃什么')['compiled_context']
    assert '苹果' in recall_memory(user, None, '晚餐吃什么')['compiled_context']


def test_fact_revision_check_and_user_scope(user):
    save(user, 'food_preference', '苹果')
    fact = fact_store.list_facts(user)[0]
    fact_store.revise_fact(user, fact['factId'], expected_revision=1, value='香蕉')
    with pytest.raises(fact_store.FactConflict):
        fact_store.revise_fact(user, fact['factId'], expected_revision=1, status='disabled')
    with pytest.raises(KeyError):
        fact_store.revise_fact(user + '-other', fact['factId'], expected_revision=2, status='disabled')
    assert fact_store.list_facts(user)[0]['value'] == '香蕉'


def test_conflicting_legacy_values_are_quarantined_not_silently_chosen(user, monkeypatch):
    from app.memory import profile_store, store
    with monkeypatch.context() as patch:
        patch.setattr(settings, 'memory_facts_enabled', False)
        save(user, 'career', '工程师')
        # Model a historical split-brain value without using a new facts command.
        profile_store._profiles[user]['work.current_role'].value = '产品经理'
    profile = get_user_profile(user, 'work.current_role')
    assert profile['status'] == 'conflicted'
    assert fact_store.migration_report(user)['conflicts'] == ['work.current_role']
    assert not recall_memory(user, None, '你好', depth='light')['profiles']
    replace_user_profile_value(user, 'work.current_role', '产品经理')
    assert fact_store.migration_report(user)['conflicts'] == []


def test_purge_removes_facts_history_and_does_not_reimport_old_rows(user):
    from app.memory.profile_store import delete_user_profile
    save(user, 'career', '工程师')
    replace_user_profile_value(user, 'work.current_role', '产品经理')
    delete_user_profile(user, 'work.current_role')
    assert fact_store.list_facts(user, include_history=True) == []
    assert get_user_profile(user, 'work.current_role') is None


def test_manual_full_rebuild_does_not_reuse_slots(user):
    save(user, 'career', '工程师')
    schema_builder.build_user_schema(user)
    schema_store.request_build(user, force=True)
    schema_builder.build_user_schema(user)
    snapshot = schema_store.read_state(user)['snapshots'][0]
    assert snapshot['reusedPaths'] == []
    assert snapshot['rebuiltPaths'] == ['work.current_role']


def test_core_budget_is_bounded_and_excludes_unrelated_interests(user, monkeypatch):
    monkeypatch.setattr(settings, 'memory_core_max_chars', 256)
    save(user, 'interaction.boundaries', '不要追问家庭')
    save(user, 'preferences.communication_style', '简短' * 300)
    save(user, 'travel_interest', '杭州')
    result = recall_memory(user, None, '你好', depth='light')
    assert '不要追问家庭' in result['compiled_context']
    assert '简短' * 300 not in result['compiled_context']
    assert '杭州' not in result['compiled_context']


def test_current_project_continues_without_keywords_but_not_across_users_or_conversations(monkeypatch):
    from app.projects import project_resolver as resolver
    projects = [{'id': 1, 'name': '支付系统', 'status': 'active'}]
    monkeypatch.setattr(resolver, 'list_project_links', lambda user, *args, **kwargs: [
        {'targetId': 'conversation-a', 'confidence': 1}] if user == 'owner' else [])
    assert resolver._project_from_conversation_link('owner', 'conversation-a', projects, '明白了')[0]['id'] == 1
    assert resolver._project_from_conversation_link('owner', 'conversation-b', projects, '明白了') is None
    assert resolver._project_from_conversation_link('other', 'conversation-a', projects, '明白了') is None
    assert resolver._project_from_conversation_link('owner', 'conversation-a', [*projects, {'id': 2}], '继续') is None


def test_fact_api_requires_revision_and_preserves_sibling_values(user, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    monkeypatch.setattr(settings, 'internal_api_token', 'memory-facts-test-internal-token')
    client = TestClient(app)
    headers = {'X-Internal-Token': settings.internal_api_token}
    save(user, 'food_preference', '苹果、香蕉')
    url = f'/ai/memory/{user}/facts'
    facts = client.get(url, headers=headers).json()['items']
    target = next(f for f in facts if f['value'] == '香蕉')
    request = {'expectedRevision': target['revision'], 'status': 'disabled'}
    assert client.put(url + '/' + target['factId'], headers=headers, json=request).status_code == 200
    assert client.put(url + '/' + target['factId'], headers=headers, json=request).status_code == 409
    assert client.put(f'/ai/memory/other/facts/{target["factId"]}', headers=headers, json=request).status_code == 404
    assert get_user_profile(user, 'preferences.food.likes')['value'] == '苹果'


def test_work_summary_does_not_erase_core_profile_from_stream_prompt(user, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.agents.memory_agent import run_memory_agent
    save(user, 'preferences.communication_style', '先给结论')
    context = run_memory_agent(user, None, '你好')
    context.update(contexts=[], tool_context='工具结果', tool_results=[{'tool': 'work_summary', 'ok': True}],
                   agent_trace=[], rag_sufficiency={})
    monkeypatch.setattr(settings, 'internal_api_token', 'memory-facts-stream-internal-token')
    monkeypatch.setattr('app.api.chat.prepare_multi_agent_context', lambda *a, **k: context)
    monkeypatch.setattr('app.api.chat._project_context_for_response', lambda *a, **k: {})
    captured = []
    def stream(*args, **kwargs):
        captured.append(args[2])
        return iter(['你好'])
    monkeypatch.setattr('app.api.chat.stream_answer', stream)
    monkeypatch.setattr('app.api.chat.run_reviewer_agent', lambda *a, **k: {'approved': True, 'score': 1, 'skipped': True})
    response = TestClient(app).post('/ai/chat/stream', headers={'X-Internal-Token': settings.internal_api_token},
        json={'userId': user, 'conversationId': 'test', 'message': '你好', 'knowledgeBaseIds': [], 'stream': True, 'aiConfig': {}})
    assert response.status_code == 200
    assert captured and '先给结论' in captured[0]


def test_fact_transaction_failure_does_not_apply_partial_edit(user):
    save(user, 'career', '工程师')
    before = fact_store.list_facts(user)
    with pytest.raises(RuntimeError):
        with fact_store._edit(user) as state:
            state['facts'].clear()
            raise RuntimeError('rollback')
    assert fact_store.list_facts(user) == before


def test_corrected_fact_blocks_old_summary(user):
    from app.memory.store import _memory_governance_snapshot, _is_governed_memory_text
    save(user, 'career', 'Java 后端工程师')
    replace_user_profile_value(user, 'work.current_role', '产品经理')
    assert _is_governed_memory_text('用户职业是 Java 后端工程师', _memory_governance_snapshot(user))
