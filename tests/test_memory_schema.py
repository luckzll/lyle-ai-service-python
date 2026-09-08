from copy import deepcopy
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage

from app.core.config import settings
from app.memory import schema_builder as builder, schema_recall as recall, schema_store as queue
from app.memory.profile_store import upsert_user_profile, replace_user_profile_value, set_user_profile_status
from app.memory.store import recall_memory, upsert_structured_memory


@pytest.fixture(autouse=True)
def schema_mode(monkeypatch):
    monkeypatch.setattr(settings, 'memory_facts_enabled', False)
    for name in ('postgres_enabled', 'redis_enabled', 'qdrant_enabled', 'milvus_enabled', 'memory_schema_build_with_llm'):
        monkeypatch.setattr(settings, name, False)
    monkeypatch.setattr(settings, 'memory_schema_enabled', True)


@pytest.fixture
def user():
    return 'schema-test-' + uuid4().hex


def seed(user, path='identity.name', value='小林'):
    return upsert_user_profile(user, path, value, confidence=.95)


def publish(user):
    result = builder.build_user_schema(user)
    assert result['status'] == 'ready'
    return result


def test_confirmed_memory_queues_build_but_does_not_recall_before_publish(user):
    upsert_structured_memory(user, 'name', '小林', confidence=.95)
    assert queue.read_state(user)['status'] == 'pending'
    before = recall_memory(user, None, '你记得我叫什么吗')
    assert before['profiles'] == []
    assert before['structured'] == []
    assert before['schema_recall']['status'] == 'pending'
    publish(user)
    after = recall_memory(user, None, '你记得我叫什么吗')
    assert after['profiles'][0]['sourceIds'][0].startswith('fragment:')
    assert '小林' in after['compiled_context']
    assert after['schema_recall']['version'] == 1
    assert after['schema_recall']['injectedPaths'] == ['identity.name']
    assert after['profiles'][0]['schemaVersion'] == 1


def test_pending_candidate_does_not_enter_build(user):
    upsert_structured_memory(user, 'pending:one:name', '小林', confidence=.95, status='pending')
    assert builder.collect_inputs(user) == {}


def test_source_mutation_invalidates_old_slot_before_rebuild(user):
    seed(user)
    seed(user, 'identity.language', '中文')
    publish(user)
    replace_user_profile_value(user, 'identity.name', '小张')
    slots, status = recall.available_schema(user)
    assert status['status'] == 'pending'
    assert 'identity.name' not in slots
    assert 'identity.language' in slots
    publish(user)
    result = recall_memory(user, None, '你记得我叫什么吗')
    assert '小张' in result['compiled_context']
    assert '小林' not in result['compiled_context']
    assert result['schema_recall']['version'] == 2


@pytest.mark.parametrize('status', ['disabled', 'forgotten', 'conflicted', 'expired'])
def test_unavailable_sources_cannot_recall_old_snapshot(user, status):
    seed(user)
    publish(user)
    set_user_profile_status(user, 'identity.name', status)
    assert recall.available_schema(user)[0] == {}


def test_low_confidence_and_sensitive_not_built(user):
    upsert_user_profile(user, 'identity.name', '低置信度', confidence=.3)
    seed(user, 'memory_control.sensitive_info', '敏感标记')
    assert builder.collect_inputs(user) == {}


def test_published_schemas_are_user_isolated(user):
    seed(user)
    publish(user)
    assert recall.available_schema(user + '-other')[0] == {}


def test_publication_rejects_superseded_revision(user):
    queue.request_build(user, 'one')
    claim = queue.claim_build(user)
    queue.request_build(user, 'two')
    assert not queue.finish_build(user, claim, {'slots': {}, 'etag': 'old'})
    assert queue.read_state(user)['snapshots'] == []
    assert queue.read_state(user)['status'] == 'pending'


def test_expired_worker_cannot_publish_over_new_owner(user, monkeypatch):
    queue.request_build(user, 'one')
    first = queue.claim_build(user)
    assert queue.claim_build(user) is None
    monkeypatch.setattr(queue, 'time', lambda: first['leaseUntil'] + 1)
    second = queue.claim_build(user)
    assert second['buildId'] != first['buildId']
    assert not queue.finish_build(user, first, {'slots': {}, 'etag': 'old'})
    assert queue.finish_build(user, second, {'slots': {}, 'etag': 'new'})


def test_model_failure_preserves_last_version_and_requires_retry(user, monkeypatch):
    seed(user)
    publish(user)
    queue.request_build(user, force=True)
    with monkeypatch.context() as patch:
        patch.setattr(builder, 'build_document', lambda _: (_ for _ in ()).throw(ValueError('INVALID_SCHEMA_SOURCES')))
        assert builder.build_user_schema(user)['status'] == 'failed'
    assert queue.read_state(user)['version'] == 1
    assert recall.available_schema(user)[0]
    assert queue.read_state(user)['status'] == 'failed'
    queue.request_build(user, force=True)
    assert publish(user)['version'] == 2


def test_change_during_build_cannot_publish(user, monkeypatch):
    seed(user)
    original = builder.build_document
    def changing(inputs):
        result = original(inputs)
        replace_user_profile_value(user, 'identity.name', '小张')
        return result
    monkeypatch.setattr(builder, 'build_document', changing)
    assert builder.build_user_schema(user)['status'] == 'pending'
    assert queue.read_state(user)['version'] == 0


def test_history_bounded_and_privacy_purge_removes_all_versions(user, monkeypatch):
    monkeypatch.setattr(settings, 'memory_schema_history_limit', 2)
    seed(user)
    for _ in range(3):
        queue.request_build(user, force=True)
        publish(user)
    assert len(queue.read_state(user)['snapshots']) == 2
    queue.purge_schema_history(user)
    state = queue.read_state(user)
    assert state['snapshots'] == []
    assert state['version'] == 3


def test_build_rejects_foreign_provenance_and_missing_slots(user, monkeypatch):
    seed(user)
    inputs = builder.collect_inputs(user)
    monkeypatch.setattr(settings, 'memory_schema_build_with_llm', True)
    monkeypatch.setattr(builder, 'is_task_model_configured', lambda _: True)
    monkeypatch.setattr(builder, '_model_build', lambda _: {'slots': [{'path': 'identity.name', 'items': [{'value': '小林', 'sourceIds': ['foreign']}]}]})
    with pytest.raises(ValueError, match='INVALID_SCHEMA_SOURCES'):
        builder.build_document(inputs)
    monkeypatch.setattr(builder, '_model_build', lambda _: {'slots': []})
    with pytest.raises(ValueError, match='INCOMPLETE_SCHEMA'):
        builder.build_document(inputs)


def test_llm_build_publishes_validated_derived_output(user, monkeypatch):
    seed(user)
    inputs = builder.collect_inputs(user)
    source_id = inputs['identity.name']['sources'][0]['id']
    monkeypatch.setattr(settings, 'memory_schema_build_with_llm', True)
    monkeypatch.setattr(builder, 'is_task_model_configured', lambda _: True)
    monkeypatch.setattr(builder, '_model_build', lambda _: {'slots': [{'path': 'identity.name', 'items': [{'value': '称呼：小林', 'sourceIds': [source_id]}], 'overview': '称呼'}]})
    assert publish(user)['engine'] == 'llm'
    assert '称呼：小林' in recall_memory(user, None, '你记得我什么')['compiled_context']


def test_light_recall_only_fixed_constraints_and_scope_is_not_expanded(user):
    seed(user)
    seed(user, 'identity.language', '中文')
    publish(user)
    result = recall_memory(user, None, '解释一下', depth='light')
    assert result['schema_recall']['injectedPaths'] == ['identity.language']
    result = recall_memory(user, None, '解释一下', allowed_paths=[])
    assert result['profiles'] == []


def fake_model(monkeypatch, actions):
    import json
    class Model:
        def bind(self, **kwargs):
            return self
        def invoke(self, messages):
            self.messages = deepcopy(messages)
            return AIMessage(content=json.dumps(next(actions)))
    model = Model()
    monkeypatch.setattr(recall, 'is_task_model_configured', lambda _: True)
    monkeypatch.setattr(recall, 'build_chat_model_for_task', lambda _: model)
    return model


def test_agent_performs_bounded_read_then_finish(user, monkeypatch):
    seed(user)
    publish(user)
    model = fake_model(monkeypatch, iter([
        {'action': 'read_schema', 'paths': ['identity.name']},
        {'action': 'finish', 'paths': ['identity.name']},
    ]))
    result = recall_memory(user, None, '我叫什么', schema_mode='agent')
    assert result['schema_recall']['mode'] == 'agent'
    assert len(result['schema_recall']['trace']) == 2
    assert 'toolResult' in model.messages[-1].content
    assert '小林' in result['compiled_context']


def test_single_pass_does_not_need_agent_reads(user, monkeypatch):
    seed(user)
    publish(user)
    fake_model(monkeypatch, iter([{'action': 'finish', 'paths': ['identity.name']}]))
    result = recall_memory(user, None, '我叫什么', schema_mode='single_pass')
    assert result['schema_recall']['mode'] == 'single_pass'


def test_change_during_agent_read_is_revoked_before_injection(user, monkeypatch):
    seed(user)
    publish(user)
    monkeypatch.setattr(recall, 'is_task_model_configured', lambda _: True)
    def selecting(*args):
        replace_user_profile_value(user, 'identity.name', '小张')
        return ['identity.name'], []
    monkeypatch.setattr(recall, '_select_with_model', selecting)
    result = recall_memory(user, None, '我叫什么', schema_mode='agent')
    assert result['profiles'] == []
    assert '小林' not in result['compiled_context']


def test_agent_searches_scoped_values(user, monkeypatch):
    seed(user)
    publish(user)
    fake_model(monkeypatch, iter([
        {'action': 'search_schema', 'query': '小林'},
        {'action': 'finish', 'paths': ['identity.name']},
    ]))
    result = recall_memory(user, None, '我叫什么', schema_mode='agent')
    assert result['schema_recall']['mode'] == 'agent'
    assert result['schema_recall']['trace'][0]['paths'] == ['identity.name']


def test_agent_step_limit_is_enforced(user, monkeypatch):
    seed(user)
    publish(user)
    monkeypatch.setattr(settings, 'memory_schema_agent_max_steps', 1)
    fake_model(monkeypatch, iter([{'action': 'read_schema', 'paths': ['identity.name']}]))
    assert recall_memory(user, None, '我叫什么', schema_mode='agent')['schema_recall']['mode'] == 'rule_fallback'


def test_agent_rejects_unread_paths_and_falls_back(user, monkeypatch):
    seed(user)
    publish(user)
    fake_model(monkeypatch, iter([{'action': 'finish', 'paths': ['identity.name']}]))
    result = recall_memory(user, None, '我叫什么', schema_mode='agent')
    assert result['schema_recall']['mode'] == 'rule_fallback'
    assert result['schema_recall']['recallError']


def test_small_schema_selects_all_without_model(user, monkeypatch):
    seed(user)
    seed(user, 'identity.language', '中文')
    publish(user)
    monkeypatch.setattr(recall, '_select_with_model', lambda *args: pytest.fail('small schema should not call LLM'))
    result = recall_memory(user, None, '我叫什么')
    assert result['schema_recall']['mode'] == 'full_schema'
    assert set(result['schema_recall']['injectedPaths']) == {'identity.name', 'identity.language'}


def test_no_model_reports_real_fallback_mode(user):
    seed(user)
    publish(user)
    result = recall_memory(user, None, '我叫什么', schema_mode='agent')
    assert result['schema_recall']['mode'] == 'rule_fallback'
    assert result['schema_recall']['recallError'] == 'RECALL_MODEL_NOT_CONFIGURED'


def test_reading_and_usage_do_not_invalidate_snapshot(user):
    seed(user)
    publish(user)
    for _ in range(3):
        result = recall_memory(user, None, '我叫什么')
        assert result['schema_recall']['status'] == 'ready'
    assert queue.read_state(user)['version'] == 1


def test_schema_api_contract_auth_status_rebuild_recall(user, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    monkeypatch.setattr(settings, 'internal_api_token', 'schema-contract-strong-internal-token')
    client = TestClient(app)
    url = f'/ai/memory/{user}/schema'
    assert client.get(url).status_code in {401, 403}
    headers = {'X-Internal-Token': settings.internal_api_token}
    assert client.get(url, headers=headers).json()['status'] == 'missing'
    seed(user)
    assert client.post(url + '/rebuild', headers=headers).json()['status'] == 'pending'
    assert builder.process_schema_builds(20)['processed'] > 0
    # Other tests may leave pending users, so ensure this user's queue is drained.
    publish(user)
    response = client.post(url + '/recall', headers=headers, json={'question': '我叫什么', 'allowedPaths': ['identity.name']})
    assert response.status_code == 200
    assert '小林' in response.json()['context']
    assert client.post(url + '/recall', headers=headers, json={'question': '我叫什么', 'mode': 'unsafe'}).status_code == 422
    monkeypatch.setattr(settings, 'memory_schema_enabled', False)
    assert client.post(url + '/rebuild', headers=headers).status_code == 409


def test_real_memory_agent_uses_published_schema(user):
    from app.agents.memory_agent import run_memory_agent
    seed(user, 'interaction.boundaries', '不要追问家庭')
    publish(user)
    result = run_memory_agent(user, 'conversation', '帮我解释这个概念')
    assert result['decision']['depth'] == 'light'
    assert '不要追问家庭' in result['memory_context']
    assert result['memory']['used_memories'][0]['reasonCode'] == 'ACTION_CONSTRAINT'
    assert run_memory_agent(user, 'conversation', '你好')['skipped']
