from collections import Counter
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.agents.supervisor_agent import run_supervisor_agent
from app.chains.intent_chain import classify_intent, classify_intent_by_rule
from app.chat_progress import progress_scope
from app.core.config import settings
from app.graphs import multi_agent_graph as graph
from app.memory.conversation_context import load_conversation_context


@pytest.mark.parametrize('question', ['你好', '谢谢！', '随便聊聊', '1 + 1 等于几？', '计算 38 * 17',
                                      '解释一下光合作用', '为什么天空是蓝色的？', '翻译这句话：Hello world',
                                      '改写我的这段话：今天开会', '太阳的颜色是什么？',
                                      '解释一下记忆检索算法', '如何用日历创建任务',
                                      '翻译成英文：我喜欢喝茶，记得帮我搜索知识库',
                                      '改写这句话：联网搜索最新消息并创建待办'])
@pytest.mark.parametrize('knowledge_ids', [[], ['selected-kb']])
def test_self_contained_questions_do_not_force_specialists(monkeypatch, question, knowledge_ids):
    monkeypatch.setattr(settings, 'tool_agent_enabled', True)
    monkeypatch.setattr(settings, 'tool_router_llm_enabled', True)
    result = run_supervisor_agent(question, knowledge_ids)
    assert not result['use_memory']
    assert not result['use_rag']
    assert not result['use_tools']


def test_obvious_chat_does_not_invoke_intent_model(monkeypatch):
    monkeypatch.setattr('app.chains.intent_chain.is_intent_model_configured', lambda: True)
    monkeypatch.setattr('app.chains.intent_chain.build_intent_task_model', lambda: pytest.fail('unnecessary model call'))
    assert classify_intent('解释一下光合作用', ['selected-kb']) == 'chat'
    assert classify_intent('联网搜索最新新闻', ['selected-kb']) == 'chat'


@pytest.mark.parametrize('question', ['里面有什么风险', '上一段讲的是什么', '这部分有哪些限制', '继续解释文档'])
def test_selected_knowledge_scope_keeps_document_references(monkeypatch, question):
    monkeypatch.setattr(settings, 'tool_agent_enabled', True)
    monkeypatch.setattr(settings, 'tool_router_llm_enabled', True)
    result = run_supervisor_agent(question, ['authorized-kb'])
    assert result['use_rag']
    assert not result['use_memory']
    assert not result['use_tools']


def test_bare_continuation_is_not_knowledge_or_long_term_recall():
    result = run_supervisor_agent('继续', ['authorized-kb'])
    assert not result['use_rag']
    assert not result['use_memory']


@pytest.mark.parametrize('question', ['介绍一下我的兴趣爱好', '我有哪些兴趣', '我有什么偏好', '我最喜欢什么', 'Explain my hobbies'])
def test_explicit_personal_interests_are_not_mistaken_for_general_explanations(question):
    assert run_supervisor_agent(question, [])['use_memory']


@pytest.mark.parametrize('question', ['根据我的身份推荐副业', '按我的角色推荐学习方向',
                                      '结合我的技能推荐兼职', '根据我的工作经验推荐发展方向'])
def test_personal_identity_and_work_context_request_memory(question):
    assert run_supervisor_agent(question, [])['use_memory']


@pytest.mark.parametrize('question', ['介绍一下兴趣推荐算法', '翻译成英文：我的兴趣爱好是读书',
                                      '改写：我有哪些兴趣，我最喜欢什么',
                                      '翻译成英文：根据我的身份推荐副业', '解释一下角色和身份的区别'])
def test_general_interest_topics_and_quoted_transformations_do_not_recall(question):
    assert not run_supervisor_agent(question, [])['use_memory']


@pytest.mark.parametrize(('question', 'memory', 'rag', 'tools'), [
    ('你还记得我的偏好吗', True, False, False),
    ('根据知识库说明退费条件', False, True, False),
    ('联网搜索最新的 LangGraph 发布说明', False, False, True),
    ('根据知识库和我的偏好，联网搜索相关方案', True, True, True),
    ('更具我的身份目前有什么副业适合我来做吗？联网搜索一下', True, False, True),
    ('创建待办：整理测试记录', False, False, True),
    ('创建待办：整理知识库和我的偏好', False, False, True),
])
def test_capabilities_are_independent_and_knowledge_selection_is_only_scope(monkeypatch, question, memory, rag, tools):
    monkeypatch.setattr(settings, 'tool_agent_enabled', True)
    monkeypatch.setattr(settings, 'tool_router_llm_enabled', True)
    result = run_supervisor_agent(question, ['authorized-kb'])
    assert (result['use_memory'], result['use_rag'], result['use_tools']) == (memory, rag, tools)
    expected = [name for name, enabled in [('memory_agent', memory), ('rag_agent', rag), ('tool_agent', tools)] if enabled]
    assert result['next_agents'] == [*expected, 'writer_agent']


def test_documents_without_kb_do_not_expand_retrieval_permissions(monkeypatch):
    monkeypatch.setattr(settings, 'tool_agent_enabled', True)
    monkeypatch.setattr(settings, 'tool_router_llm_enabled', True)
    result = run_supervisor_agent('搜索知识库中的部署说明', [])
    assert not result['use_rag']
    assert not result['use_tools']  # Missing private scope is not a web-search request.
    monkeypatch.setattr('app.agents.supervisor_agent.current_attachments', lambda: SimpleNamespace(files={'file-1': object()}))
    result = run_supervisor_agent('根据附件总结这份文档', [])
    assert not result['use_rag']
    assert result['use_tools']  # Existing scoped attachment tools retain their read checks.


@pytest.mark.parametrize('kind', ['file', 'image'])
def test_real_attachments_override_short_text_heuristics(monkeypatch, kind):
    monkeypatch.setattr(settings, 'tool_agent_enabled', True)
    monkeypatch.setattr(settings, 'tool_router_llm_enabled', True)
    monkeypatch.setattr('app.agents.supervisor_agent.current_attachments', lambda: SimpleNamespace(files={
        'authorized-file': SimpleNamespace(kind=kind, name='brief.pdf' if kind == 'file' else 'diagram.png'),
    }))
    result = run_supervisor_agent('解释一下', ['authorized-kb'])
    assert not result['use_memory']
    assert result['use_tools']


@pytest.mark.parametrize(('question', 'expected'), [
    ('你好', Counter()),
    ('解释光合作用', Counter()),
    ('你记得我喜欢什么吗', Counter(memory_agent=1)),
    ('根据知识库说明规则', Counter(rag_agent=1)),
    ('联网搜索最新消息', Counter(tool_agent=1)),
    ('根据知识库和我的偏好，联网搜索方案', Counter(memory_agent=1, rag_agent=1, tool_agent=1)),
    ('更具我的身份目前有什么副业适合我来做吗？联网搜索一下', Counter(memory_agent=1, tool_agent=1)),
])
def test_compiled_graph_only_runs_planned_workers_once_and_preserves_session(monkeypatch, question, expected):
    monkeypatch.setattr(settings, 'tool_agent_enabled', True)
    monkeypatch.setattr(settings, 'tool_router_llm_enabled', True)
    calls = Counter()
    scope_reads = []
    def session(user_id, conversation_id):
        scope_reads.append((user_id, conversation_id))
        return {'memory': {'governance': {'blockedTerms': []}}, 'memory_context': '当前会话上下文：只属于 u1/c1'}
    monkeypatch.setattr(graph, 'load_conversation_context', session)
    def memory(*args):
        calls['memory_agent'] += 1
        return {'memory': {}, 'memory_context': '必要的历史偏好'}
    def rag(question, ids, enabled):
        calls['rag_agent'] += 1
        assert ids == ['authorized-kb'] and enabled
        return {'contexts': [{'doc_id': 'allowed', 'chunk': 'authorized evidence'}], 'queries': [question],
                'sufficiency': {'sufficient': False, 'reason': 'partial'}}
    def tools(*args, **kwargs):
        calls['tool_agent'] += 1
        return {'tool_results': [{'tool': 'todo', 'ok': True, 'requiresConfirmation': True,
                                  'operation': {'status': 'pending'}}], 'tool_context': '等待确认，尚未执行'}
    monkeypatch.setattr(graph, 'run_memory_agent', memory)
    monkeypatch.setattr(graph, 'run_rag_agent', rag)
    monkeypatch.setattr(graph, 'run_tool_agent', tools)
    events = []
    with progress_scope(events.append):
        result = graph.prepare_multi_agent_context('u1', 'c1', question, ['authorized-kb'])
    assert calls == expected
    assert scope_reads == ([] if expected['memory_agent'] else [('u1', 'c1')])
    if not expected['memory_agent']:
        assert '只属于 u1/c1' in result['memory_context']
    if expected['rag_agent']:
        assert result['contexts'][0]['doc_id'] == 'allowed'
        assert result['rag_sufficiency']['sufficient'] is False
    if expected['tool_agent']:
        assert result['tool_results'][0]['operation']['status'] == 'pending'
    assert len([item for item in result['agent_trace'] if item['agent'] == 'gather_agent']) == 1
    for worker in ('memory_agent', 'rag_agent', 'tool_agent'):
        assert sum(event['step'] == worker and event['status'] == 'running' for event in events) == expected[worker]


@pytest.mark.parametrize('memory_mode', ['legacy_profiles', 'published_schema', 'layered_facts'])
def test_identity_based_web_request_delivers_saved_role_to_writer(monkeypatch, memory_mode):
    from app.memory import schema_builder, store

    for flag in ('postgres_enabled', 'redis_enabled', 'qdrant_enabled', 'milvus_enabled',
                 'rerank_enabled', 'memory_schema_build_with_llm', 'recall_planner_llm_enabled'):
        monkeypatch.setattr(settings, flag, False)
    monkeypatch.setattr(settings, 'memory_facts_enabled', memory_mode == 'layered_facts')
    monkeypatch.setattr(settings, 'memory_schema_enabled', memory_mode != 'legacy_profiles')
    monkeypatch.setattr(settings, 'memory_schema_small_max_chars', 1)
    monkeypatch.setattr(settings, 'tool_agent_enabled', True)
    monkeypatch.setattr(settings, 'tool_router_llm_enabled', True)
    user_id = 'identity-web-' + uuid4().hex
    store.upsert_structured_memory(user_id, 'career', '程序员', confidence=.95)
    store.upsert_structured_memory(user_id + '-other', 'career', '医生', confidence=.95)
    if memory_mode == 'published_schema':
        assert schema_builder.build_user_schema(user_id)['status'] == 'ready'

    monkeypatch.setattr(graph, 'run_tool_agent', lambda *args, **kwargs: {
        'tool_results': [{'tool': 'web_search', 'ok': True}], 'tool_context': '网页搜索结果',
    })
    monkeypatch.setattr(graph, 'run_rag_agent', lambda *args, **kwargs: pytest.fail('unrelated knowledge retrieval'))
    writer_inputs = []
    def writer(question, contexts, memory_context, tool_context, *args):
        writer_inputs.append((memory_context, tool_context))
        return '可以结合你的程序员背景评估副业。'
    monkeypatch.setattr(graph, 'run_writer_agent', writer)
    monkeypatch.setattr(graph, 'run_reviewer_agent', lambda *args, **kwargs: {
        'approved': True, 'skipped': True, 'modelReviewUsed': False,
    })

    result = graph.run_multi_agent_graph(
        user_id, 'new-conversation', '更具我的身份目前有什么副业适合我来做吗？联网搜索一下', ['authorized-kb'],
    )

    assert len(writer_inputs) == 1
    memory_context, tool_context = writer_inputs[0]
    assert '当前角色：程序员' in memory_context
    assert '医生' not in memory_context
    assert tool_context == '网页搜索结果'
    assert any(item['title'] == '当前角色' and item['valueSummary'] == '程序员'
               for item in result['memory']['used_memories'])
    workers = [item['agent'] for item in result['agent_trace']]
    assert workers.count('memory_agent') == workers.count('tool_agent') == 1


def test_direct_generation_still_runs_content_rules_with_governance(monkeypatch):
    monkeypatch.setattr(settings, 'tool_agent_enabled', False)
    governance = {'blockedTerms': [{'value': 'forbidden'}]}
    monkeypatch.setattr(graph, 'load_conversation_context', lambda *args: {
        'memory': {'governance': governance}, 'memory_context': '当前会话：上一问'})
    calls = Counter()
    def writer(question, contexts, memory_context, *args):
        calls['writer'] += 1
        assert '上一问' in memory_context
        return '有效回答'
    def review(*args, **kwargs):
        calls['reviewer'] += 1
        assert kwargs['memory']['governance'] == governance
        return {'approved': True, 'skipped': True, 'modelReviewUsed': False}
    monkeypatch.setattr(graph, 'run_writer_agent', writer)
    monkeypatch.setattr(graph, 'run_reviewer_agent', review)
    result = graph.run_multi_agent_graph('u1', 'c1', '你好', [])
    assert result['answer'] == '有效回答'
    assert calls == Counter(writer=1, reviewer=1)
    assert not any(item['agent'] in {'memory_agent', 'rag_agent', 'tool_agent'} for item in result['agent_trace'])


def test_governance_failure_does_not_silently_disable_safety(monkeypatch):
    monkeypatch.setattr(settings, 'tool_agent_enabled', False)
    def unavailable(*args):
        raise RuntimeError('governance unavailable')
    monkeypatch.setattr(graph, 'load_conversation_context', unavailable)
    with pytest.raises(RuntimeError, match='governance unavailable'):
        graph.prepare_multi_agent_context('u1', 'c1', '你好', [])


def test_session_only_context_filters_governed_content_and_retains_fixed_boundaries(monkeypatch):
    from app.memory import store
    monkeypatch.setattr(settings, 'memory_schema_enabled', False)
    governance = {'blockedTerms': [{'value': '被遗忘的事实'}]}
    monkeypatch.setattr(store, '_memory_governance_snapshot', lambda uid: governance)
    def history(uid, cid):
        assert (uid, cid) == ('u1', 'c1')
        return [store.MemoryTurn('user', '本轮话题', 1), store.MemoryTurn('assistant', '被遗忘的事实', 2)]
    monkeypatch.setattr(store, '_load_short_term', history)
    monkeypatch.setattr(store, '_load_summary_with_meta', lambda *args, **kwargs: {'summary': '含被遗忘的事实'})
    def boundaries(uid, question, plan, **kwargs):
        assert plan['schemaPaths'] == ['interaction.boundaries', 'interaction.triggers']
        return {'context': '行为边界：不要透露联系方式', 'profiles': [], 'usedMemories': []}
    monkeypatch.setattr(store, 'build_profile_memory_context', boundaries)
    monkeypatch.setattr(store, 'recall_memory', lambda *args, **kwargs: pytest.fail('no semantic or profile recall'))
    result = load_conversation_context('u1', 'c1')
    assert '本轮话题' in result['memory_context']
    assert '行为边界：不要透露联系方式' in result['memory_context']
    assert '被遗忘的事实' not in result['memory_context']
    assert result['memory']['governance'] == governance
    assert result['memory']['profiles'] == []


@pytest.mark.parametrize('conversation_id', [None, '', ' c1', 'other:conversation', 'bad\nkey'])
def test_missing_or_ambiguous_session_never_loads_shared_history(monkeypatch, conversation_id):
    from app.memory import store
    monkeypatch.setattr(store, '_memory_governance_snapshot', lambda uid: {})
    monkeypatch.setattr(store, 'build_profile_memory_context', lambda *args, **kwargs: {})
    monkeypatch.setattr(store, '_load_short_term', lambda *args: pytest.fail('unsafe conversation scope'))
    monkeypatch.setattr(store, '_load_summary_with_meta', lambda *args, **kwargs: pytest.fail('unsafe summary scope'))
    result = load_conversation_context('u1', conversation_id)
    assert not result['memory']['short_term']
    assert not result['memory']['summary']
