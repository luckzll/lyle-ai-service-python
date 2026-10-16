from contextlib import contextmanager

import pytest

from app.projects import project_resolver
from app.projects.project_resolver import resolve_project, should_resolve_project


def test_resolve_project_prefers_explicit_project_id(monkeypatch):
    monkeypatch.setattr(
        "app.projects.project_resolver.get_project",
        lambda user_id, project_id: {"id": project_id, "name": "T2测试项目"},
    )
    linked = {}

    def fake_link(user_id, project_id, payload):
        linked.update({"user_id": user_id, "project_id": project_id, "payload": payload})
        return {"id": 1}

    monkeypatch.setattr("app.projects.project_resolver.create_project_link", fake_link)

    result = resolve_project("u1", "c1", "新增一个目标", project_id=9)

    assert result["projectId"] == 9
    assert result["projectName"] == "T2测试项目"
    assert result["confidence"] == 1.0
    assert result["requiresClarification"] is False
    assert linked["project_id"] == 9


def test_resolve_project_uses_conversation_link_for_followup(monkeypatch):
    monkeypatch.setattr(
        "app.projects.project_resolver.list_projects",
        lambda user_id, status="active", limit=50: [{"id": 9, "name": "V3测试项目", "slug": "v3-test"}],
    )
    monkeypatch.setattr(
        "app.projects.project_resolver.list_project_links",
        lambda user_id, project_id, target_type=None, limit=20: [{
            "projectId": project_id,
            "targetType": "conversation",
            "targetId": "c1",
            "confidence": 0.92,
        }],
    )

    result = resolve_project("u1", "c1", "帮我基于这个目标拆一个三步推进计划。")

    assert result["projectId"] == 9
    assert result["projectName"] == "V3测试项目"
    assert result["reason"] == "沿用当前会话已关联项目"


@pytest.mark.parametrize(("message", "prechecks"), [
    ("你好！", 0), ("hello", 0), ("谢谢", 0), ("你是谁？", 0),
    ("1+1等于几", 0), ("请帮我计算一下2 * (3 + 4)", 0),
    ("解释一下递归", 1), ("什么是机器学习？", 1), ("Explain HTTP caching", 1),
    ("联网搜索今天的天气", 1), ("帮我搜一下Python文档", 1),
    ("search the web for local restaurants", 1),
])
def test_independent_turns_only_precheck_possible_names_without_link_lookup(monkeypatch, message, prechecks):
    calls = []
    def forbidden(*args, **kwargs):
        raise AssertionError("independent turns must not query project links")

    def candidates(user_id, status, limit):
        calls.append((user_id, status, limit))
        return []
    monkeypatch.setattr(project_resolver.settings, "postgres_enabled", True)
    monkeypatch.setattr(project_resolver, "list_projects", candidates)
    monkeypatch.setattr(project_resolver, "_has_active_conversation_project_link", forbidden)

    assert not should_resolve_project(message, conversation_id="c1", user_id="u1")
    assert calls == [("u1", "active", 50)] * prechecks


@pytest.mark.parametrize("message", [
    "解释一下 Alpha 的架构", "联网搜索 star-bridge 的参考方案", "介绍一下 RAG 的工作原理",
])
def test_named_project_in_explanation_or_search_retains_resolved_context(monkeypatch, message):
    import importlib
    chat = importlib.import_module('app.api.chat')
    calls = []
    candidate = {"id": 9, "name": "Alpha", "slug": "star-bridge", "description": "RAG", "status": "active"}
    def candidates(user_id, status, limit):
        calls.append((user_id, status, limit))
        return [candidate]
    monkeypatch.setattr(project_resolver.settings, 'postgres_enabled', True)
    monkeypatch.setattr(project_resolver, 'list_projects', candidates)
    monkeypatch.setattr(project_resolver, '_has_active_conversation_project_link', lambda *_: pytest.fail('named projects need no link prequery'))
    monkeypatch.setattr(project_resolver, 'list_project_links', lambda *_, **__: pytest.fail('keyword match needs no link scan'))
    monkeypatch.setattr(project_resolver, 'create_project_link', lambda *args: {'id': 1})
    def context(user_id, project_id):
        assert (user_id, project_id) == ('u1', 9)
        return {'summary': 'Alpha scoped project context'}
    monkeypatch.setattr(chat, 'build_project_context', context)

    assert should_resolve_project(message, conversation_id='c1', user_id='u1')
    result = chat._project_context_for_response('u1', 'c1', message)

    assert result['resolver']['projectId'] == 9
    assert result['context']['summary'] == 'Alpha scoped project context'
    assert calls == [('u1', 'active', 50), ('u1', 'active', 50)]


def test_named_project_precheck_disabled_storage_never_queries(monkeypatch):
    monkeypatch.setattr(project_resolver.settings, 'postgres_enabled', False)
    monkeypatch.setattr(project_resolver, 'list_projects', lambda *_, **__: pytest.fail('disabled store'))
    assert not should_resolve_project('解释一下 Alpha 的架构', user_id='u1')


def test_named_project_precheck_keeps_regular_resolution_on_storage_error(monkeypatch):
    monkeypatch.setattr(project_resolver.settings, 'postgres_enabled', True)
    def unavailable(*args, **kwargs):
        raise ConnectionError('project metadata unavailable')
    monkeypatch.setattr(project_resolver, 'list_projects', unavailable)
    assert should_resolve_project('解释一下 Alpha 的架构', user_id='u1')


@pytest.mark.parametrize("message", [
    "解释一下这个项目", "联网搜索项目参考", "V3目标怎么推进", "Lyle有什么进展", "review the project roadmap",
])
def test_project_requests_keep_resolution_without_prequery(monkeypatch, message):
    monkeypatch.setattr(
        project_resolver, "_has_active_conversation_project_link",
        lambda *args: pytest.fail("explicit project intent needs no prequery"),
    )

    assert should_resolve_project(message, conversation_id="c1", user_id="u1")


def test_explicit_project_id_always_keeps_resolution():
    assert should_resolve_project("你好", project_id=9)
    assert should_resolve_project("", project_id=9)


@pytest.mark.parametrize("message", ["好的", "它呢？", "这个呢", "解释一下这个", "继续", "去同步一下"])
@pytest.mark.parametrize("linked", [False, True])
def test_contextual_followups_consult_conversation_link_once(monkeypatch, message, linked):
    calls = []

    def lookup(user_id, conversation_id):
        calls.append((user_id, conversation_id))
        return linked

    monkeypatch.setattr(project_resolver, "_has_active_conversation_project_link", lookup)

    assert should_resolve_project(message, conversation_id="c1", user_id="u1") is linked
    assert calls == [("u1", "c1")]


def test_unknown_project_name_remains_eligible_for_regular_resolution():
    assert should_resolve_project("Alpha的验收条件有哪些？", conversation_id="c1", user_id="u1")
    assert should_resolve_project("Alpha", conversation_id="c1", user_id="u1")


def test_followup_without_available_identity_is_conservative():
    assert should_resolve_project("它呢", conversation_id="c1")
    assert not should_resolve_project("", conversation_id="c1", user_id="u1")


@pytest.mark.parametrize("row, expected", [((1,), True), (None, False)])
def test_conversation_link_gate_uses_one_user_scoped_active_lookup(monkeypatch, row, expected):
    queries = []

    class Cursor:
        def fetchone(self):
            return row

    class Connection:
        def execute(self, sql, parameters):
            queries.append((" ".join(sql.split()), parameters))
            return Cursor()

    @contextmanager
    def connection():
        yield Connection()

    monkeypatch.setattr(project_resolver.settings, "postgres_enabled", True)
    monkeypatch.setattr(project_resolver, "postgres_connection", connection)

    assert should_resolve_project("它呢", conversation_id="c1", user_id="u1") is expected
    assert len(queries) == 1
    sql, parameters = queries[0]
    assert parameters == ("u1", "u1", "c1")
    assert "link.user_id = %s AND project.user_id = %s" in sql
    assert "project.user_id = link.user_id" in sql
    assert "project.status = 'active'" in sql
    assert "LIMIT 1" in sql


def test_conversation_link_gate_does_not_discard_context_on_storage_error(monkeypatch):
    def unavailable():
        raise ConnectionError("storage unavailable")

    monkeypatch.setattr(project_resolver.settings, "postgres_enabled", True)
    monkeypatch.setattr(project_resolver, "postgres_connection", unavailable)

    assert should_resolve_project("它呢", conversation_id="c1", user_id="u1")


@pytest.mark.parametrize("message", ["它呢", "这个呢", "好的", "去同步一下"])
def test_resolver_keeps_linked_short_followup_with_legacy_memory(monkeypatch, message):
    monkeypatch.setattr("app.memory.fact_store.enabled", lambda: False)
    monkeypatch.setattr(
        project_resolver, "list_projects",
        lambda *args, **kwargs: [{"id": 9, "name": "Alpha", "slug": "alpha"}],
    )
    monkeypatch.setattr(
        project_resolver, "list_project_links",
        lambda *args, **kwargs: [{"targetId": "c1", "confidence": 0.9}],
    )

    result = resolve_project("u1", "c1", message)

    assert result["projectId"] == 9
    assert result["requiresClarification"] is False
    assert result["reason"] == "沿用当前会话已关联项目"


def test_ambiguous_linked_followup_still_requires_project_clarification(monkeypatch):
    monkeypatch.setattr("app.memory.fact_store.enabled", lambda: True)
    monkeypatch.setattr(
        project_resolver, "list_projects",
        lambda *args, **kwargs: [{"id": 9, "name": "Alpha"}, {"id": 10, "name": "Beta"}],
    )
    monkeypatch.setattr(
        project_resolver, "list_project_links",
        lambda *args, **kwargs: [{"targetId": "c1", "confidence": 0.9}],
    )

    result = resolve_project("u1", "c1", "它呢")

    assert result["projectId"] is None
    assert result["requiresClarification"] is True
