import pytest

from app.streaming_text import streaming_policy


def policy(**overrides):
    inputs = dict(
        reviewer_enabled=False, contexts=[], memory={}, sufficiency={}, tool_results=[],
        project_context={}, candidate_goals=[], candidate_plans=[], memory_statement=False,
        memory_overview=False, has_attachments=False,
    )
    return streaming_policy(**{**inputs, **overrides})


@pytest.mark.parametrize("project_context", [
    {},
    {"resolver": {}},
    {"resolver": {"projectId": None, "requiresClarification": False}, "context": {}},
    {"resolver": {"projectId": None, "reason": "未命中项目关键词", "candidates": []}, "context": {}},
    {"resolver": {"projectId": None, "reason": "project_store_unavailable:ConnectionError"}, "context": {}},
])
def test_no_match_project_envelope_does_not_buffer_ordinary_chat(project_context):
    assert policy(project_context=project_context) == "incremental"


@pytest.mark.parametrize("project_context", [
    {"resolver": {"projectId": 9}, "context": {}},
    {"resolver": {"projectId": None, "requiresClarification": True}, "context": {}},
    {"resolver": {}, "context": {"summary": "Project facts"}},
    {"resolver": {"projectId": 9}, "context": {"error": "unavailable"}},
])
def test_actual_project_dependencies_remain_buffered(project_context):
    assert policy(project_context=project_context) == "tool_or_project_review"


@pytest.mark.parametrize("dependency, expected", [
    ({"reviewer_enabled": True}, "full_model_review"),
    ({"contexts": [{"chunk": "evidence"}]}, "rag_grounding_review"),
    ({"sufficiency": {"sufficient": False}}, "rag_grounding_review"),
    ({"memory": {"governance": {"blockedTerms": [{"value": "forgotten"}]}}}, "memory_governance_review"),
    ({"memory_statement": True}, "memory_confirmation_review"),
    ({"tool_results": [{"ok": False}]}, "tool_or_project_review"),
    ({"candidate_goals": [{"title": "Goal"}]}, "tool_or_project_review"),
    ({"candidate_plans": [{"title": "Plan"}]}, "tool_or_project_review"),
    ({"has_attachments": True}, "tool_or_project_review"),
])
def test_no_match_project_does_not_bypass_other_content_gates(dependency, expected):
    assert policy(project_context={"resolver": {"projectId": None}, "context": {}}, **dependency) == expected
