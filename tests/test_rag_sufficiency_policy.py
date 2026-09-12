import json

import pytest
from langchain_core.runnables import RunnableLambda

from app.chains import rag_sufficiency_chain as sufficiency
from app.core.runtime_config import temporary_ai_config


CONTEXTS = [{"doc_id": "doc", "title": "版本", "chunk": "版本 2.4 支持逐段返回。"}]


@pytest.fixture(autouse=True)
def enable_optional_model_for_contract_tests():
    with temporary_ai_config({"RAG_AUXILIARY_MODEL_ENABLED": "true"}):
        yield


@pytest.mark.parametrize("raw", [{"sufficient": "false"}, {"sufficient": "true"}, {}, [], {"sufficient": 1}])
def test_non_boolean_model_decision_never_claims_verified_sufficiency(monkeypatch, raw):
    monkeypatch.setattr(sufficiency, "is_judge_model_configured", lambda: True)
    monkeypatch.setattr(sufficiency, "build_judge_task_model", lambda: RunnableLambda(lambda _: json.dumps(raw)))
    with temporary_ai_config({"RAG_SUFFICIENCY_CHECK_ENABLED": "true"}):
        result = sufficiency.judge_rag_sufficiency("下一版本的发布日期", CONTEXTS)
    assert result["method"] == "rule"


def test_valid_negative_model_decision_is_not_replaced_with_positive_rule(monkeypatch):
    monkeypatch.setattr(sufficiency, "is_judge_model_configured", lambda: True)
    monkeypatch.setattr(sufficiency, "build_judge_task_model", lambda: RunnableLambda(lambda _: '{"sufficient":false,"score":0.1}'))
    with temporary_ai_config({"RAG_SUFFICIENCY_CHECK_ENABLED": "true"}):
        result = sufficiency.judge_rag_sufficiency("版本支持什么", CONTEXTS)
    assert result["method"] == "model"
    assert result["sufficient"] is False


def test_disabled_sufficiency_is_marked_unchecked_without_model_call(monkeypatch):
    monkeypatch.setattr(sufficiency, "build_judge_task_model", lambda: pytest.fail("disabled must not call model"))
    with temporary_ai_config({"RAG_SUFFICIENCY_CHECK_ENABLED": "false"}):
        result = sufficiency.judge_rag_sufficiency("q", CONTEXTS)
    assert result["method"] == "disabled"


def test_auxiliary_failure_degrades_to_explicit_rule_decision(monkeypatch):
    def fail(_):
        raise TimeoutError("controlled model timeout")

    monkeypatch.setattr(sufficiency, "is_judge_model_configured", lambda: True)
    monkeypatch.setattr(sufficiency, "build_judge_task_model", lambda: RunnableLambda(fail))
    with temporary_ai_config({"RAG_SUFFICIENCY_CHECK_ENABLED": "true"}):
        result = sufficiency.judge_rag_sufficiency("版本支持什么", CONTEXTS)
    assert result["method"] == "rule"
    assert "timeout" not in json.dumps(result)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "invalid"])
def test_model_confidence_is_finite(value):
    assert sufficiency._safe_float(value, 0.5) == 0.5


def test_disabled_auxiliary_model_keeps_rule_check_without_network(monkeypatch):
    monkeypatch.setattr(sufficiency, "build_judge_task_model", lambda: pytest.fail("optional model must not be called"))
    with temporary_ai_config({"RAG_AUXILIARY_MODEL_ENABLED": "false", "RAG_SUFFICIENCY_CHECK_ENABLED": "true"}):
        assert sufficiency.judge_rag_sufficiency("版本支持什么", CONTEXTS)["method"] == "rule"


def test_query_rewrite_json_example_is_literal_and_model_is_actually_called(monkeypatch):
    from app.chains import query_rewrite_chain as rewrite
    prompts = []

    def model(prompt):
        prompts.append(prompt.to_string())
        return '{"queries":["版本改进内容"]}'

    monkeypatch.setattr(rewrite, "is_intent_model_configured", lambda: True)
    monkeypatch.setattr(rewrite, "build_intent_task_model", lambda: RunnableLambda(model))
    with temporary_ai_config({"RAG_QUERY_REWRITE_ENABLED": "true", "RAG_MULTI_QUERY_COUNT": "2"}):
        result = rewrite.rewrite_rag_queries("版本有什么变化")
    assert result == ["版本有什么变化", "版本改进内容"]
    assert len(prompts) == 1 and '{"queries":[' in prompts[0]
