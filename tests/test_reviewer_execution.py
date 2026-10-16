import importlib
from types import SimpleNamespace

from app.agents.reviewer_agent import run_reviewer_agent

chain = importlib.import_module("app.chains.reviewer_chain")


def test_enabled_reviewer_without_configured_judge_reports_local_rules(monkeypatch):
    monkeypatch.setattr(chain, "is_judge_model_configured", lambda: False)
    review = run_reviewer_agent("解释 Markdown", "Markdown 是文本标记语法。", [], "", model_review_enabled=True)
    assert review["approved"] is True
    assert review["skipped"] is True
    assert review["modelReviewUsed"] is False


def test_configured_judge_is_called_once_and_reports_actual_use(monkeypatch):
    calls = []
    class Judge:
        def bind(self, **_):
            return self
        def invoke(self, messages):
            calls.append(messages)
            return SimpleNamespace(content='{"approved":true,"score":0.9}')
    monkeypatch.setattr(chain, "is_judge_model_configured", lambda: True)
    monkeypatch.setattr(chain, "build_judge_task_model", lambda: Judge())
    review = run_reviewer_agent("解释 Markdown", "Markdown 是文本标记语法。", [], "", model_review_enabled=True)
    assert len(calls) == 1
    assert review["skipped"] is False
    assert review["modelReviewUsed"] is True


def test_local_governance_block_never_calls_judge(monkeypatch):
    def unexpected(*_, **__):
        raise AssertionError("local failure must stop before a judge call")
    monkeypatch.setattr(chain, "is_judge_model_configured", unexpected)
    review = run_reviewer_agent("继续", "已遗忘的事实", [], "", memory={
        "governance": {"blockedTerms": [{"value": "已遗忘的事实", "status": "forgotten"}]},
    }, model_review_enabled=True)
    assert review["approved"] is False
    assert review["ruleBlocked"] is True
    assert review["modelReviewUsed"] is False
