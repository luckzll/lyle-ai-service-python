from app.core.config import settings
from app.evaluation import scorer
from app.evaluation.scorer import evaluate_answer
from app.api.evaluation import _build_evaluation_report


def test_evaluate_answer_scores_non_empty_answer():
    result = evaluate_answer(
        "海蓝星是什么？",
        "海蓝星是一颗被海洋覆盖的虚构星球。",
        [{"title": "doc", "chunk": "海蓝星是一颗被海洋覆盖的虚构星球。"}],
    )

    assert result["score"] > 0
    assert result["checks"]["has_sources"] is True
    assert result["scoringMode"] == "heuristic"
    assert result["heuristicScore"] == result["score"]
    assert "dimensions" in result
    assert "groundedness" in result["dimensions"]


def test_evaluate_answer_uses_expected_answer_and_keywords():
    result = evaluate_answer(
        "海蓝星是什么？",
        "海蓝星是一颗虚构星球。",
        [],
        expected_answer="海蓝星是一颗被海洋覆盖的虚构星球。",
        must_include=["海洋覆盖"],
        must_not_include=["真实存在"],
    )

    assert result["score"] < 0.8
    assert "海洋覆盖" in result["checks"]["missing_keywords"]


def test_evaluate_answer_can_use_judge_score(monkeypatch):
    monkeypatch.setattr(settings, "evaluation_judge_enabled", True)
    monkeypatch.setattr(scorer, "is_judge_model_configured", lambda: True)
    monkeypatch.setattr(scorer, "_judge_evaluation", lambda **kwargs: {
        "enabled": True,
        "ok": True,
        "score": 1.0,
        "dimensions": {
            "relevance": 1,
            "accuracy": 1,
            "completeness": 1,
            "groundedness": 1,
            "safety": 1,
        },
        "reason": "裁判通过",
    })

    result = evaluate_answer("问题", "这是一个完整回答。", [])

    assert result["scoringMode"] == "hybrid"
    assert result["judgeScore"] == 1.0
    assert result["score"] >= result["heuristicScore"]


def test_build_evaluation_report_contains_summary_and_cases():
    markdown = _build_evaluation_report({
        "id": 7,
        "name": "smoke-eval",
        "averageScore": 0.82,
        "count": 1,
        "badcaseCount": 0,
        "promptVersion": "lyle-agent-v1",
        "modelProvider": "deepseek",
        "modelName": "deepseek-v4-flash",
        "createdAt": "2026-07-01T00:00:00",
        "items": [
            {
                "id": "case-1",
                "question": "接口是否清楚？",
                "answer": "清楚。",
                "score": 0.82,
                "checks": {"has_sources": True},
                "badcase": False,
            }
        ],
    })

    assert "# 评测报告：smoke-eval" in markdown
    assert "- 平均分: 0.82" in markdown
    assert "### case-1 · 0.82 · OK" in markdown
    assert "接口是否清楚？" in markdown
