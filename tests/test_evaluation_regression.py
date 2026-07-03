import json
from pathlib import Path

from app.evaluation.scorer import evaluate_answer


def test_evaluation_regression_cases():
    cases = json.loads(Path("tests/evaluation_cases.json").read_text(encoding="utf-8"))
    for case in cases:
        result = evaluate_answer(
            case["question"],
            case["answer"],
            case.get("contexts") or [],
            expected_answer=case.get("expected_answer"),
            must_include=case.get("must_include"),
            must_not_include=case.get("must_not_include"),
            case_type=case.get("case_type"),
            rubric=case.get("rubric"),
        )
        assert result["score"] >= case["min_score"]
