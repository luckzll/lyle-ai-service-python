import json
from pathlib import Path

from app.evaluation.scorer import evaluate_answer


def test_evaluation_regression_cases():
    cases = json.loads(Path("tests/evaluation_cases.json").read_text(encoding="utf-8"))
    for case in cases:
        result = evaluate_answer(case["question"], case["answer"], case["contexts"])
        assert result["score"] >= case["min_score"]
