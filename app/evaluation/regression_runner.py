from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evaluation.scorer import evaluate_answer


def run_regression_cases(path: str | Path = "tests/evaluation_cases.json") -> dict[str, object]:
    case_path = Path(path)
    cases = json.loads(case_path.read_text(encoding="utf-8"))
    items: list[dict[str, object]] = []
    passed = 0
    for index, case in enumerate(cases, start=1):
        result = evaluate_answer(
            question=str(case.get("question") or ""),
            answer=str(case.get("answer") or ""),
            contexts=case.get("contexts") or [],
            expected_answer=case.get("expected_answer"),
            must_include=case.get("must_include") or [],
            must_not_include=case.get("must_not_include") or [],
            case_type=case.get("case_type"),
            rubric=case.get("rubric"),
        )
        min_score = float(case.get("min_score") or 0.7)
        ok = float(result["score"]) >= min_score
        passed += 1 if ok else 0
        items.append({
            "index": index,
            "caseType": case.get("case_type") or "general",
            "question": case.get("question") or "",
            "score": result["score"],
            "minScore": min_score,
            "passed": ok,
            "reason": result.get("reason") or "",
        })
    total = len(items)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "passRate": round(passed / total, 4) if total else 0.0,
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Lyle V2 evaluation regression cases.")
    parser.add_argument("--cases", default="tests/evaluation_cases.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = run_regression_cases(args.cases)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(f"Evaluation regression: {result['passed']}/{result['total']} passed, passRate={result['passRate']}")
    for item in result["items"]:
        if not item["passed"]:
            print(f"- FAIL #{item['index']} {item['caseType']}: score={item['score']} min={item['minScore']} {item['question']}")


if __name__ == "__main__":
    main()
