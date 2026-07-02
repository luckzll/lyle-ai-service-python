from fastapi import APIRouter
from pydantic import BaseModel, Field
from psycopg.types.json import Jsonb

from app.core.config import settings
from app.core.database import postgres_connection
from app.evaluation.scorer import evaluate_answer

router = APIRouter()


class EvaluationRequest(BaseModel):
    question: str
    answer: str
    contexts: list[dict[str, str]] = Field(default_factory=list)
    expected_answer: str | None = Field(default=None, alias="expectedAnswer")
    must_include: list[str] = Field(default_factory=list, alias="mustInclude")
    must_not_include: list[str] = Field(default_factory=list, alias="mustNotInclude")
    type: str | None = None
    rubric: str | None = None


class BatchEvaluationCase(BaseModel):
    id: str | None = None
    question: str
    answer: str
    contexts: list[dict[str, str]] = Field(default_factory=list)
    expected_answer: str | None = Field(default=None, alias="expectedAnswer")
    must_include: list[str] = Field(default_factory=list, alias="mustInclude")
    must_not_include: list[str] = Field(default_factory=list, alias="mustNotInclude")
    type: str | None = None
    rubric: str | None = None


class BatchEvaluationRequest(BaseModel):
    cases: list[BatchEvaluationCase]
    name: str | None = None
    evaluation_set_id: int | None = Field(default=None, alias="evaluationSetId")


class EvaluationSetRequest(BaseModel):
    name: str
    description: str = ""
    cases: list[BatchEvaluationCase]


@router.post("/evaluate")
def evaluate(request: EvaluationRequest) -> dict[str, object]:
    return evaluate_answer(
        request.question,
        request.answer,
        request.contexts,
        expected_answer=request.expected_answer,
        must_include=request.must_include,
        must_not_include=request.must_not_include,
        case_type=request.type,
        rubric=request.rubric,
    )


@router.post("/evaluate/batch")
def evaluate_batch(request: BatchEvaluationRequest) -> dict[str, object]:
    payload = _evaluate_cases(request.cases)
    payload["evaluationSetId"] = request.evaluation_set_id
    run_id = _save_evaluation_run(request.name or "手动批量评测", payload, request.evaluation_set_id)
    if run_id is not None:
        payload["runId"] = run_id
    return payload


@router.get("/evaluate/sets")
def evaluation_sets(limit: int = 20) -> dict[str, object]:
    if not settings.postgres_enabled:
        return {"items": []}
    with postgres_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, name, description, jsonb_array_length(cases), created_at, updated_at
            FROM evaluation_set
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            (max(1, min(limit, 100)),),
        ).fetchall()
    return {
        "items": [
            {
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "caseCount": row[3],
                "createdAt": row[4].isoformat(),
                "updatedAt": row[5].isoformat(),
            }
            for row in rows
        ]
    }


@router.post("/evaluate/sets")
def save_evaluation_set(request: EvaluationSetRequest) -> dict[str, object]:
    if not settings.postgres_enabled:
        return {"ok": False, "message": "PostgreSQL is not enabled."}
    cases = [_case_to_dict(item, index) for index, item in enumerate(request.cases, start=1)]
    with postgres_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO evaluation_set (name, description, cases, updated_at)
            VALUES (%s, %s, %s, now())
            RETURNING id, name, description, jsonb_array_length(cases), created_at, updated_at
            """,
            (request.name[:256], request.description, Jsonb(cases)),
        ).fetchone()
        conn.commit()
    return {
        "id": row[0],
        "name": row[1],
        "description": row[2],
        "caseCount": row[3],
        "createdAt": row[4].isoformat(),
        "updatedAt": row[5].isoformat(),
    }


@router.put("/evaluate/sets/{set_id}")
def update_evaluation_set(set_id: int, request: EvaluationSetRequest) -> dict[str, object]:
    if not settings.postgres_enabled:
        return {"ok": False, "message": "PostgreSQL is not enabled."}
    cases = [_case_to_dict(item, index) for index, item in enumerate(request.cases, start=1)]
    with postgres_connection() as conn:
        row = conn.execute(
            """
            UPDATE evaluation_set
            SET name = %s, description = %s, cases = %s, updated_at = now()
            WHERE id = %s
            RETURNING id, name, description, jsonb_array_length(cases), created_at, updated_at
            """,
            (request.name[:256], request.description, Jsonb(cases), set_id),
        ).fetchone()
        conn.commit()
    if row is None:
        return {}
    return {
        "id": row[0],
        "name": row[1],
        "description": row[2],
        "caseCount": row[3],
        "createdAt": row[4].isoformat(),
        "updatedAt": row[5].isoformat(),
    }


@router.delete("/evaluate/sets/{set_id}")
def delete_evaluation_set(set_id: int) -> dict[str, object]:
    if not settings.postgres_enabled:
        return {"deleted": False}
    with postgres_connection() as conn:
        row = conn.execute(
            "DELETE FROM evaluation_set WHERE id = %s RETURNING id",
            (set_id,),
        ).fetchone()
        conn.commit()
    return {"deleted": row is not None, "id": set_id}


@router.get("/evaluate/sets/{set_id}")
def evaluation_set_detail(set_id: int) -> dict[str, object]:
    row = _load_evaluation_set_row(set_id)
    return {
        "id": row[0],
        "name": row[1],
        "description": row[2],
        "cases": row[3],
        "createdAt": row[4].isoformat(),
        "updatedAt": row[5].isoformat(),
    }


@router.post("/evaluate/sets/{set_id}/run")
def run_evaluation_set(set_id: int, badcasesOnly: bool = False) -> dict[str, object]:
    row = _load_evaluation_set_row(set_id)
    cases = [_dict_to_case(item) for item in row[3]]
    if badcasesOnly:
        previous_badcase_ids = set(_load_latest_badcase_ids(set_id))
        cases = [item for item in cases if (item.id or "") in previous_badcase_ids]
    payload = _evaluate_cases(cases)
    payload["evaluationSetId"] = set_id
    payload["badcasesOnly"] = badcasesOnly
    run_id = _save_evaluation_run(
        f"{row[1]}{' badcase 复测' if badcasesOnly else ' 全量评测'}",
        payload,
        set_id,
    )
    if run_id is not None:
        payload["runId"] = run_id
    return payload


@router.get("/evaluate/runs/{run_id}")
def evaluation_run_detail(run_id: int) -> dict[str, object]:
    if not settings.postgres_enabled:
        return {}
    with postgres_connection() as conn:
        row = conn.execute(
            """
            SELECT id, name, count, average_score, prompt_version, model_provider, model_name,
                   items, evaluation_set_id, badcase_count, created_at
            FROM evaluation_run
            WHERE id = %s
            """,
            (run_id,),
        ).fetchone()
    if row is None:
        return {}
    payload = {
        "id": row[0],
        "name": row[1],
        "count": row[2],
        "averageScore": row[3],
        "promptVersion": row[4],
        "modelProvider": row[5],
        "modelName": row[6],
        "items": row[7],
        "evaluationSetId": row[8],
        "badcaseCount": row[9],
        "createdAt": row[10].isoformat(),
    }
    return payload


@router.get("/evaluate/runs/{run_id}/report")
def evaluation_run_report(run_id: int) -> dict[str, object]:
    detail = evaluation_run_detail(run_id)
    if not detail:
        return {}
    markdown = _build_evaluation_report(detail)
    return {
        "runId": run_id,
        "fileName": f"evaluation-run-{run_id}.md",
        "contentType": "text/markdown",
        "markdown": markdown,
    }


@router.get("/evaluate/runs")
def evaluation_runs(limit: int = 20) -> dict[str, object]:
    if not settings.postgres_enabled:
        return {"items": []}
    with postgres_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, name, count, average_score, prompt_version, model_provider, model_name,
                   evaluation_set_id, badcase_count, created_at
            FROM evaluation_run
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (max(1, min(limit, 100)),),
        ).fetchall()
    return {
        "items": [
            {
                "id": row[0],
                "name": row[1],
                "count": row[2],
                "averageScore": row[3],
                "promptVersion": row[4],
                "modelProvider": row[5],
                "modelName": row[6],
                "evaluationSetId": row[7],
                "badcaseCount": row[8],
                "createdAt": row[9].isoformat(),
            }
            for row in rows
        ]
    }


@router.get("/evaluate/trends")
def evaluation_trends(limit: int = 50) -> dict[str, object]:
    if not settings.postgres_enabled:
        return {"points": [], "promptCompare": [], "modelCompare": []}
    limit = max(1, min(limit, 200))
    with postgres_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, name, average_score, prompt_version, model_provider, model_name, badcase_count, created_at
            FROM evaluation_run
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,),
        ).fetchall()
    points = [
        {
            "id": row[0],
            "name": row[1],
            "averageScore": row[2],
            "promptVersion": row[3],
            "modelProvider": row[4],
            "modelName": row[5],
            "badcaseCount": row[6],
            "createdAt": row[7].isoformat(),
        }
        for row in reversed(rows)
    ]
    return {
        "points": points,
        "promptCompare": _compare_by(points, "promptVersion"),
        "modelCompare": _compare_by(points, "modelName"),
    }


def _compare_by(points: list[dict[str, object]], key: str) -> list[dict[str, object]]:
    grouped: dict[str, list[float]] = {}
    badcases: dict[str, int] = {}
    for item in points:
        name = str(item.get(key) or "unknown")
        grouped.setdefault(name, []).append(float(item.get("averageScore") or 0))
        badcases[name] = badcases.get(name, 0) + int(item.get("badcaseCount") or 0)
    return [
        {
            "name": name,
            "count": len(scores),
            "averageScore": round(sum(scores) / len(scores), 3) if scores else 0,
            "badcaseCount": badcases.get(name, 0),
        }
        for name, scores in sorted(grouped.items(), key=lambda pair: sum(pair[1]) / len(pair[1]) if pair[1] else 0, reverse=True)
    ]


def _save_evaluation_run(name: str, payload: dict[str, object], evaluation_set_id: int | None = None) -> int | None:
    if not settings.postgres_enabled:
        return None
    try:
        items = list(payload.get("items") or [])
        badcase_count = len([item for item in items if float(item.get("score") or 0) < 0.7])
        with postgres_connection() as conn:
            row = conn.execute(
                """
                INSERT INTO evaluation_run (
                  name, count, average_score, prompt_version, model_provider, model_name,
                  items, evaluation_set_id, badcase_count
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    name[:256],
                    payload["count"],
                    payload["averageScore"],
                    settings.prompt_version,
                    settings.chat_model_provider,
                    settings.chat_model_name,
                    Jsonb(payload["items"]),
                    evaluation_set_id,
                    badcase_count,
                ),
            ).fetchone()
            conn.commit()
        return int(row[0]) if row else None
    except Exception:
        return None


def _evaluate_cases(cases: list[BatchEvaluationCase]) -> dict[str, object]:
    items = []
    for index, item in enumerate(cases, start=1):
        result = evaluate_answer(
            item.question,
            item.answer,
            item.contexts,
            expected_answer=item.expected_answer,
            must_include=item.must_include,
            must_not_include=item.must_not_include,
            case_type=item.type,
            rubric=item.rubric,
        )
        items.append({
            "id": item.id or str(index),
            "question": item.question,
            "answer": item.answer,
            "score": result["score"],
            "checks": result["checks"],
            "dimensions": result.get("dimensions"),
            "reason": result.get("reason"),
            "scoringMode": result.get("scoringMode"),
            "heuristicScore": result.get("heuristicScore"),
            "judgeScore": result.get("judgeScore"),
            "promptVersion": result["prompt_version"],
            "badcase": float(result["score"]) < 0.7,
        })
    avg_score = round(sum(float(item["score"]) for item in items) / len(items), 3) if items else 0.0
    return {
        "count": len(items),
        "averageScore": avg_score,
        "badcaseCount": len([item for item in items if item["badcase"]]),
        "items": items,
    }


def _case_to_dict(item: BatchEvaluationCase, index: int) -> dict[str, object]:
    return {
        "id": item.id or str(index),
        "question": item.question,
        "answer": item.answer,
        "contexts": item.contexts,
        "expectedAnswer": item.expected_answer,
        "mustInclude": item.must_include,
        "mustNotInclude": item.must_not_include,
        "type": item.type,
        "rubric": item.rubric,
    }


def _dict_to_case(item: dict[str, object]) -> BatchEvaluationCase:
    return BatchEvaluationCase(
        id=str(item.get("id") or ""),
        question=str(item.get("question") or ""),
        answer=str(item.get("answer") or ""),
        contexts=list(item.get("contexts") or []),
        expectedAnswer=str(item.get("expectedAnswer") or item.get("expected_answer") or "") or None,
        mustInclude=list(item.get("mustInclude") or item.get("must_include") or []),
        mustNotInclude=list(item.get("mustNotInclude") or item.get("must_not_include") or []),
        type=str(item.get("type") or "") or None,
        rubric=str(item.get("rubric") or "") or None,
    )


def _load_evaluation_set_row(set_id: int):
    if not settings.postgres_enabled:
        raise ValueError("PostgreSQL is not enabled.")
    with postgres_connection() as conn:
        row = conn.execute(
            """
            SELECT id, name, description, cases, created_at, updated_at
            FROM evaluation_set
            WHERE id = %s
            """,
            (set_id,),
        ).fetchone()
    if row is None:
        raise ValueError("Evaluation set not found.")
    return row


def _load_latest_badcase_ids(set_id: int) -> list[str]:
    with postgres_connection() as conn:
        row = conn.execute(
            """
            SELECT items
            FROM evaluation_run
            WHERE evaluation_set_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (set_id,),
        ).fetchone()
    if row is None:
        return []
    return [
        str(item.get("id"))
        for item in row[0]
        if float(item.get("score") or 0) < 0.7
    ]


def _build_evaluation_report(detail: dict[str, object]) -> str:
    items = list(detail.get("items") or [])
    lines = [
        f"# 评测报告：{detail.get('name') or detail.get('id')}",
        "",
        "## 概览",
        "",
        f"- Run ID: {detail.get('id')}",
        f"- 平均分: {detail.get('averageScore')}",
        f"- Case 数: {detail.get('count')}",
        f"- Badcase 数: {detail.get('badcaseCount')}",
        f"- Prompt: {detail.get('promptVersion')}",
        f"- 模型: {detail.get('modelProvider')}/{detail.get('modelName')}",
        f"- 时间: {detail.get('createdAt')}",
        "",
        "## Case 明细",
        "",
    ]
    for item in items:
        badge = "Badcase" if item.get("badcase") else "OK"
        lines.extend([
            f"### {item.get('id')} · {item.get('score')} · {badge}",
            "",
            "**问题**",
            "",
            str(item.get("question") or ""),
            "",
            "**回答**",
            "",
            str(item.get("answer") or ""),
            "",
            "**检查项**",
            "",
            "```json",
            json_dumps(item.get("checks") or {}),
            "```",
            "",
        ])
    return "\n".join(lines).strip() + "\n"


def json_dumps(value: object) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2)
