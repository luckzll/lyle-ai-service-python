from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings
from app.llm.tasks.judge_model import build_judge_task_model, is_judge_model_configured


DIMENSION_WEIGHTS = {
    "relevance": 0.25,
    "accuracy": 0.25,
    "completeness": 0.2,
    "groundedness": 0.2,
    "safety": 0.1,
}

BADCASE_THRESHOLD = 0.7


def evaluate_answer(
    question: str,
    answer: str,
    contexts: list[dict[str, str]] | None = None,
    expected_answer: str | None = None,
    must_include: list[str] | None = None,
    must_not_include: list[str] | None = None,
    case_type: str | None = None,
    rubric: str | None = None,
) -> dict[str, object]:
    contexts = contexts or []
    must_include = must_include or []
    must_not_include = must_not_include or []

    heuristic = _heuristic_evaluation(
        question=question,
        answer=answer,
        contexts=contexts,
        expected_answer=expected_answer or "",
        must_include=must_include,
        must_not_include=must_not_include,
        case_type=case_type or "general",
    )
    judge = _judge_evaluation(
        question=question,
        answer=answer,
        contexts=contexts,
        expected_answer=expected_answer or "",
        must_include=must_include,
        must_not_include=must_not_include,
        case_type=case_type or "general",
        rubric=rubric or "",
    )

    heuristic_score = float(heuristic["score"])
    if judge.get("enabled") and judge.get("ok"):
        judge_score = float(judge.get("score") or 0)
        score = _clamp(round(heuristic_score * 0.3 + judge_score * 0.7, 2))
        scoring_mode = "hybrid"
    else:
        judge_score = None
        score = heuristic_score
        scoring_mode = "heuristic"

    dimensions = _merge_dimensions(
        heuristic.get("dimensions") or {},
        judge.get("dimensions") if judge.get("ok") else None,
    )
    reasons = list(heuristic.get("reasons") or [])
    if judge.get("reason"):
        reasons.append(f"Judge：{judge['reason']}")

    return {
        "score": score,
        "prompt_version": settings.prompt_version,
        "scoringMode": scoring_mode,
        "heuristicScore": heuristic_score,
        "judgeScore": judge_score,
        "dimensions": dimensions,
        "reason": "；".join(reasons[:6]) or "评测通过。",
        "checks": {
            **heuristic["checks"],
            "scoring_mode": scoring_mode,
            "heuristic_score": heuristic_score,
            "judge_score": judge_score,
            "dimensions": dimensions,
            "reasons": reasons,
            "judge": judge,
            "badcase_threshold": BADCASE_THRESHOLD,
        },
    }


def _heuristic_evaluation(
    question: str,
    answer: str,
    contexts: list[dict[str, str]],
    expected_answer: str,
    must_include: list[str],
    must_not_include: list[str],
    case_type: str,
) -> dict[str, object]:
    stripped_question = question.strip()
    stripped_answer = answer.strip()
    answer_length = len(stripped_answer)
    context_text = "\n".join(item.get("chunk", "") for item in contexts)
    has_sources = bool(context_text.strip())
    has_uncertainty = any(word in stripped_answer for word in ["不确定", "不知道", "资料不足", "无法确认", "没有足够信息"])
    failure_markers = ["Traceback", "Exception", "API Key", "调用失败", "Chat stream failed", "500"]

    if not stripped_question or not stripped_answer:
        return {
            "score": 0.0,
            "dimensions": _dimensions(0, 0, 0, 0, 0),
            "reasons": ["问题或回答为空。"],
            "checks": _checks(answer_length, has_sources, False, has_uncertainty, [], [], 0, 0, case_type),
        }

    question_overlap = _overlap_ratio(stripped_question, stripped_answer)
    expected_overlap = _overlap_ratio(expected_answer, stripped_answer) if expected_answer.strip() else None
    context_overlap = _overlap_ratio(stripped_answer, context_text) if has_sources else 0.0
    included = [keyword for keyword in must_include if keyword and keyword in stripped_answer]
    missing = [keyword for keyword in must_include if keyword and keyword not in stripped_answer]
    forbidden_hit = [keyword for keyword in must_not_include if keyword and keyword in stripped_answer]
    mentions_context = _mentions_context(stripped_answer, contexts)

    relevance = _clamp(0.45 + question_overlap * 0.55)
    if answer_length < 12:
        relevance -= 0.15

    completeness = 0.35
    if answer_length >= 20:
        completeness += 0.2
    if answer_length >= 80:
        completeness += 0.15
    if must_include:
        completeness += 0.25 * (len(included) / len(must_include))
    else:
        completeness += 0.15
    completeness = _clamp(completeness)

    if expected_overlap is None:
        accuracy = 0.65
        if forbidden_hit:
            accuracy -= 0.35
        if "不知道" in stripped_answer and not has_sources:
            accuracy += 0.1
    else:
        accuracy = _clamp(0.25 + expected_overlap * 0.75)
        if forbidden_hit:
            accuracy -= 0.35
    accuracy = _clamp(accuracy)

    if has_sources:
        groundedness = _clamp(0.35 + context_overlap * 0.55 + (0.1 if mentions_context else 0))
        if has_uncertainty:
            groundedness -= 0.08
    else:
        groundedness = 0.75 if has_uncertainty else 0.55

    safety = 1.0
    if any(marker in stripped_answer for marker in failure_markers):
        safety -= 0.45
    if forbidden_hit:
        safety -= 0.25
    safety = _clamp(safety)

    dims = _dimensions(relevance, accuracy, completeness, groundedness, safety)
    score = _weighted_score(dims)
    reasons = _heuristic_reasons(
        answer_length=answer_length,
        has_sources=has_sources,
        has_uncertainty=has_uncertainty,
        missing=missing,
        forbidden_hit=forbidden_hit,
        expected_overlap=expected_overlap,
        context_overlap=context_overlap,
        score=score,
    )

    return {
        "score": score,
        "dimensions": dims,
        "reasons": reasons,
        "checks": _checks(
            answer_length,
            has_sources,
            mentions_context,
            has_uncertainty,
            missing,
            forbidden_hit,
            question_overlap,
            context_overlap,
            case_type,
        ),
    }


def _judge_evaluation(
    question: str,
    answer: str,
    contexts: list[dict[str, str]],
    expected_answer: str,
    must_include: list[str],
    must_not_include: list[str],
    case_type: str,
    rubric: str,
) -> dict[str, object]:
    if not settings.evaluation_judge_enabled:
        return {"enabled": False, "ok": False, "reason": "evaluation judge disabled"}
    if not is_judge_model_configured():
        return {"enabled": True, "ok": False, "reason": "judge model not configured"}

    model = build_judge_task_model().bind(response_format={"type": "json_object"})
    try:
        response = model.invoke([
            SystemMessage(
                content=(
                    "你是严格的 AI 评测裁判。请只返回 JSON，不要输出 Markdown。"
                    "你需要判断回答是否真正解决用户问题，是否事实正确，是否完整，"
                    "如果有知识库上下文，是否基于上下文而不是编造。"
                    "维度包括 relevance、accuracy、completeness、groundedness、safety，取值 0 到 1。"
                    "返回格式："
                    "{\"score\":0.8,\"dimensions\":{\"relevance\":0.9,\"accuracy\":0.8,"
                    "\"completeness\":0.7,\"groundedness\":0.8,\"safety\":1},\"reason\":\"...\"}"
                )
            ),
            HumanMessage(
                content=(
                    f"Case 类型：{case_type}\n"
                    f"评测规则：{rubric or '无额外规则'}\n\n"
                    f"用户问题：\n{question}\n\n"
                    f"模型回答：\n{answer}\n\n"
                    f"期望答案：\n{expected_answer or '无'}\n\n"
                    f"必须包含：{must_include or []}\n"
                    f"不得包含：{must_not_include or []}\n\n"
                    f"知识库上下文：\n{_format_contexts(contexts)}\n\n"
                    "请给出严格评分，只返回 JSON。"
                )
            ),
        ])
    except Exception as exc:
        return {"enabled": True, "ok": False, "reason": f"judge call failed: {exc}"}

    return _parse_judge_json(str(response.content or ""))


def _parse_judge_json(content: str) -> dict[str, object]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return {"enabled": True, "ok": False, "reason": "judge returned invalid JSON"}
    dimensions = parsed.get("dimensions") if isinstance(parsed, dict) else {}
    return {
        "enabled": True,
        "ok": True,
        "score": _clamp(_safe_float(parsed.get("score"), 0.0)),
        "dimensions": _normalise_dimensions(dimensions if isinstance(dimensions, dict) else {}),
        "reason": str(parsed.get("reason") or ""),
    }


def _dimensions(relevance: float, accuracy: float, completeness: float, groundedness: float, safety: float) -> dict[str, float]:
    return {
        "relevance": _clamp(relevance),
        "accuracy": _clamp(accuracy),
        "completeness": _clamp(completeness),
        "groundedness": _clamp(groundedness),
        "safety": _clamp(safety),
    }


def _normalise_dimensions(dimensions: dict[str, object]) -> dict[str, float]:
    return {
        key: _clamp(_safe_float(dimensions.get(key), 0.0))
        for key in DIMENSION_WEIGHTS
    }


def _merge_dimensions(heuristic: dict[str, object], judge: dict[str, object] | None) -> dict[str, float]:
    heuristic_dims = _normalise_dimensions(heuristic)
    if not judge:
        return heuristic_dims
    judge_dims = _normalise_dimensions(judge)
    return {
        key: _clamp(round(heuristic_dims[key] * 0.3 + judge_dims[key] * 0.7, 2))
        for key in DIMENSION_WEIGHTS
    }


def _weighted_score(dimensions: dict[str, float]) -> float:
    return _clamp(round(sum(dimensions[key] * weight for key, weight in DIMENSION_WEIGHTS.items()), 2))


def _checks(
    answer_length: int,
    has_sources: bool,
    mentions_context: bool,
    has_uncertainty: bool,
    missing_keywords: list[str],
    forbidden_hits: list[str],
    question_overlap: float,
    context_overlap: float,
    case_type: str,
) -> dict[str, object]:
    return {
        "answer_length": answer_length,
        "has_sources": has_sources,
        "mentions_context": mentions_context,
        "has_uncertainty": has_uncertainty,
        "missing_keywords": missing_keywords,
        "forbidden_hits": forbidden_hits,
        "question_overlap": round(question_overlap, 3),
        "context_overlap": round(context_overlap, 3),
        "case_type": case_type,
    }


def _heuristic_reasons(
    answer_length: int,
    has_sources: bool,
    has_uncertainty: bool,
    missing: list[str],
    forbidden_hit: list[str],
    expected_overlap: float | None,
    context_overlap: float,
    score: float,
) -> list[str]:
    reasons = []
    if answer_length < 20:
        reasons.append("回答较短，完整性偏弱。")
    if missing:
        reasons.append(f"缺少必须包含内容：{', '.join(missing[:5])}。")
    if forbidden_hit:
        reasons.append(f"命中禁止内容：{', '.join(forbidden_hit[:5])}。")
    if expected_overlap is not None and expected_overlap < 0.35:
        reasons.append("与期望答案重合度较低。")
    if has_sources and context_overlap < 0.2:
        reasons.append("回答与知识库上下文关联较弱。")
    if has_uncertainty and has_sources:
        reasons.append("已有上下文时仍表达不确定，可能没有充分利用资料。")
    if not reasons and score >= BADCASE_THRESHOLD:
        reasons.append("规则评估通过。")
    return reasons


def _mentions_context(answer: str, contexts: list[dict[str, str]]) -> bool:
    for item in contexts:
        chunk = item.get("chunk", "").strip()
        if chunk and chunk[:20] in answer:
            return True
    return False


def _overlap_ratio(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def _tokens(text: str) -> set[str]:
    lowered = text.lower()
    latin = re.findall(r"[a-z0-9_]+", lowered)
    chinese = re.findall(r"[\u4e00-\u9fff]", lowered)
    tokens = set(latin + chinese)
    stop_tokens = set("的一是在了和与或吗呢啊吧呀")
    return {token for token in tokens if token not in stop_tokens and token.strip()}


def _format_contexts(contexts: list[dict[str, str]]) -> str:
    if not contexts:
        return "无"
    return "\n\n".join(
        f"来源：{item.get('title', '')}\n内容：{item.get('chunk', '')}"
        for item in contexts[:8]
    )


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, round(value, 2)))
