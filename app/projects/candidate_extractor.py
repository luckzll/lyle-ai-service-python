from __future__ import annotations

import re
from typing import Any


def extract_project_candidates(message: str, project_context: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    resolver = project_context.get("resolver") if isinstance(project_context, dict) else {}
    if not isinstance(resolver, dict) or not resolver.get("projectId") or resolver.get("requiresClarification"):
        return {"candidateGoals": [], "candidateDecisions": [], "candidateOpenQuestions": []}
    project_id = int(resolver["projectId"])
    text = _clean(message)
    return {
        "candidateGoals": _goal_candidates(text, project_id),
        "candidateDecisions": _decision_candidates(text, project_id),
        "candidateOpenQuestions": _open_question_candidates(text, project_id),
    }


def _goal_candidates(text: str, project_id: int) -> list[dict[str, Any]]:
    if _is_plan_request_without_goal_write(text):
        return []
    if not any(keyword in text for keyword in ["目标", "这周", "本周", "本月", "希望", "想把", "计划", "完成"]):
        return []
    title = _goal_title(text)
    if len(title) < 4:
        return []
    horizon = "week" if any(keyword in text for keyword in ["这周", "本周"]) else "month" if "本月" in text else "long_term" if "长期" in text else "week"
    return [{
        "type": "goal",
        "projectId": project_id,
        "title": title[:120],
        "description": text[:240],
        "horizon": horizon,
        "confidence": 0.72,
        "status": "pending",
        "source": text[:240],
    }]


def _is_plan_request_without_goal_write(text: str) -> bool:
    goal_write_markers = [
        "新增一个目标",
        "新增目标",
        "添加一个目标",
        "添加目标",
        "创建一个目标",
        "创建目标",
        "记录一个目标",
        "记录目标",
        "目标：",
        "目标:",
    ]
    if any(marker in text for marker in goal_write_markers):
        return False
    plan_markers = ["拆计划", "拆一个", "拆解", "分解", "推进计划", "三步计划", "三步推进"]
    return "目标" in text and any(marker in text for marker in plan_markers)


def _decision_candidates(text: str, project_id: int) -> list[dict[str, Any]]:
    if not any(keyword in text for keyword in ["决定", "确定", "暂不", "不做", "优先", "选择", "改成"]):
        return []
    title = _sentence_title(text)
    return [{
        "type": "decision",
        "projectId": project_id,
        "title": title[:80],
        "content": text[:300],
        "rationale": "",
        "confidence": 0.7,
        "status": "pending",
        "source": text[:240],
    }]


def _open_question_candidates(text: str, project_id: int) -> list[dict[str, Any]]:
    question_like = "?" in text or "？" in text or any(keyword in text for keyword in ["待定", "不确定", "需要确认", "还没确定", "开放问题"])
    if not question_like:
        return []
    question = text
    match = re.search(r"([^。！？!?]*(?:\?|？|待定|不确定|需要确认|还没确定)[^。！？!?]*)", text)
    if match:
        question = match.group(1)
    if len(question) < 4:
        return []
    return [{
        "type": "open_question",
        "projectId": project_id,
        "question": question[:240],
        "confidence": 0.68,
        "status": "pending",
        "source": text[:240],
    }]


def _clean(text: str) -> str:
    return " ".join((text or "").replace("\r\n", "\n").split()).strip()


def _trim_prefix(text: str, prefixes: list[str]) -> str:
    result = text
    for prefix in prefixes:
        result = result.replace(prefix, "")
    return re.sub(r"^[：:，,\s]+", "", result).strip()


def _goal_title(text: str) -> str:
    patterns = [
        r"(?:新增|添加|创建|记录)(?:一个)?目标[：:]\s*([^。！？!?，,]+)",
        r"目标[：:]\s*([^。！？!?，,]+)",
        r"(?:我的目标是|目标是|这周要|本周要|本月要|我希望|我想把)([^。！？!?，,]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip(" ：:，,。.")
    return _trim_prefix(
        text,
        ["我的目标是", "目标是", "这周要", "本周要", "本月要", "我希望", "我想把", "计划", "完成"],
    )


def _sentence_title(text: str) -> str:
    first = re.split(r"[。！？!?]", text, maxsplit=1)[0].strip()
    return first or text[:80]
