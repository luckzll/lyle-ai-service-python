from __future__ import annotations

from app.core.config import settings
from app.memory.profile_schema import PROFILE_SCHEMA


_RULES: list[tuple[tuple[str, ...], list[str], str, str]] = [
    (("喜欢什么", "讨厌什么", "我的偏好"), ["preferences.food.likes", "preferences.food.dislikes", "preferences.general.likes", "preferences.general.dislikes", "preferences.visual.likes", "preferences.visual.dislikes", "preferences.entertainment.likes", "preferences.entertainment.dislikes"], "用户在询问已有偏好。", "preferences"),
    (("记得我什么", "你记得", "我的记忆", "用户画像", "记忆概览"), ["*"], "用户要求查看完整记忆概览。", "overview"),
    (("名字", "我是谁", "身份", "介绍我", "怎么称呼"), ["identity.name", "identity.alias", "identity.language", "identity.zodiac"], "用户在询问身份信息。", "identity"),
    (("职业", "工作", "岗位", "产品经理", "工程师", "背景"), ["work.current_role", "work.background", "work.working_style", "work.company_or_project"], "用户在询问工作背景。", "work"),
    (("项目", "规划", "推进", "产品定位", "文档", "需求", "方案", "PRD", "prd", "MVP", "mvp"), ["work.current_role", "work.background", "projects.current", "projects.decisions", "projects.open_questions", "goals.current_week", "preferences.output_style", "preferences.communication_style"], "用户在询问项目或工作规划。", "project"),
    (("这周", "本周", "目标", "计划", "待办", "进度", "复盘"), ["goals.current_week", "goals.current_month", "projects.current", "work.current_role", "preferences.output_style"], "用户在询问近期目标或计划。", "planning"),
    (("图片", "头像", "帽子", "帽衫", "视觉", "风格", "配色", "设计", "颜色", "形象", "画面"), ["preferences.visual.likes", "preferences.visual.dislikes"], "用户在询问视觉偏好。", "visual"),
    (("怎么说", "表达", "语气", "沟通", "回复风格", "输出", "写法", "文案"), ["preferences.communication_style", "preferences.output_style", "work.working_style"], "用户在询问沟通或输出偏好。", "communication"),
    (("吃", "食物", "水果", "餐", "咖啡", "奶茶"), ["preferences.food.likes", "preferences.food.dislikes"], "用户在询问饮食偏好。", "food"),
    (("旅行", "旅游", "城市", "杭州", "目的地"), ["preferences.travel.interests"], "用户在询问旅行偏好。", "travel"),
    (("娱乐", "游戏", "电影", "音乐", "唱歌"), ["preferences.entertainment.likes", "preferences.entertainment.dislikes"], "用户在询问娱乐偏好。", "entertainment"),
]


def plan_recall(question: str, *, allow_model: bool = True) -> dict[str, object]:
    if allow_model and settings.recall_planner_llm_enabled:
        model_plan = _plan_recall_with_model(question)
        if model_plan:
            return model_plan
    return _plan_recall_by_rule(question)


def _plan_recall_with_model(question: str) -> dict[str, object] | None:
    try:
        from app.chains.recall_planner_chain import (
            invoke_recall_planner_chain,
            is_recall_planner_model_configured,
        )

        if not is_recall_planner_model_configured():
            return None
        return _normalise_model_plan(invoke_recall_planner_chain(question))
    except Exception:
        return None


def _normalise_model_plan(plan: dict[str, object]) -> dict[str, object] | None:
    raw_paths = plan.get("schemaPaths") or plan.get("schema_paths") or []
    if not isinstance(raw_paths, list):
        return None
    schema_paths = [str(item).strip() for item in raw_paths if str(item).strip()]
    if "*" not in schema_paths:
        schema_paths = [path for path in schema_paths if path in PROFILE_SCHEMA]
    if not schema_paths:
        return None
    if "*" in schema_paths:
        schema_paths = ["*"]
    return {
        "schemaPaths": schema_paths[:10],
        "reason": str(plan.get("reason") or "模型选择需要召回的画像路径。")[:240],
        "intent": str(plan.get("intent") or "model"),
        "fallback": False,
        "planner": "llm",
        "confidence": _safe_confidence(plan.get("confidence")),
    }


def _safe_confidence(value: object) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.75


def _plan_recall_by_rule(question: str) -> dict[str, object]:
    cleaned = question.strip()
    matched: list[dict[str, object]] = []
    for terms, paths, reason, intent in _RULES:
        if any(term in question for term in terms):
            matched.append({
                "schemaPaths": paths,
                "reason": reason,
                "intent": intent,
                "fallback": False,
                "planner": "rule",
            })
    if matched:
        overview = next((item for item in matched if item["schemaPaths"] == ["*"]), None)
        if overview:
            return overview
        paths: list[str] = []
        reasons: list[str] = []
        intents: list[str] = []
        for item in matched:
            reasons.append(str(item["reason"]))
            intents.append(str(item["intent"]))
            for path in item["schemaPaths"]:  # type: ignore[union-attr]
                if str(path) not in paths:
                    paths.append(str(path))
        return {
            "schemaPaths": paths[:10],
            "reason": "；".join(dict.fromkeys(reasons)),
            "intent": "+".join(dict.fromkeys(intents)),
            "fallback": False,
            "planner": "rule",
        }
    if not cleaned:
        return {"schemaPaths": ["identity.name", "projects.current", "goals.current_week"], "reason": "空问题使用最小核心画像。", "intent": "minimal", "fallback": True, "planner": "rule"}
    return {
        "schemaPaths": [],
        "reason": "未命中主题，仅保留策略允许的行为约束和会话上下文。",
        "intent": "default",
        "fallback": False,
        "planner": "rule",
    }
