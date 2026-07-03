from __future__ import annotations


_RULES: list[tuple[tuple[str, ...], list[str], str]] = [
    (("名字", "我是谁", "身份", "介绍我"), ["identity.name", "identity.alias", "identity.language"], "用户在询问身份信息。"),
    (("职业", "工作", "岗位", "产品经理", "工程师"), ["work.current_role", "work.background", "work.working_style"], "用户在询问工作背景。"),
    (("项目", "规划", "推进", "产品定位", "文档", "需求"), ["work.current_role", "projects.current", "projects.decisions", "goals.current_week", "preferences.output_style"], "用户在询问项目或工作规划。"),
    (("这周", "本周", "目标", "计划", "待办"), ["goals.current_week", "projects.current", "work.current_role", "preferences.output_style"], "用户在询问近期目标或计划。"),
    (("图片", "头像", "帽子", "视觉", "风格", "配色", "设计"), ["preferences.visual.likes", "preferences.visual.dislikes"], "用户在询问视觉偏好。"),
    (("吃", "食物", "水果", "餐", "咖啡", "奶茶"), ["preferences.food.likes", "preferences.food.dislikes"], "用户在询问饮食偏好。"),
    (("旅行", "旅游", "城市"), ["preferences.travel.interests"], "用户在询问旅行偏好。"),
    (("记得我什么", "你记得", "记忆", "画像"), ["*"], "用户要求查看完整记忆概览。"),
]


def plan_recall(question: str) -> dict[str, object]:
    for terms, paths, reason in _RULES:
        if any(term in question for term in terms):
            return {"schemaPaths": paths, "reason": reason, "fallback": False}
    return {
        "schemaPaths": [
            "identity.name",
            "work.current_role",
            "projects.current",
            "goals.current_week",
            "preferences.output_style",
        ],
        "reason": "默认轻量召回核心画像。",
        "fallback": False,
    }
