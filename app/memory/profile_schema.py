from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProfileSchemaItem:
    path: str
    label: str
    group: str
    multi_value: bool = False
    sensitive: bool = False
    control: bool = False


PROFILE_SCHEMA: dict[str, ProfileSchemaItem] = {
    "identity.name": ProfileSchemaItem("identity.name", "姓名", "身份"),
    "identity.alias": ProfileSchemaItem("identity.alias", "别名", "身份", multi_value=True),
    "identity.language": ProfileSchemaItem("identity.language", "语言", "身份"),
    "identity.zodiac": ProfileSchemaItem("identity.zodiac", "星座", "身份"),
    "work.current_role": ProfileSchemaItem("work.current_role", "当前角色", "工作"),
    "work.background": ProfileSchemaItem("work.background", "背景", "工作"),
    "work.company_or_project": ProfileSchemaItem("work.company_or_project", "公司或项目", "工作"),
    "work.working_style": ProfileSchemaItem("work.working_style", "工作方式", "工作"),
    "preferences.food.likes": ProfileSchemaItem("preferences.food.likes", "喜欢的食物", "偏好", multi_value=True),
    "preferences.food.dislikes": ProfileSchemaItem("preferences.food.dislikes", "不喜欢的食物", "偏好", multi_value=True),
    "preferences.visual.likes": ProfileSchemaItem("preferences.visual.likes", "视觉偏好", "偏好", multi_value=True),
    "preferences.visual.dislikes": ProfileSchemaItem("preferences.visual.dislikes", "视觉负向偏好", "偏好", multi_value=True),
    "preferences.communication_style": ProfileSchemaItem("preferences.communication_style", "沟通风格", "偏好"),
    "preferences.output_style": ProfileSchemaItem("preferences.output_style", "输出风格", "偏好"),
    "preferences.verbal_habits": ProfileSchemaItem("preferences.verbal_habits", "语言习惯", "偏好"),
    "relationships.social_preferences": ProfileSchemaItem("relationships.social_preferences", "社交偏好", "关系", multi_value=True),
    "relationships.social_style": ProfileSchemaItem("relationships.social_style", "社交风格", "关系"),
    "interaction.boundaries": ProfileSchemaItem("interaction.boundaries", "互动边界", "互动", multi_value=True),
    "interaction.triggers": ProfileSchemaItem("interaction.triggers", "应避免的互动触发点", "互动", multi_value=True),
    "preferences.travel.interests": ProfileSchemaItem("preferences.travel.interests", "旅行兴趣", "偏好", multi_value=True),
    "preferences.entertainment.likes": ProfileSchemaItem("preferences.entertainment.likes", "娱乐偏好", "偏好", multi_value=True),
    "preferences.entertainment.dislikes": ProfileSchemaItem("preferences.entertainment.dislikes", "娱乐负向偏好", "偏好", multi_value=True),
    "preferences.general.likes": ProfileSchemaItem("preferences.general.likes", "一般偏好", "偏好", multi_value=True),
    "preferences.general.dislikes": ProfileSchemaItem("preferences.general.dislikes", "一般负向偏好", "偏好", multi_value=True),
    "goals.current_week": ProfileSchemaItem("goals.current_week", "本周目标", "目标", multi_value=True),
    "goals.current_month": ProfileSchemaItem("goals.current_month", "本月目标", "目标", multi_value=True),
    "goals.long_term": ProfileSchemaItem("goals.long_term", "长期目标", "目标", multi_value=True),
    "projects.current": ProfileSchemaItem("projects.current", "当前项目", "项目", multi_value=True),
    "projects.decisions": ProfileSchemaItem("projects.decisions", "项目决策", "项目", multi_value=True),
    "projects.open_questions": ProfileSchemaItem("projects.open_questions", "开放问题", "项目", multi_value=True),
    "habits.routine": ProfileSchemaItem("habits.routine", "日常习惯", "习惯", multi_value=True),
    "habits.reminders": ProfileSchemaItem("habits.reminders", "提醒偏好", "习惯", multi_value=True),
    "relationships.people": ProfileSchemaItem("relationships.people", "重要关系", "关系", multi_value=True),
    "memory_control.forget_requests": ProfileSchemaItem("memory_control.forget_requests", "遗忘请求", "治理", multi_value=True, control=True),
    "memory_control.sensitive_info": ProfileSchemaItem("memory_control.sensitive_info", "敏感信息", "治理", multi_value=True, sensitive=True),
}


LEGACY_KEY_TO_SCHEMA_PATH: dict[str, str] = {
    "name": "identity.name",
    "identity": "identity.alias",
    "zodiac": "identity.zodiac",
    "career": "work.current_role",
    "role": "work.current_role",
    "current_project": "projects.current",
    "project": "projects.current",
    "goal": "goals.current_week",
    "travel_interest": "preferences.travel.interests",
    "food_preference": "preferences.food.likes",
    "food_negative_preference": "preferences.food.dislikes",
    "pet_preference": "preferences.general.likes",
    "pet_negative_preference": "preferences.general.dislikes",
    "sport_preference": "preferences.general.likes",
    "sport_negative_preference": "preferences.general.dislikes",
    "entertainment_preference": "preferences.entertainment.likes",
    "entertainment_negative_preference": "preferences.entertainment.dislikes",
    "negative_preference": "preferences.general.dislikes",
    "preference": "preferences.general.likes",
    "technology_stack": "work.background",
    "constraint": "projects.open_questions",
    "forget_request": "memory_control.forget_requests",
    "sensitive_info": "memory_control.sensitive_info",
}

_VISUAL_TERMS = {"帽子", "帽衫", "头像", "发型", "头发", "穿搭", "图片", "照片", "形象", "视觉", "设计", "颜色", "配色"}


def schema_path_for_memory(key: str, value: str = "") -> str:
    cleaned_key = key.strip()
    if cleaned_key in PROFILE_SCHEMA:
        return cleaned_key
    mapped = LEGACY_KEY_TO_SCHEMA_PATH.get(cleaned_key, cleaned_key)
    if cleaned_key in {"negative_preference", "preference"} and _looks_visual(value):
        return "preferences.visual.dislikes" if cleaned_key == "negative_preference" else "preferences.visual.likes"
    if cleaned_key.endswith("_negative_preference") and _looks_visual(value):
        return "preferences.visual.dislikes"
    return mapped


def schema_item(path: str) -> ProfileSchemaItem:
    return PROFILE_SCHEMA.get(path, ProfileSchemaItem(path, path, path.split(".", 1)[0] or "其他"))


def is_multi_value_path(path: str) -> bool:
    return schema_item(path).multi_value


def is_sensitive_path(path: str) -> bool:
    return schema_item(path).sensitive


def is_control_path(path: str) -> bool:
    return schema_item(path).control


def profile_schema_overview() -> list[dict[str, object]]:
    return [
        {
            "path": item.path,
            "label": item.label,
            "group": item.group,
            "multiValue": item.multi_value,
            "sensitive": item.sensitive,
            "control": item.control,
        }
        for item in PROFILE_SCHEMA.values()
    ]


def _looks_visual(value: str) -> bool:
    return any(term in value for term in _VISUAL_TERMS)
