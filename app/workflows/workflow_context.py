from __future__ import annotations

import re
from typing import Any


REFERENCE_PATTERN = re.compile(r"\{\{\s*([a-zA-Z0-9_.-]+)\s*\}\}")
MISSING = object()
SUPPORTED_OPERATORS = {"exists", "not_exists", "equals", "not_equals", "truthy", "falsy", "contains"}


class TemplateResolutionError(ValueError):
    pass


def resolve_template(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: resolve_template(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_template(item, context) for item in value]
    if not isinstance(value, str):
        return value

    full_match = REFERENCE_PATTERN.fullmatch(value)
    if full_match:
        return _required_path(context, full_match.group(1))

    def replace(match: re.Match[str]) -> str:
        resolved = _required_path(context, match.group(1))
        return "" if resolved is None else str(resolved)

    return REFERENCE_PATTERN.sub(replace, value)


def condition_matches(condition: dict[str, Any] | None, context: dict[str, Any]) -> bool:
    if not condition:
        return True
    path = str(condition.get("path") or "").strip()
    operator = str(condition.get("operator") or "truthy").strip()
    if not path or operator not in SUPPORTED_OPERATORS:
        raise ValueError("工作流条件配置无效。")
    actual = _path(context, path)
    expected = condition.get("value")
    if operator == "exists":
        return actual is not MISSING
    if operator == "not_exists":
        return actual is MISSING
    if operator == "truthy":
        return actual is not MISSING and bool(actual)
    if operator == "falsy":
        return actual is MISSING or not bool(actual)
    if operator == "equals":
        return actual is not MISSING and actual == expected
    if operator == "not_equals":
        return actual is MISSING or actual != expected
    if actual is MISSING:
        return False
    if isinstance(actual, (list, tuple, set, dict, str)):
        return expected in actual
    return False


def validate_condition(condition: object) -> dict[str, Any] | None:
    if condition in (None, {}):
        return None
    if not isinstance(condition, dict):
        raise ValueError("工作流条件必须是对象。")
    path = str(condition.get("path") or "").strip()
    operator = str(condition.get("operator") or "truthy").strip()
    if not path or not (path.startswith("input.") or path.startswith("steps.")):
        raise ValueError("条件路径必须以 input. 或 steps. 开头。")
    if operator not in SUPPORTED_OPERATORS:
        raise ValueError("工作流条件运算符无效。")
    return {"path": path, "operator": operator, "value": condition.get("value")}


def _required_path(context: dict[str, Any], path: str) -> Any:
    value = _path(context, path)
    if value is MISSING:
        raise TemplateResolutionError(f"找不到工作流引用：{path}")
    return value


def _path(context: dict[str, Any], path: str) -> Any:
    current: Any = context
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, (list, tuple)) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return MISSING
    return current
