from __future__ import annotations

from app.memory.fragments import list_memory_fragments
from app.memory.profile_schema import is_sensitive_path, schema_item
from app.memory.profile_store import PROFILE_ACTIVE_RECALL_MIN_CONFIDENCE, list_user_profiles, mark_profiles_used


def build_profile_memory_context(user_id: str, question: str, plan: dict[str, object], *, limit: int = 8) -> dict[str, object]:
    requested_paths = [str(item) for item in plan.get("schemaPaths") or []]
    profiles = list_user_profiles(user_id, status="active", limit=200)
    profiles = [
        item for item in profiles
        if float(item.get("confidence") or 0) >= PROFILE_ACTIVE_RECALL_MIN_CONFIDENCE
    ]
    if requested_paths != ["*"]:
        allowed = set(requested_paths)
        profiles = [item for item in profiles if str(item.get("schemaPath")) in allowed]
    profiles = profiles[:limit]
    mark_profiles_used(user_id, [str(item.get("schemaPath")) for item in profiles if item.get("schemaPath")])

    fragments: list[dict[str, object]] = []
    for profile in profiles:
        fragments.extend(
            list_memory_fragments(
                user_id,
                status="active",
                category=str(profile.get("schemaPath") or ""),
                limit=2,
            )
        )

    return {
        "context": _format_context(profiles),
        "profiles": profiles,
        "fragments": fragments[:limit],
        "recallPlan": plan,
    }


def _format_context(profiles: list[dict[str, object]]) -> str:
    if not profiles:
        return ""
    normal_lines: list[str] = []
    cautious_lines: list[str] = []
    sensitive_count = 0
    for profile in profiles:
        path = str(profile.get("schemaPath") or "")
        if is_sensitive_path(path):
            sensitive_count += 1
            continue
        label = schema_item(path).label
        line = f"- {label}：{profile.get('summary') or profile.get('value')}"
        if float(profile.get("confidence") or 0) < 0.75:
            cautious_lines.append(line)
        else:
            normal_lines.append(line)
    parts: list[str] = []
    if normal_lines:
        parts.append("用户画像：\n" + "\n".join(normal_lines))
    if cautious_lines:
        parts.append("低置信度，仅作参考：\n" + "\n".join(cautious_lines))
    if sensitive_count:
        parts.append(f"敏感画像：本轮已过滤 {sensitive_count} 条。")
    return "\n\n".join(parts)
