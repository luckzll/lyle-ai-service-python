from __future__ import annotations

from collections.abc import Callable
from app.core.config import settings

from app.memory.fragments import list_memory_fragments
from app.memory.evidence_store import evidence_count_by_profile
from app.memory.governance import reliability_for_profile
from app.memory.profile_schema import is_control_path, is_sensitive_path, schema_item
from app.memory.profile_store import PROFILE_ACTIVE_RECALL_MIN_CONFIDENCE, list_user_profiles, mark_profiles_used


def build_profile_memory_context(
    user_id: str, question: str, plan: dict[str, object], *, limit: int = 8,
    max_chars: int = 4000, is_blocked: Callable[[str], bool] | None = None,
    candidates: list[dict] | None = None,
) -> dict[str, object]:
    requested_paths = [str(item) for item in plan.get("schemaPaths") or []]
    all_profiles = list_user_profiles(user_id, limit=500) if candidates is None else candidates
    profiles = [item for item in all_profiles if str(item.get("status") or "") == "active"]
    profiles = [
        item for item in profiles
        if float(item.get("confidence") or 0) >= PROFILE_ACTIVE_RECALL_MIN_CONFIDENCE
    ]
    if requested_paths != ["*"]:
        allowed = set(requested_paths)
        profiles = [item for item in profiles if str(item.get("schemaPath")) in allowed]
    sensitive_count = sum(1 for item in profiles if is_sensitive_path(str(item.get("schemaPath") or "")))
    profiles = [item for item in profiles if not is_sensitive_path(str(item.get("schemaPath") or ""))
                and not is_control_path(str(item.get("schemaPath") or ""))]
    if is_blocked:
        profiles = [item for item in profiles if not is_blocked(str(item.get("summary") or item.get("value") or ""))]
    constraints = list(plan.get("constraintPaths") or [])
    core_paths = list(plan.get("corePaths") or [])
    order = {path: index for index, path in enumerate(dict.fromkeys([*core_paths, *constraints, *requested_paths]))}
    profiles.sort(key=lambda item: order.get(str(item.get("schemaPath")), len(order)))
    selected: list[dict[str, object]] = []
    relevant_count = 0
    for profile in profiles:
        core = profile.get("schemaPath") in core_paths
        fixed = profile.get("schemaPath") in constraints or core
        if core and len(_format_context([*[item for item in selected if item.get('schemaPath') in core_paths], profile])) > settings.memory_core_max_chars:
            continue
        if not fixed and relevant_count >= limit:
            continue
        if len(_format_context([*selected, profile])) > max_chars:
            continue
        selected.append(profile)
        if not fixed:
            relevant_count += 1
    profiles = selected
    mark_profiles_used(user_id, [str(item.get("schemaPath")) for item in profiles if item.get("schemaPath")])

    evidence_counts = evidence_count_by_profile(user_id)
    reason_code = _reason_code(question, plan)
    used_memories = [
        {
            "memoryId": f"profile:{int(profile.get('id') or 0)}",
            "title": str(profile.get("label") or schema_item(str(profile.get("schemaPath") or "")).label),
            "valueSummary": str(profile.get("summary") or profile.get("value") or "")[:240],
            "reasonCode": "ACTION_CONSTRAINT" if profile.get("schemaPath") in constraints else "CORE_PROFILE" if profile.get("schemaPath") in core_paths else reason_code,
            "reliability": reliability_for_profile(profile),
            "sourceCount": profile.get('sourceCount', len(profile["sourceIds"]) if "sourceIds" in profile else evidence_counts.get(int(profile.get("id") or 0), 0)),
        }
        for profile in profiles
    ]

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

    if candidates is not None:
        allowed_sources = {source for profile in profiles for source in profile.get("sourceIds", [])}
        fragments = [fragment for fragment in fragments if f"fragment:{fragment['id']}" in allowed_sources]
    return {
        "context": _format_context(profiles),
        "profiles": profiles,
        "fragments": fragments[:limit],
        "recallPlan": plan,
        "usedMemories": used_memories,
        "memoryExclusions": {
            "conflicted": sum(1 for item in all_profiles if str(item.get("status") or "") == "conflicted"),
            "sensitive": sensitive_count,
            "governed": sum(
                1 for item in all_profiles
                if str(item.get("status") or "") in {"disabled", "forgotten", "expired"}
            ),
        },
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


def _reason_code(question: str, plan: dict[str, object]) -> str:
    if any(marker in question for marker in ("记得", "回忆", "之前", "上次")):
        return "USER_REQUESTED_RECALL"
    if str(plan.get("intent") or "") in {"project", "planning"}:
        return "PROJECT_CONTEXT"
    return "QUESTION_RELEVANT"
