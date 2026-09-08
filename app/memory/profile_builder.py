from __future__ import annotations

import re

from app.memory.evidence_store import create_profile_evidence, list_profile_evidence
from app.memory.fragments import create_memory_fragment
from app.memory.profile_schema import is_multi_value_path, schema_path_for_memory
from app.memory.profile_store import get_user_profile, set_user_profile_status, update_user_profile_summary, upsert_user_profile


def build_profile_from_structured_memory(
    user_id: str,
    key: str,
    value: str,
    source: str,
    confidence: float,
) -> dict[str, object]:
    schema_path = schema_path_for_memory(key, value)
    source_ref, source_text = parse_source(source)
    existing = get_user_profile(user_id, schema_path)
    # A retry of the same message confirmation is not an independent source.
    # Without a message anchor, keep distinct observations rather than guessing.
    if existing and existing.get("status") == "active" and _source_message_id_from_ref(source_ref):
        for item in list_profile_evidence(user_id, int(existing["id"]), limit=100):
            prior = item.get("fragment") if isinstance(item.get("fragment"), dict) else {}
            if (prior.get("status") == "active" and prior.get("sourceRef") == source_ref
                    and _normalized_comparison(str(prior.get("normalizedText") or "")) == _normalized_comparison(value)):
                return {
                    "action": "duplicate_source", "schemaPath": schema_path, "value": value,
                    "relation": str(item.get("relation") or "duplicate"), "fragment": prior,
                    "profile": existing, "evidence": item, "requiresUserConfirmation": False,
                }
    fragment = create_memory_fragment(
        user_id,
        source_text=source_text or value,
        normalized_text=value,
        category=schema_path,
        conversation_id=_conversation_id_from_ref(source_ref),
        source_type=_source_type_from_ref(source_ref),
        source_id=_source_id_from_ref(source_ref),
        source_ref=source_ref,
        source_message_id=_source_message_id_from_ref(source_ref),
        confidence=confidence,
        status="active",
    )
    relation = _relation_for_profile(existing, value)
    if relation == "contradict":
        profile = set_user_profile_status(user_id, schema_path, "conflicted")
        action = "conflict"
    else:
        profile = upsert_user_profile(
            user_id,
            schema_path,
            value,
            summary=value,
            confidence=confidence,
            status="active",
        )
        action = "upsert_profile"
    evidence = {}
    if profile:
        evidence = create_profile_evidence(
            user_id,
            int(profile["id"]),
            int(fragment["id"]),
            relation=relation,
            confidence=confidence,
        )
        if action == "upsert_profile":
            profile = synthesize_profile_summary(user_id, profile)
        elif action == "conflict":
            from app.memory.governance import create_memory_conflict

            create_memory_conflict(user_id, schema_path, int(fragment["id"]))
    return {
        "action": action,
        "schemaPath": schema_path,
        "value": value,
        "relation": relation,
        "fragment": fragment,
        "profile": profile,
        "evidence": evidence,
        "requiresUserConfirmation": action == "conflict",
    }


def synthesize_profile_summary(user_id: str, profile: dict[str, object]) -> dict[str, object]:
    profile_id = int(profile.get("id") or 0)
    schema_path = str(profile.get("schemaPath") or "")
    if not profile_id or not schema_path:
        return profile
    evidence = list_profile_evidence(user_id, profile_id, limit=50)
    # A single-valued slot must not concatenate historical versions into a new fact.
    values = _evidence_values(evidence) if is_multi_value_path(schema_path) else [str(profile.get("value") or "")]
    if not values:
        values = _split_items(str(profile.get("value") or ""))
    if not values:
        return profile
    summary = _synthesize_summary(schema_path, values)
    if not summary:
        return profile
    boost = 0.02 if len(values) >= 2 else 0.0
    return update_user_profile_summary(user_id, schema_path, summary, confidence_boost=boost) or profile


def parse_source(source: str) -> tuple[str, str]:
    if not source.startswith("source_ref:"):
        return "", source
    _, _, rest = source.partition("source_ref:")
    ref, separator, text = rest.partition("\n")
    return ref.strip(), text if separator else ""


def _relation_for_profile(existing: dict[str, object] | None, value: str) -> str:
    if not existing:
        return "support"
    if existing.get("status") == "conflicted":
        return "contradict"  # Only explicit conflict resolution may reactivate it.
    existing_value = str(existing.get("value") or "")
    if not existing_value:
        return "support"
    normalized_existing = _normalized_comparison(existing_value)
    normalized_candidate = _normalized_comparison(value)
    if normalized_existing == normalized_candidate:
        return "duplicate"
    if is_multi_value_path(str(existing.get("schemaPath") or "")):
        existing_items = set(_split_items(existing_value))
        candidate_items = set(_split_items(value))
        return "duplicate" if candidate_items and candidate_items.issubset(existing_items) else "supplement"
    if normalized_candidate in normalized_existing or normalized_existing in normalized_candidate:
        return "support"
    return "contradict"


def _evidence_values(evidence: list[dict[str, object]]) -> list[str]:
    values: list[str] = []
    for item in reversed(evidence):
        if item.get("relation") not in {"support", "supplement", "duplicate"}:
            continue
        fragment = item.get("fragment") if isinstance(item.get("fragment"), dict) else {}
        if fragment.get("status") != "active":
            continue
        values.extend(_split_items(str(fragment.get("normalizedText") or "")))
    return _dedupe_values(values)


def _split_items(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[、,，;；]+", value or "") if item.strip()]


def _normalized_comparison(value: str) -> str:
    return re.sub(r"[\s，。,.!！?？：:、\"'“”‘’（）()\[\]【】]", "", value or "").lower()


def _dedupe_values(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _synthesize_summary(schema_path: str, values: list[str]) -> str:
    joined = "、".join(values[:8])
    if schema_path == "preferences.visual.dislikes":
        return f"用户在视觉形象上不喜欢{joined}。"
    if schema_path == "preferences.visual.likes":
        return f"用户在视觉形象上偏好{joined}。"
    if schema_path == "preferences.food.likes":
        return f"用户喜欢{joined}。"
    if schema_path == "preferences.food.dislikes":
        return f"用户不喜欢{joined}。"
    if schema_path == "preferences.entertainment.likes":
        return f"用户的娱乐偏好包括{joined}。"
    if schema_path == "preferences.general.likes":
        return f"用户偏好{joined}。"
    if schema_path == "preferences.general.dislikes":
        return f"用户不喜欢{joined}。"
    if schema_path == "projects.current":
        return f"用户当前关注项目包括{joined}。" if len(values) > 1 else f"用户当前关注项目是{joined}。"
    if schema_path.startswith("goals."):
        return f"用户当前目标包括{joined}。" if len(values) > 1 else f"用户当前目标是{joined}。"
    if schema_path == "work.current_role":
        return values[-1]
    if schema_path == "work.background":
        return f"用户背景包括{joined}。" if is_multi_value_path(schema_path) else f"用户背景是{values[-1]}。"
    if schema_path.startswith("memory_control."):
        return joined
    return joined


def _conversation_id_from_ref(source_ref: str) -> str | None:
    if not source_ref.startswith("conversation:"):
        return None
    return source_ref.removeprefix("conversation:").split(";message:", 1)[0] or None


def _source_message_id_from_ref(source_ref: str) -> str | None:
    if ";message:" not in source_ref:
        return None
    return source_ref.split(";message:", 1)[1] or None


def _source_type_from_ref(source_ref: str) -> str:
    if source_ref.startswith("knowledge:"):
        return "knowledge_file"
    if source_ref.startswith("conversation:"):
        return "conversation"
    return "manual"


def _source_id_from_ref(source_ref: str) -> str:
    if ":" not in source_ref:
        return source_ref
    return source_ref.split(":", 1)[1].split(";message:", 1)[0]
