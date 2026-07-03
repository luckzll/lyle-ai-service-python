from __future__ import annotations

from app.memory.evidence_store import create_profile_evidence
from app.memory.fragments import create_memory_fragment
from app.memory.profile_schema import schema_path_for_memory
from app.memory.profile_store import get_user_profile, set_user_profile_status, upsert_user_profile


def build_profile_from_structured_memory(
    user_id: str,
    key: str,
    value: str,
    source: str,
    confidence: float,
) -> dict[str, object]:
    schema_path = schema_path_for_memory(key, value)
    source_ref, source_text = parse_source(source)
    fragment = create_memory_fragment(
        user_id,
        source_text=source_text or value,
        normalized_text=value,
        category=schema_path,
        conversation_id=_conversation_id_from_ref(source_ref),
        source_type=_source_type_from_ref(source_ref),
        source_id=_source_id_from_ref(source_ref),
        source_ref=source_ref,
        confidence=confidence,
        status="active",
    )
    existing = get_user_profile(user_id, schema_path)
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


def parse_source(source: str) -> tuple[str, str]:
    if not source.startswith("source_ref:"):
        return "", source
    _, _, rest = source.partition("source_ref:")
    ref, separator, text = rest.partition("\n")
    return ref.strip(), text if separator else ""


def _relation_for_profile(existing: dict[str, object] | None, value: str) -> str:
    if not existing:
        return "support"
    existing_value = str(existing.get("value") or "")
    if not existing_value or value in existing_value or existing_value in value:
        return "support"
    return "correction" if int(existing.get("correctionCount") or 0) > 0 else "support"


def _conversation_id_from_ref(source_ref: str) -> str | None:
    return source_ref.removeprefix("conversation:") if source_ref.startswith("conversation:") else None


def _source_type_from_ref(source_ref: str) -> str:
    if source_ref.startswith("knowledge:"):
        return "knowledge_file"
    if source_ref.startswith("conversation:"):
        return "conversation"
    return "manual"


def _source_id_from_ref(source_ref: str) -> str:
    if ":" not in source_ref:
        return source_ref
    return source_ref.split(":", 1)[1]
