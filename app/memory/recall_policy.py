"""Server-owned recall policy. Prompt/model output selects relevance, never authority."""
from dataclasses import dataclass

from app.core.config import settings
from app.memory.profile_schema import PROFILE_SCHEMA


# Order matters under a tight budget: boundaries before stylistic preferences.
ACTION_CONSTRAINT_PATHS = (
    "interaction.boundaries",
    "interaction.triggers",
    "preferences.general.dislikes",
    "preferences.communication_style",
    "preferences.output_style",
    "identity.language",
    "preferences.verbal_habits",
    "relationships.social_preferences",
    "relationships.social_style",
)


@dataclass(frozen=True)
class RecallPolicy:
    depth: str
    context_chars: int
    profile_chars: int
    short_term_chars: int
    include_vector: bool
    include_summary: bool


def recall_policy(depth: str = "full", *, include_vector: bool = True) -> RecallPolicy:
    if depth not in {"light", "full"}:
        raise ValueError("memory recall depth must be light or full")
    light = depth == "light"
    budget = settings.memory_light_context_max_chars if light else settings.memory_full_context_max_chars
    return RecallPolicy(depth, budget, budget // 2, budget // 4, include_vector and not light, not light)


def constrain_plan(plan: dict[str, object], policy: RecallPolicy) -> dict[str, object]:
    safe_paths = {path for path, item in PROFILE_SCHEMA.items() if not item.sensitive and not item.control}
    requested = plan.get("schemaPaths") or []
    if not isinstance(requested, list):
        requested = []
    selected = list(safe_paths) if "*" in requested else [path for path in requested if isinstance(path, str) and path in safe_paths]
    if policy.depth == "light":
        selected = []
    paths = list(dict.fromkeys([*ACTION_CONSTRAINT_PATHS, *(sorted(selected) if "*" in requested else selected)]))
    return {**plan, "schemaPaths": paths, "constraintPaths": list(ACTION_CONSTRAINT_PATHS), "depth": policy.depth}
