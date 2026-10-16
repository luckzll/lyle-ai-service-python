"""Session continuity and mandatory governance without long-term retrieval."""
from dataclasses import asdict

from app.core.config import settings


def _valid_scope(user_id: str, conversation_id: str | None) -> bool:
    # Never alias a missing conversation to the shared legacy 'default' key.
    return bool(user_id and conversation_id and all(
        value == value.strip() and len(value) <= 256 and ':' not in value
        and not any(ord(char) < 32 for char in value)
        for value in (user_id, conversation_id)
    ))


def load_conversation_context(user_id: str, conversation_id: str | None) -> dict[str, object]:
    from app.memory import store

    governance = store._memory_governance_snapshot(user_id)
    # Behavior boundaries remain mandatory even when personal/semantic recall is
    # unnecessary. Reuse the same active/confidence/governance filters as normal
    # recall, limited to these fixed configuration paths and a small budget.
    boundary_paths = ['interaction.boundaries', 'interaction.triggers']
    candidates = None
    from app.memory.fact_store import enabled as facts_enabled
    if facts_enabled():
        from app.memory.layered_context import core_profiles
        candidates = core_profiles(user_id, allowed_paths=boundary_paths)
    budget = settings.memory_light_context_max_chars
    boundaries = store.build_profile_memory_context(
        user_id, '', {'schemaPaths': boundary_paths, 'constraintPaths': boundary_paths},
        limit=0, max_chars=budget // 2, candidates=candidates,
        is_blocked=lambda text: store._is_governed_memory_text(text, governance),
    )
    turns = []
    summary = ''
    if _valid_scope(user_id, conversation_id):
        turns = [asdict(item) for item in store._load_short_term(user_id, conversation_id)[-settings.memory_recall_limit:]]
        summary = str(store._load_summary_with_meta(user_id, conversation_id, allow_fallback=False).get('summary') or '')
    # Pending/forgotten/disabled facts must not leak back through conversation
    # history or a cached session summary after long-term retrieval is skipped.
    turns = [item for item in turns if not store._is_governed_memory_text(str(item.get('content') or ''), governance)]
    if store._is_governed_memory_text(summary, governance):
        summary = ''
    memory = {'short_term': turns, 'summary': summary, 'governance': governance,
              'structured': [], 'semantic': [], 'vector': [], 'retrieved': [],
              'profile_context': boundaries, 'profiles': boundaries.get('profiles') or [],
              'fragments': boundaries.get('fragments') or [],
              'used_memories': boundaries.get('usedMemories') or [],
              'memory_exclusions': boundaries.get('memoryExclusions') or {},
              'recalled_refs': store._build_recalled_memory_refs(boundaries, [])}
    store._compile_recall_context(memory, budget, budget * 3 // 4)
    return {'memory': memory, 'memory_context': store.format_memory_context(memory)}
