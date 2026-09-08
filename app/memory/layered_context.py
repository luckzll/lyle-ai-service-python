"""Small core profile is a read-only fact projection, not another memory store."""
from app.memory.profile_store import list_user_profiles
from app.memory.schema_builder import collect_inputs

# Deliberately excludes interests, all projects, and historical detail.
CORE_PATHS = (
    'interaction.boundaries', 'interaction.triggers',
    'preferences.communication_style', 'preferences.output_style',
    'identity.language', 'preferences.verbal_habits',
    'identity.name', 'work.current_role',
)


def core_profiles(user_id, *, allowed_paths=None):
    inputs = collect_inputs(user_id)
    profiles = {p['schemaPath']: p for p in list_user_profiles(user_id, limit=500)}
    allowed = set(allowed_paths) if allowed_paths is not None else set(CORE_PATHS)
    return [{**profiles[path], 'summary': inputs[path]['confirmedValue'], 'layer': 'core',
             'sourceCount': len({ref for source in inputs[path]['sources']
                                for ref in (source.get('evidenceIds') or [source.get('sourceRef') or source['id']])}),
             'sourceIds': [source['id'] for source in inputs[path]['sources']]}
            for path in CORE_PATHS if path in allowed and path in inputs]
