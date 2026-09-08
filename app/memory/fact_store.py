"""Authoritative confirmed facts. Profiles are projections, never a second writer.

Legacy profile rows remain as ID anchors for evidence and existing UI links.
Per-user row locking serializes migration and commands; fact changes and schema
invalidation commit together in PostgreSQL. No destructive legacy migration.
"""
from contextlib import contextmanager
from copy import deepcopy
import re
from threading import RLock
from time import time
from uuid import uuid4

from psycopg.types.json import Jsonb

from app.core.config import settings
from app.core.database import postgres_connection
from app.memory.profile_schema import PROFILE_SCHEMA, is_multi_value_path, schema_path_for_memory

_lock = RLock()
_states = {}
_ready = False
DDL = """
CREATE TABLE IF NOT EXISTS memory_fact_owner (
 user_id TEXT PRIMARY KEY, metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE TABLE IF NOT EXISTS memory_fact (
 fact_id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES memory_fact_owner(user_id),
 schema_path TEXT NOT NULL, data JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_fact_user_path ON memory_fact(user_id, schema_path);
"""


def enabled():
    return settings.memory_facts_enabled and settings.memory_schema_enabled


def _ensure():
    global _ready
    if not _ready:
        with _lock:
            if not _ready:
                from app.memory.schema_store import _ensure_schema
                _ensure_schema()
                with postgres_connection() as conn:
                    conn.execute(DDL)
                    conn.commit()
                _ready = True


@contextmanager
def _edit(user_id):
    if settings.postgres_enabled:
        _ensure()
        with postgres_connection() as conn:
            with conn.transaction():
                conn.execute("INSERT INTO memory_fact_owner(user_id) VALUES (%s) ON CONFLICT DO NOTHING", (user_id,))
                meta = conn.execute("SELECT metadata FROM memory_fact_owner WHERE user_id=%s FOR UPDATE", (user_id,)).fetchone()[0]
                rows = conn.execute("SELECT fact_id,data FROM memory_fact WHERE user_id=%s", (user_id,)).fetchall()
                state = {**meta, 'facts': {row[0]: row[1] for row in rows}}
                before = deepcopy(state)
                yield state
                for fact_id, fact in state['facts'].items():
                    if before['facts'].get(fact_id) != fact:
                        conn.execute("INSERT INTO memory_fact(fact_id,user_id,schema_path,data) VALUES (%s,%s,%s,%s) ON CONFLICT(fact_id) DO UPDATE SET data=EXCLUDED.data", (fact_id, user_id, fact['schemaPath'], Jsonb(fact)))
                for fact_id in before['facts'].keys() - state['facts'].keys():
                    conn.execute("DELETE FROM memory_fact WHERE user_id=%s AND fact_id=%s", (user_id, fact_id))
                if state != before:
                    conn.execute("UPDATE memory_fact_owner SET metadata=%s WHERE user_id=%s", (Jsonb({k: v for k, v in state.items() if k != 'facts'}), user_id))
                    # Durable outbox-equivalent queue marker in the SAME transaction.
                    from app.memory.schema_store import _empty
                    conn.execute("INSERT INTO memory_schema_state(user_id,state_json) VALUES (%s,%s) ON CONFLICT DO NOTHING", (user_id, Jsonb(_empty())))
                    queue = conn.execute("SELECT state_json FROM memory_schema_state WHERE user_id=%s FOR UPDATE", (user_id,)).fetchone()[0]
                    queue.update(revision=queue['revision'] + 1, desiredHash='', errorCode='', attempts=0,
                                 status='building' if queue['status'] == 'building' else 'pending')
                    conn.execute("UPDATE memory_schema_state SET state_json=%s,updated_at=now() WHERE user_id=%s", (Jsonb(queue), user_id))
    else:
        with _lock:
            state = deepcopy(_states.get(user_id, {'facts': {}}))
            before = deepcopy(state)
            yield state
            _states[user_id] = state
            if before != state:
                from app.memory.schema_store import request_build
                request_build(user_id)


def _parts(path, value):
    # Never split natural-language 'and/和/与': only explicit list punctuation.
    values = re.split(r'[、,，;；]+', value) if is_multi_value_path(path) else [value]
    seen, result = set(), []
    for value in values:
        value = value.strip()
        key = re.sub(r'\s+', ' ', value).casefold()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _new(path, value, profile_id, *, confidence=.9, status='active', source='confirmed', updated_at=None, expires_at=None):
    return {'factId': str(uuid4()), 'schemaPath': path, 'profileId': profile_id,
            'value': value, 'confidence': confidence, 'status': status, 'revision': 1,
            'source': source, 'updatedAt': updated_at or time(), 'expiresAt': expires_at}


def _anchor(user_id, path):
    from app.memory import profile_store as profiles
    if settings.postgres_enabled:
        profiles._ensure_postgres_schema()
        with postgres_connection() as conn:
            row = conn.execute("""INSERT INTO user_profile(user_id,schema_path,value,summary,confidence,status)
                VALUES (%s,%s,'','',0.9,'active') ON CONFLICT(user_id,schema_path)
                DO UPDATE SET schema_path=EXCLUDED.schema_path RETURNING id""", (user_id, path)).fetchone()
            conn.commit()
        return int(row[0])
    with profiles._lock:
        items = profiles._profiles.setdefault(user_id, {})
        if path not in items:
            items[path] = profiles.UserProfile(profiles._next_id, user_id, path, '', '', .9, 'active', None, 0, 0, time(), time())
            profiles._next_id += 1
        return items[path].id


def _bootstrap(user_id, state):
    if state.get('migrated'):
        return
    from app.memory.profile_store import _legacy_list_user_profiles
    from app.memory.store import _legacy_load_all_structured
    profiles = _legacy_list_user_profiles(user_id, limit=500)
    legacy = _legacy_load_all_structured(user_id)
    report = {'imported': 0, 'conflicts': [], 'archivedLegacyRows': len(legacy),
              'unmappedKeys': sorted({item.key for item in legacy if not item.key.startswith('pending:')
                                     and schema_path_for_memory(item.key, item.value) not in PROFILE_SCHEMA})}
    paths = set()
    for profile in profiles:
        path = profile['schemaPath']
        if path not in PROFILE_SCHEMA or not profile.get('value'):
            continue
        paths.add(path)
        same_path = [item for item in legacy if not item.key.startswith('pending:') and schema_path_for_memory(item.key, item.value) == path]
        # Preserve existing governance decisions. Ambiguous active values require review.
        disagree = profile['status'] == 'active' and any(item.status == 'active' and item.value not in profile['value'] and profile['value'] not in item.value for item in same_path)
        if disagree:
            report['conflicts'].append(path)
        for value in _parts(path, profile['value']):
            expired = [item.expires_at for item in same_path if item.expires_at is not None and value in item.value]
            fact = _new(path, value, profile['id'], confidence=profile['confidence'],
                        status='conflicted' if disagree else profile['status'], source='legacy_profile',
                        expires_at=min(expired) if expired else None)
            fact['sourceUpdatedAt'] = profile['updatedAt']
            state['facts'][fact['factId']] = fact
            report['imported'] += 1
    for item in legacy:
        path = schema_path_for_memory(item.key, item.value)
        if item.key.startswith('pending:') or item.status == 'pending' or path not in PROFILE_SCHEMA or path in paths:
            continue
        # Legacy-only identities are imported conservatively; conflicting aliases are quarantined.
        peers = [other for other in legacy if not other.key.startswith('pending:') and schema_path_for_memory(other.key, other.value) == path]
        conflict = len({other.value for other in peers if other.status == 'active'}) > 1 and not is_multi_value_path(path)
        if conflict:
            report['conflicts'].append(path)
        for peer in peers:
            for value in _parts(path, peer.value):
                fact = _new(path, value, _anchor(user_id, path), confidence=peer.confidence,
                            status='conflicted' if conflict else peer.status, source='legacy_structured',
                            updated_at=peer.updated_at, expires_at=peer.expires_at)
                if not any(f['schemaPath'] == path and f['value'] == value for f in state['facts'].values()):
                    state['facts'][fact['factId']] = fact
                    report['imported'] += 1
        paths.add(path)
    state.update(migrated=True, migrationReport=report)


def list_facts(user_id, *, include_history=False):
    with _edit(user_id) as state:
        _bootstrap(user_id, state)
        result = deepcopy(list(state['facts'].values()))
    for fact in result:
        if fact['status'] == 'active' and fact.get('expiresAt') is not None and fact['expiresAt'] <= time():
            fact['status'] = 'expired'
    return [fact for fact in result if include_history or fact['status'] != 'superseded']


def migration_report(user_id):
    list_facts(user_id)
    with _edit(user_id) as state:
        report = deepcopy(state['migrationReport'])
        report['originalConflicts'] = report['conflicts']
        report['conflicts'] = [path for path in report['conflicts'] if any(
            f['schemaPath'] == path and f['status'] == 'conflicted' for f in state['facts'].values())]
        return report


def project_profiles(user_id, *, status=None, limit=200):
    from app.memory.profile_store import _legacy_list_user_profiles
    anchors = {item['id']: item for item in _legacy_list_user_profiles(user_id, limit=500)}
    grouped = {}
    for fact in list_facts(user_id):
        grouped.setdefault(fact['schemaPath'], []).append(fact)
    result = []
    for path, facts in grouped.items():
        active = [fact for fact in facts if fact['status'] == 'active']
        selected = active or facts
        state = 'conflicted' if any(fact['status'] == 'conflicted' for fact in facts) else 'active' if active else selected[0]['status']
        if status and state != status:
            continue
        anchor = anchors.get(selected[0]['profileId'], {})
        definition = PROFILE_SCHEMA[path]
        confidence = min(float(fact['confidence']) for fact in selected)
        reference = max(float(anchor.get('lastUsedAt') or 0), max(fact['updatedAt'] for fact in selected))
        confidence = max(0, confidence - int(max(0, time() - reference) // (30 * 86400)) * .01)
        value = '、'.join(dict.fromkeys(fact['value'] for fact in selected))
        result.append({**anchor, 'id': selected[0]['profileId'], 'userId': user_id, 'schemaPath': path,
                       'value': value, 'summary': value, 'status': state, 'confidence': confidence,
                       'label': definition.label, 'group': definition.group, 'multiValue': definition.multi_value,
                       'sensitive': definition.sensitive, 'updatedAt': max(f['updatedAt'] for f in selected),
                       'factIds': [f['factId'] for f in selected], 'factRevisions': {f['factId']: f['revision'] for f in selected}})
    return sorted(result, key=lambda item: item['updatedAt'], reverse=True)[:max(1, min(limit, 500))]


def write_slot(user_id, path, value=None, *, replace=False, status='active', confidence=.9, expires_at=None):
    if path not in PROFILE_SCHEMA:
        raise ValueError('Unknown memory schema path')
    with _edit(user_id) as state:
        _bootstrap(user_id, state)
        facts = [f for f in state['facts'].values() if f['schemaPath'] == path and f['status'] != 'superseded']
        if value is None:
            for fact in facts:
                if fact['status'] != status:
                    fact.update(status=status, revision=fact['revision'] + 1, updatedAt=time())
        else:
            values = _parts(path, value)
            if not values:
                raise ValueError('Memory value cannot be empty')
            profile_id = facts[0]['profileId'] if facts else _anchor(user_id, path)
            incoming_keys = {item.casefold() for item in values}
            for fact in facts:
                if (replace or not is_multi_value_path(path)) and fact['value'].casefold() not in incoming_keys:
                    fact.update(status='superseded', revision=fact['revision'] + 1, updatedAt=time())
            for item in values:
                existing = next((f for f in facts if f['value'].strip().casefold() == item.casefold() and f['status'] != 'superseded'), None)
                if existing:
                    next_confidence = min(.95, max(existing['confidence'], confidence))
                    if existing['status'] != status or existing.get('expiresAt') != expires_at or existing['confidence'] != next_confidence:
                        existing.update(status=status, confidence=next_confidence, expiresAt=expires_at,
                                        revision=existing['revision'] + 1, updatedAt=time())
                else:
                    fact = _new(path, item, profile_id, confidence=min(.95, confidence), status=status, expires_at=expires_at)
                    state['facts'][fact['factId']] = fact
    return next((p for p in project_profiles(user_id, limit=500) if p['schemaPath'] == path), {})


def purge_slot(user_id, path):
    with _edit(user_id) as state:
        _bootstrap(user_id, state)
        ids = [key for key, fact in state['facts'].items() if fact['schemaPath'] == path]
        for key in ids:
            del state['facts'][key]
    return bool(ids)


class FactConflict(ValueError):
    pass


def revise_fact(user_id, fact_id, *, expected_revision, status=None, value=None):
    """Granular governance: editing one list member must not replace its siblings."""
    with _edit(user_id) as state:
        _bootstrap(user_id, state)
        fact = state['facts'].get(fact_id)
        if not fact:
            raise KeyError('MEMORY_FACT_NOT_FOUND')
        if fact['revision'] != expected_revision or fact['status'] == 'superseded':
            raise FactConflict('MEMORY_FACT_REVISION_CONFLICT')
        if fact['status'] == 'conflicted' and status == 'active':
            raise FactConflict('RESOLVE_MEMORY_CONFLICT_FIRST')
        before_status = fact['status']
        if value is not None:
            if not value.strip() or len(value) > 4000:
                raise ValueError('INVALID_MEMORY_FACT_VALUE')
            value = value.strip()
            if any(f['factId'] != fact_id and f['schemaPath'] == fact['schemaPath'] and f['status'] == 'active'
                   and f['value'].casefold() == value.casefold() for f in state['facts'].values()):
                raise FactConflict('MEMORY_FACT_DUPLICATE_VALUE')
            previous = deepcopy(fact)
            previous.update(factId=str(uuid4()), status='superseded', supersededBy=fact_id)
            state['facts'][previous['factId']] = previous
            fact.update(value=value, confidence=.95, source='manual:correction')
        if status is not None:
            if status not in {'active', 'disabled', 'forgotten'}:
                raise ValueError('INVALID_MEMORY_FACT_STATUS')
            fact['status'] = status
        fact.update(revision=fact['revision'] + 1, updatedAt=time())
        state['events'] = [*state.get('events', []), {'factId': fact_id, 'revision': fact['revision'],
            'action': 'correct' if value is not None else 'status', 'beforeStatus': before_status,
            'afterStatus': fact['status'], 'at': fact['updatedAt']}][-1000:]
        return deepcopy(fact)
