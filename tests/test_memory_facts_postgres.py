"""Opt-in contract against a disposable, migrated PostgreSQL database only."""
import os
from uuid import uuid4
import pytest

from app.core.config import settings
from app.core.database import close_postgres_pool, postgres_connection

pytestmark = pytest.mark.skipif(not os.getenv('MEMORY_FACTS_TEST_POSTGRES_DSN'),
    reason='Requires MEMORY_FACTS_TEST_POSTGRES_DSN pointing to a disposable PostgreSQL database')


def test_postgres_fact_write_and_queue_marker_are_atomic(monkeypatch):
    from app.memory import fact_store, schema_store, profile_store
    from app.memory.store import upsert_structured_memory
    user = 'fact-pg-contract-' + uuid4().hex
    close_postgres_pool()
    monkeypatch.setattr(settings, 'postgres_dsn', os.environ['MEMORY_FACTS_TEST_POSTGRES_DSN'])
    monkeypatch.setattr(settings, 'postgres_enabled', True)
    monkeypatch.setattr(settings, 'memory_schema_enabled', True)
    monkeypatch.setattr(settings, 'memory_facts_enabled', True)
    monkeypatch.setattr(fact_store, '_ready', False)
    monkeypatch.setattr(schema_store, '_ready', False)
    monkeypatch.setattr(profile_store, '_postgres_ready', False)
    try:
        upsert_structured_memory(user, 'career', '工程师', confidence=.95)
        with postgres_connection() as conn:
            assert conn.execute('SELECT count(*) FROM memory_fact WHERE user_id=%s', (user,)).fetchone()[0] == 1
            assert conn.execute('SELECT value FROM user_profile WHERE user_id=%s', (user,)).fetchone()[0] == ''
            assert conn.execute("SELECT state_json->>'status' FROM memory_schema_state WHERE user_id=%s", (user,)).fetchone()[0] == 'pending'
        before = fact_store.list_facts(user)
        with pytest.raises(RuntimeError):
            with fact_store._edit(user) as state:
                state['facts'].clear()
                raise RuntimeError('rollback')
        assert fact_store.list_facts(user) == before
    finally:
        # Explicit test-user scope only; no schema/database drops.
        with postgres_connection() as conn:
            for table in ('profile_evidence', 'memory_fragment', 'memory_fact', 'memory_fact_owner',
                          'memory_schema_state', 'structured_memory_history', 'structured_memory', 'user_profile'):
                conn.execute(f'DELETE FROM {table} WHERE user_id=%s', (user,))
            conn.commit()
        close_postgres_pool()
