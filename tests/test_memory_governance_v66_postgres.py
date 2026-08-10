import os
import uuid

import pytest

from app.core.config import settings
from app.core.database import close_postgres_pool, postgres_connection
from app.memory.fragments import create_memory_fragment
from app.memory.governance import create_memory_conflict
from app.memory.store import upsert_structured_memory


pytestmark = pytest.mark.skipif(
    not os.getenv("V66_TEST_POSTGRES_DSN"),
    reason="set V66_TEST_POSTGRES_DSN to a migrated disposable PostgreSQL database",
)


def test_postgres_semantic_conflict_dedupe_across_distinct_fragments(monkeypatch):
    user_id = f"v66-pg-dedupe-{uuid.uuid4()}"
    close_postgres_pool()
    monkeypatch.setattr(settings, "postgres_enabled", True)
    monkeypatch.setattr(settings, "postgres_dsn", os.environ["V66_TEST_POSTGRES_DSN"])
    monkeypatch.setattr(settings, "redis_enabled", False)
    monkeypatch.setattr(settings, "qdrant_enabled", False)
    monkeypatch.setattr(settings, "milvus_enabled", False)
    monkeypatch.setattr(settings, "memory_source_validation_enabled", False)
    try:
        upsert_structured_memory(user_id, "career", "产品经理", confidence=0.9, status="active")
        first_fragment = create_memory_fragment(
            user_id,
            source_text="我转岗做产品设计师了",
            normalized_text="产品设计师",
            category="work.current_role",
            conversation_id="conversation-a",
            source_message_id="message-a",
            confidence=0.9,
        )
        second_fragment = create_memory_fragment(
            user_id,
            source_text="另一个来源也说我转岗做产品设计师了",
            normalized_text="产品设计师",
            category="work.current_role",
            conversation_id="conversation-b",
            source_message_id="message-b",
            confidence=0.9,
        )

        first = create_memory_conflict(user_id, "work.current_role", int(first_fragment["id"]))
        second = create_memory_conflict(user_id, "work.current_role", int(second_fragment["id"]))

        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*), count(DISTINCT conflict_fingerprint) "
                    "FROM memory_conflict WHERE user_id = %s AND status = 'OPEN'",
                    (user_id,),
                )
                open_count, fingerprint_count = cur.fetchone()
        assert second["id"] == first["id"]
        assert open_count == 1
        assert fingerprint_count == 1
    finally:
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM memory_governance_event WHERE user_id = %s", (user_id,))
                cur.execute("DELETE FROM memory_conflict WHERE user_id = %s", (user_id,))
                cur.execute("DELETE FROM profile_evidence WHERE user_id = %s", (user_id,))
                cur.execute("DELETE FROM memory_fragment WHERE user_id = %s", (user_id,))
                cur.execute("DELETE FROM user_profile WHERE user_id = %s", (user_id,))
                cur.execute("DELETE FROM structured_memory_history WHERE user_id = %s", (user_id,))
                cur.execute("DELETE FROM structured_memory WHERE user_id = %s", (user_id,))
            conn.commit()
        close_postgres_pool()
