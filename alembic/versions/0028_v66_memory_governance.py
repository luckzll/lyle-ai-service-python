"""add v6.6 memory governance

Revision ID: 0028_v66_memory_governance
Revises: 0027_v64_usage_event_outbox
Create Date: 2026-08-10
"""

from alembic import op


revision = "0028_v66_memory_governance"
down_revision = "0027_v64_usage_event_outbox"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE memory_fragment ADD COLUMN IF NOT EXISTS source_message_id VARCHAR(128)")
    op.execute("ALTER TABLE memory_fragment ADD COLUMN IF NOT EXISTS source_chunk_id VARCHAR(256)")
    op.execute("ALTER TABLE memory_fragment ADD COLUMN IF NOT EXISTS source_created_at TIMESTAMPTZ")
    op.execute("ALTER TABLE user_profile ADD COLUMN IF NOT EXISTS evidence_confidence DOUBLE PRECISION")
    op.execute("UPDATE user_profile SET evidence_confidence = confidence WHERE evidence_confidence IS NULL")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_conflict (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          schema_path VARCHAR(256) NOT NULL,
          status VARCHAR(32) NOT NULL DEFAULT 'OPEN',
          current_profile_id BIGINT REFERENCES user_profile(id) ON DELETE SET NULL,
          candidate_fragment_id BIGINT REFERENCES memory_fragment(id) ON DELETE SET NULL,
          conflict_fingerprint VARCHAR(128) NOT NULL,
          resolution VARCHAR(32),
          resolved_value TEXT,
          version BIGINT NOT NULL DEFAULT 1,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          resolved_at TIMESTAMPTZ
        )
        """
    )
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uk_memory_conflict_open "
        "ON memory_conflict(user_id, conflict_fingerprint) WHERE status = 'OPEN'"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_conflict_user_status "
        "ON memory_conflict(user_id, status, updated_at DESC)"
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_governance_event (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          subject_type VARCHAR(32) NOT NULL,
          subject_id VARCHAR(256) NOT NULL,
          action VARCHAR(64) NOT NULL,
          before_snapshot_json JSONB NOT NULL DEFAULT '{}'::jsonb,
          after_snapshot_json JSONB NOT NULL DEFAULT '{}'::jsonb,
          reason_code VARCHAR(64) NOT NULL DEFAULT '',
          source_ref TEXT NOT NULL DEFAULT '',
          client_request_id VARCHAR(64) NOT NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          UNIQUE(user_id, client_request_id)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_governance_event_user_created "
        "ON memory_governance_event(user_id, created_at DESC)"
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_purge_job (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          memory_id VARCHAR(256) NOT NULL,
          status VARCHAR(32) NOT NULL DEFAULT 'PENDING',
          attempt_count INTEGER NOT NULL DEFAULT 0,
          next_attempt_at TIMESTAMPTZ,
          step_state_json JSONB NOT NULL DEFAULT '{}'::jsonb,
          client_request_id VARCHAR(64) NOT NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          completed_at TIMESTAMPTZ,
          UNIQUE(user_id, client_request_id)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_purge_job_claim "
        "ON memory_purge_job(status, next_attempt_at, created_at)"
    )


def downgrade() -> None:
    op.drop_table("memory_purge_job")
    op.drop_table("memory_governance_event")
    op.drop_table("memory_conflict")
    op.execute("ALTER TABLE user_profile DROP COLUMN IF EXISTS evidence_confidence")
    op.execute("ALTER TABLE memory_fragment DROP COLUMN IF EXISTS source_created_at")
    op.execute("ALTER TABLE memory_fragment DROP COLUMN IF EXISTS source_chunk_id")
    op.execute("ALTER TABLE memory_fragment DROP COLUMN IF EXISTS source_message_id")
