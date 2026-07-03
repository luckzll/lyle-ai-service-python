"""add memory fragments and user profile

Revision ID: 0011_memory_fragments_and_user_profile
Revises: 0010_pending_operations
Create Date: 2026-07-03
"""

from alembic import op


revision = "0011_memory_fragments_and_user_profile"
down_revision = "0010_pending_operations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_fragment (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          conversation_id VARCHAR(128),
          source_type VARCHAR(32) NOT NULL,
          source_id VARCHAR(256) NOT NULL DEFAULT '',
          source_ref TEXT NOT NULL DEFAULT '',
          source_text TEXT NOT NULL,
          normalized_text TEXT NOT NULL,
          category VARCHAR(128) NOT NULL,
          confidence DOUBLE PRECISION NOT NULL DEFAULT 0.8,
          status VARCHAR(32) NOT NULL DEFAULT 'active',
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_memory_fragment_user_status ON memory_fragment(user_id, status, updated_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_memory_fragment_user_category ON memory_fragment(user_id, category)")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS user_profile (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          schema_path VARCHAR(256) NOT NULL,
          value TEXT NOT NULL,
          summary TEXT NOT NULL DEFAULT '',
          confidence DOUBLE PRECISION NOT NULL DEFAULT 0.8,
          status VARCHAR(32) NOT NULL DEFAULT 'active',
          last_used_at TIMESTAMPTZ,
          hit_count BIGINT NOT NULL DEFAULT 0,
          correction_count BIGINT NOT NULL DEFAULT 0,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          UNIQUE(user_id, schema_path)
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_user_profile_user_status ON user_profile(user_id, status, updated_at DESC)")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS profile_evidence (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          profile_id BIGINT NOT NULL REFERENCES user_profile(id) ON DELETE CASCADE,
          fragment_id BIGINT NOT NULL REFERENCES memory_fragment(id) ON DELETE CASCADE,
          relation VARCHAR(32) NOT NULL DEFAULT 'support',
          confidence DOUBLE PRECISION NOT NULL DEFAULT 0.8,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          UNIQUE(profile_id, fragment_id, relation)
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_profile_evidence_profile ON profile_evidence(profile_id, created_at DESC)")


def downgrade() -> None:
    op.drop_table("profile_evidence")
    op.drop_table("user_profile")
    op.drop_table("memory_fragment")
