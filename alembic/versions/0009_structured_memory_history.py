"""add structured memory history

Revision ID: 0009_structured_memory_history
Revises: 0008_prompt_templates
Create Date: 2026-06-30
"""

from alembic import op


revision = "0009_structured_memory_history"
down_revision = "0008_prompt_templates"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS structured_memory_history (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          memory_key VARCHAR(128) NOT NULL,
          old_value TEXT,
          new_value TEXT,
          action VARCHAR(32) NOT NULL,
          source TEXT NOT NULL DEFAULT '',
          confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0,
          status VARCHAR(32) NOT NULL DEFAULT 'active',
          created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_structured_memory_history_user_key ON structured_memory_history(user_id, memory_key, created_at DESC)"
    )


def downgrade() -> None:
    op.drop_table("structured_memory_history")
