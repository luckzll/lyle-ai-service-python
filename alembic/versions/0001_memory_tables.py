"""create memory tables

Revision ID: 0001_memory_tables
Revises:
Create Date: 2026-06-29
"""

from alembic import op


revision = "0001_memory_tables"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_summary (
          user_id VARCHAR(128) NOT NULL,
          conversation_id VARCHAR(128) NOT NULL,
          summary TEXT NOT NULL,
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          PRIMARY KEY (user_id, conversation_id)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS structured_memory (
          user_id VARCHAR(128) NOT NULL,
          memory_key VARCHAR(128) NOT NULL,
          memory_value TEXT NOT NULL,
          source TEXT NOT NULL,
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          PRIMARY KEY (user_id, memory_key)
        )
        """
    )


def downgrade() -> None:
    op.drop_table("structured_memory")
    op.drop_table("memory_summary")
