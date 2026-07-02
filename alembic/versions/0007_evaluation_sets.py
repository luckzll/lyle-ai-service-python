"""add evaluation sets

Revision ID: 0007_evaluation_sets
Revises: 0006_evaluation_runs
Create Date: 2026-06-30
"""

from alembic import op


revision = "0007_evaluation_sets"
down_revision = "0006_evaluation_runs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS evaluation_set (
          id BIGSERIAL PRIMARY KEY,
          name VARCHAR(256) NOT NULL,
          description TEXT NOT NULL DEFAULT '',
          cases JSONB NOT NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_evaluation_set_updated ON evaluation_set(updated_at DESC)")
    op.execute("ALTER TABLE evaluation_run ADD COLUMN IF NOT EXISTS evaluation_set_id BIGINT")
    op.execute("ALTER TABLE evaluation_run ADD COLUMN IF NOT EXISTS badcase_count INTEGER NOT NULL DEFAULT 0")


def downgrade() -> None:
    op.execute("ALTER TABLE evaluation_run DROP COLUMN IF EXISTS badcase_count")
    op.execute("ALTER TABLE evaluation_run DROP COLUMN IF EXISTS evaluation_set_id")
    op.drop_table("evaluation_set")
