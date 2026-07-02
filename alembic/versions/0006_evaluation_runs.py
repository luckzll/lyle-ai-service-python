"""add evaluation run history

Revision ID: 0006_evaluation_runs
Revises: 0005_agent_notifications
Create Date: 2026-06-30
"""

from alembic import op


revision = "0006_evaluation_runs"
down_revision = "0005_agent_notifications"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS evaluation_run (
          id BIGSERIAL PRIMARY KEY,
          name VARCHAR(256) NOT NULL DEFAULT '',
          count INTEGER NOT NULL,
          average_score DOUBLE PRECISION NOT NULL,
          prompt_version VARCHAR(128) NOT NULL,
          model_provider VARCHAR(128) NOT NULL DEFAULT '',
          model_name VARCHAR(256) NOT NULL DEFAULT '',
          items JSONB NOT NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_evaluation_run_created ON evaluation_run(created_at DESC)")


def downgrade() -> None:
    op.drop_table("evaluation_run")
