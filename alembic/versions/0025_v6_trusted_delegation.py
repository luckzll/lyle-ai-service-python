"""add V6 trusted delegation compensation and project-scoped todos

Revision ID: 0025_v6_trusted_delegation
Revises: 0024_v42_result_indexes
Create Date: 2026-07-16
"""

from alembic import op


revision = "0025_v6_trusted_delegation"
down_revision = "0024_v42_result_indexes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE agent_todo ADD COLUMN IF NOT EXISTS project_id BIGINT")
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_agent_todo_user_project_status
        ON agent_todo(user_id, project_id, status, updated_at DESC)
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_delegated_compensation (
          snapshot_ref VARCHAR(512) PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          tool_ref VARCHAR(255) NOT NULL,
          resource_ref VARCHAR(512) NOT NULL,
          payload JSONB NOT NULL,
          status VARCHAR(32) NOT NULL,
          expires_at TIMESTAMPTZ NOT NULL,
          compensation_key VARCHAR(128),
          created_at TIMESTAMPTZ NOT NULL,
          updated_at TIMESTAMPTZ NOT NULL,
          compensated_at TIMESTAMPTZ
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_agent_delegated_compensation_user_status
        ON agent_delegated_compensation(user_id, status, expires_at)
        """
    )


def downgrade() -> None:
    op.drop_table("agent_delegated_compensation")
    op.drop_index("idx_agent_todo_user_project_status", table_name="agent_todo")
    op.drop_column("agent_todo", "project_id")
