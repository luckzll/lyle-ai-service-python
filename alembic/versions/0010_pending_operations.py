"""add pending operations

Revision ID: 0010_pending_operations
Revises: 0009_structured_memory_history
Create Date: 2026-07-02
"""

from alembic import op


revision = "0010_pending_operations"
down_revision = "0009_structured_memory_history"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_pending_operation (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          conversation_id VARCHAR(128) NOT NULL DEFAULT '',
          tool_name VARCHAR(128) NOT NULL,
          operation_type VARCHAR(128) NOT NULL,
          risk_level VARCHAR(32) NOT NULL DEFAULT 'write',
          summary TEXT NOT NULL,
          payload JSONB NOT NULL,
          status VARCHAR(32) NOT NULL DEFAULT 'pending',
          result JSONB,
          source_text TEXT NOT NULL DEFAULT '',
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          confirmed_at TIMESTAMPTZ,
          rejected_at TIMESTAMPTZ
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_agent_pending_operation_user_status "
        "ON agent_pending_operation(user_id, status, updated_at DESC)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_agent_pending_operation_conversation "
        "ON agent_pending_operation(user_id, conversation_id, created_at DESC)"
    )


def downgrade() -> None:
    op.drop_table("agent_pending_operation")
