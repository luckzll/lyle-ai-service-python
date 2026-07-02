"""add tool audit log

Revision ID: 0003_tool_audit_log
Revises: 0002_structured_memory_state
Create Date: 2026-06-30
"""

from alembic import op


revision = "0003_tool_audit_log"
down_revision = "0002_structured_memory_state"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS tool_audit_log (
          id BIGSERIAL PRIMARY KEY,
          tool_name VARCHAR(128) NOT NULL,
          user_id VARCHAR(128) NOT NULL,
          conversation_id VARCHAR(128) NOT NULL DEFAULT '',
          ok BOOLEAN NOT NULL,
          elapsed_ms INTEGER NOT NULL,
          payload JSONB NOT NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_tool_audit_user_created ON tool_audit_log(user_id, created_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_tool_audit_tool_created ON tool_audit_log(tool_name, created_at DESC)")


def downgrade() -> None:
    op.drop_table("tool_audit_log")
