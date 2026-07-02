"""add agent todo and calendar tables

Revision ID: 0004_agent_tasks
Revises: 0003_tool_audit_log
Create Date: 2026-06-30
"""

from alembic import op


revision = "0004_agent_tasks"
down_revision = "0003_tool_audit_log"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_todo (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          conversation_id VARCHAR(128) NOT NULL DEFAULT '',
          title VARCHAR(256) NOT NULL,
          status VARCHAR(32) NOT NULL DEFAULT 'open',
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_calendar_event (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          conversation_id VARCHAR(128) NOT NULL DEFAULT '',
          title VARCHAR(256) NOT NULL,
          starts_at TIMESTAMPTZ,
          status VARCHAR(32) NOT NULL DEFAULT 'scheduled',
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_agent_todo_user_status ON agent_todo(user_id, status, updated_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_agent_calendar_user_time ON agent_calendar_event(user_id, starts_at)")


def downgrade() -> None:
    op.drop_table("agent_calendar_event")
    op.drop_table("agent_todo")
