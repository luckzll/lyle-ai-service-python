"""add agent notifications

Revision ID: 0005_agent_notifications
Revises: 0004_agent_tasks
Create Date: 2026-06-30
"""

from alembic import op


revision = "0005_agent_notifications"
down_revision = "0004_agent_tasks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE agent_todo ADD COLUMN IF NOT EXISTS due_at TIMESTAMPTZ")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_notification (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          source_type VARCHAR(32) NOT NULL,
          source_id BIGINT NOT NULL,
          title VARCHAR(256) NOT NULL,
          message TEXT NOT NULL,
          status VARCHAR(32) NOT NULL DEFAULT 'unread',
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          read_at TIMESTAMPTZ,
          UNIQUE (user_id, source_type, source_id)
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_agent_notification_user_status ON agent_notification(user_id, status, created_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_agent_todo_user_due ON agent_todo(user_id, due_at)")


def downgrade() -> None:
    op.drop_table("agent_notification")
    op.execute("ALTER TABLE agent_todo DROP COLUMN IF EXISTS due_at")
