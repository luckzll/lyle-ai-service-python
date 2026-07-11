"""add automation event trigger state

Revision ID: 0019_automation_event_triggers
Revises: 0018_workflow_plan_source_unique
Create Date: 2026-07-10
"""

from alembic import op


revision = "0019_automation_event_triggers"
down_revision = "0018_workflow_plan_source_unique"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE automation_rule ADD COLUMN IF NOT EXISTS last_event_key VARCHAR(512)")
    op.execute("ALTER TABLE automation_rule ADD COLUMN IF NOT EXISTS last_event_at TIMESTAMPTZ")
    op.execute("ALTER TABLE automation_rule ADD COLUMN IF NOT EXISTS pending_event_key VARCHAR(512)")
    op.execute("ALTER TABLE automation_rule ADD COLUMN IF NOT EXISTS pending_event_input JSONB")
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_automation_rule_event_scan
        ON automation_rule(status, trigger_type, pending_event_key)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_automation_rule_event_scan")
    op.execute("ALTER TABLE automation_rule DROP COLUMN IF EXISTS pending_event_input")
    op.execute("ALTER TABLE automation_rule DROP COLUMN IF EXISTS pending_event_key")
    op.execute("ALTER TABLE automation_rule DROP COLUMN IF EXISTS last_event_at")
    op.execute("ALTER TABLE automation_rule DROP COLUMN IF EXISTS last_event_key")
