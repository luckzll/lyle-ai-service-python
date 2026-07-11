"""add automation input and run index

Revision ID: 0015_automation_input_and_run_index
Revises: 0014_v4_mcp_cli_workflows_automation
Create Date: 2026-07-09
"""

from alembic import op


revision = "0015_automation_input_and_run_index"
down_revision = "0014_v4_mcp_cli_workflows_automation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE automation_rule ADD COLUMN IF NOT EXISTS input JSONB")
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_automation_rule_due
        ON automation_rule(status, next_run_at)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_automation_run_automation_created
        ON automation_run(automation_id, created_at DESC)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_automation_run_automation_created")
    op.execute("DROP INDEX IF EXISTS idx_automation_rule_due")
    op.execute("ALTER TABLE automation_rule DROP COLUMN IF EXISTS input")
