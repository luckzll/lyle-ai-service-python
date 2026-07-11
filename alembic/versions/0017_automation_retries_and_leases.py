"""add automation retries and leases

Revision ID: 0017_automation_retries_and_leases
Revises: 0016_workflow_conditions_and_references
Create Date: 2026-07-10
"""

from alembic import op


revision = "0017_automation_retries_and_leases"
down_revision = "0016_workflow_conditions_and_references"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE automation_rule ADD COLUMN IF NOT EXISTS max_retries INTEGER NOT NULL DEFAULT 0")
    op.execute("ALTER TABLE automation_rule ADD COLUMN IF NOT EXISTS retry_delay_seconds INTEGER NOT NULL DEFAULT 60")
    op.execute("ALTER TABLE automation_rule ADD COLUMN IF NOT EXISTS consecutive_failures INTEGER NOT NULL DEFAULT 0")
    op.execute("ALTER TABLE automation_rule ADD COLUMN IF NOT EXISTS notify_on_failure BOOLEAN NOT NULL DEFAULT true")
    op.execute("ALTER TABLE automation_rule ADD COLUMN IF NOT EXISTS lease_until TIMESTAMPTZ")
    op.execute("ALTER TABLE automation_run ADD COLUMN IF NOT EXISTS attempt INTEGER NOT NULL DEFAULT 1")
    op.execute("ALTER TABLE automation_run ADD COLUMN IF NOT EXISTS trigger_source VARCHAR(32) NOT NULL DEFAULT 'manual'")


def downgrade() -> None:
    op.execute("ALTER TABLE automation_run DROP COLUMN IF EXISTS trigger_source")
    op.execute("ALTER TABLE automation_run DROP COLUMN IF EXISTS attempt")
    op.execute("ALTER TABLE automation_rule DROP COLUMN IF EXISTS lease_until")
    op.execute("ALTER TABLE automation_rule DROP COLUMN IF EXISTS notify_on_failure")
    op.execute("ALTER TABLE automation_rule DROP COLUMN IF EXISTS consecutive_failures")
    op.execute("ALTER TABLE automation_rule DROP COLUMN IF EXISTS retry_delay_seconds")
    op.execute("ALTER TABLE automation_rule DROP COLUMN IF EXISTS max_retries")
