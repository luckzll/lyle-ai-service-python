"""add workflow conditions and output references

Revision ID: 0016_workflow_conditions_and_references
Revises: 0015_automation_input_and_run_index
Create Date: 2026-07-10
"""

from alembic import op


revision = "0016_workflow_conditions_and_references"
down_revision = "0015_automation_input_and_run_index"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE workflow_step_template ADD COLUMN IF NOT EXISTS output_key VARCHAR(128)")
    op.execute("ALTER TABLE workflow_step_template ADD COLUMN IF NOT EXISTS condition_expr JSONB")


def downgrade() -> None:
    op.execute("ALTER TABLE workflow_step_template DROP COLUMN IF EXISTS condition_expr")
    op.execute("ALTER TABLE workflow_step_template DROP COLUMN IF EXISTS output_key")
