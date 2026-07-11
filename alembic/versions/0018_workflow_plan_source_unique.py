"""make plan to workflow conversion idempotent

Revision ID: 0018_workflow_plan_source_unique
Revises: 0017_automation_retries_and_leases
Create Date: 2026-07-10
"""

from alembic import op


revision = "0018_workflow_plan_source_unique"
down_revision = "0017_automation_retries_and_leases"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_workflow_template_user_source_plan
        ON workflow_template(user_id, created_from_plan_id)
        WHERE created_from_plan_id IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_workflow_template_user_source_plan")
