"""add V6.3 execution-result attention dismissal

Revision ID: 0026_v63_result_dismissal
Revises: 0025_v6_trusted_delegation
"""

from alembic import op


revision = "0026_v63_result_dismissal"
down_revision = "0025_v6_trusted_delegation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE execution_result_summary
          ADD COLUMN IF NOT EXISTS dismissed_at TIMESTAMPTZ;
        CREATE INDEX IF NOT EXISTS idx_execution_result_user_attention
          ON execution_result_summary(user_id, created_at DESC, id DESC)
          WHERE dismissed_at IS NULL
            AND status IN ('failed', 'partial', 'awaiting_confirmation');
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_execution_result_user_attention")
    op.execute("ALTER TABLE execution_result_summary DROP COLUMN IF EXISTS dismissed_at")
