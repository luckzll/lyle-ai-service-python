"""add pending operation idempotency

Revision ID: 0012_pending_operation_idempotency
Revises: 0011_memory_fragments_and_user_profile
Create Date: 2026-07-03
"""

from alembic import op


revision = "0012_pending_operation_idempotency"
down_revision = "0011_memory_fragments_and_user_profile"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE agent_pending_operation ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(128)")
    op.execute(
        """
        UPDATE agent_pending_operation
        SET idempotency_key = 'legacy:' || id::text
        WHERE idempotency_key IS NULL OR idempotency_key = ''
        """
    )
    op.execute("ALTER TABLE agent_pending_operation ALTER COLUMN idempotency_key SET NOT NULL")
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_pending_operation_user_idempotency
        ON agent_pending_operation(user_id, idempotency_key)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_agent_pending_operation_user_idempotency")
    op.execute("ALTER TABLE agent_pending_operation DROP COLUMN IF EXISTS idempotency_key")
