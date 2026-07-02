"""add structured memory state

Revision ID: 0002_structured_memory_state
Revises: 0001_memory_tables
Create Date: 2026-06-30
"""

from alembic import op


revision = "0002_structured_memory_state"
down_revision = "0001_memory_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE structured_memory ADD COLUMN IF NOT EXISTS status VARCHAR(32) NOT NULL DEFAULT 'active'")
    op.execute("ALTER TABLE structured_memory ADD COLUMN IF NOT EXISTS confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0")
    op.execute("ALTER TABLE structured_memory ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ")


def downgrade() -> None:
    op.execute("ALTER TABLE structured_memory DROP COLUMN IF EXISTS expires_at")
    op.execute("ALTER TABLE structured_memory DROP COLUMN IF EXISTS confidence")
    op.execute("ALTER TABLE structured_memory DROP COLUMN IF EXISTS status")
