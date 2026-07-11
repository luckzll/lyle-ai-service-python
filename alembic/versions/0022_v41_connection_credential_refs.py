"""add nullable credential references for v4.1 connections

Revision ID: 0022_v41_credentials
Revises: 0021_v41_execution
"""

from alembic import op


revision = "0022_v41_credentials"
down_revision = "0021_v41_execution"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE mcp_server ADD COLUMN IF NOT EXISTS secret_ref VARCHAR(512)")


def downgrade() -> None:
    op.execute("ALTER TABLE mcp_server DROP COLUMN IF EXISTS secret_ref")
