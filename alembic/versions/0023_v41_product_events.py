"""add privacy-preserving V4.1 product events

Revision ID: 0023_v41_product_events
Revises: 0022_v41_credentials
"""

from alembic import op


revision = "0023_v41_product_events"
down_revision = "0022_v41_credentials"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS experience_product_event (
          id BIGSERIAL PRIMARY KEY,
          anonymous_user_id VARCHAR(64) NOT NULL,
          event_type VARCHAR(64) NOT NULL,
          template_id VARCHAR(64),
          status VARCHAR(32),
          duration_ms BIGINT,
          metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS idx_experience_product_event_type_created
          ON experience_product_event(event_type, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_experience_product_event_user_created
          ON experience_product_event(anonymous_user_id, created_at DESC);
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS experience_product_event")
