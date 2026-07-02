"""add prompt templates

Revision ID: 0008_prompt_templates
Revises: 0007_evaluation_sets
Create Date: 2026-06-30
"""

from alembic import op


revision = "0008_prompt_templates"
down_revision = "0007_evaluation_sets"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS prompt_template (
          id BIGSERIAL PRIMARY KEY,
          name VARCHAR(128) NOT NULL,
          version VARCHAR(128) NOT NULL,
          system_prompt TEXT NOT NULL,
          status VARCHAR(32) NOT NULL DEFAULT 'draft',
          traffic_percent INTEGER NOT NULL DEFAULT 100,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_prompt_template_status ON prompt_template(status)")
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_prompt_template_version ON prompt_template(version)")


def downgrade() -> None:
    op.drop_table("prompt_template")
