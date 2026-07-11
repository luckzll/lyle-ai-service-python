"""add v4.1 execution result summaries and confirmation inbox

Revision ID: 0021_v41_execution
Revises: 0020_v41_experience
"""

from alembic import op


revision = "0021_v41_execution"
down_revision = "0020_v41_experience"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS execution_result_summary (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          source_type VARCHAR(32) NOT NULL,
          source_id VARCHAR(128) NOT NULL,
          project_id BIGINT,
          template_id VARCHAR(64),
          status VARCHAR(32) NOT NULL,
          title VARCHAR(256) NOT NULL,
          conclusion TEXT NOT NULL,
          completed_items_json JSONB NOT NULL DEFAULT '[]'::jsonb,
          risks_json JSONB NOT NULL DEFAULT '[]'::jsonb,
          recommended_actions_json JSONB NOT NULL DEFAULT '[]'::jsonb,
          data_scope_json JSONB NOT NULL DEFAULT '[]'::jsonb,
          duration_ms BIGINT,
          trace_ref VARCHAR(256),
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          UNIQUE(source_type, source_id)
        );
        CREATE INDEX IF NOT EXISTS idx_execution_result_user_created
          ON execution_result_summary(user_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_execution_result_user_status_created
          ON execution_result_summary(user_id, status, created_at DESC);
        CREATE TABLE IF NOT EXISTS confirmation_inbox_item (
          pending_operation_id BIGINT PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          project_id BIGINT,
          source_run_id VARCHAR(128),
          priority INTEGER NOT NULL DEFAULT 0,
          expires_at TIMESTAMPTZ,
          display_summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS idx_confirmation_inbox_user_priority
          ON confirmation_inbox_item(user_id, priority DESC, created_at ASC);
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS confirmation_inbox_item")
    op.execute("DROP TABLE IF EXISTS execution_result_summary")
