"""add v4.1 onboarding and automation drafts

Revision ID: 0020_v41_experience
Revises: 0019_automation_event_triggers
"""

from alembic import op


revision = "0020_v41_experience"
down_revision = "0019_automation_event_triggers"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS experience_onboarding_state (
          user_id VARCHAR(128) PRIMARY KEY,
          first_opened_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          completed_at TIMESTAMPTZ,
          skipped_at TIMESTAMPTZ,
          selected_project_id BIGINT,
          selected_workspace_id BIGINT,
          first_run_id BIGINT,
          first_success_at TIMESTAMPTZ,
          last_step VARCHAR(64) NOT NULL DEFAULT 'welcome',
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE TABLE IF NOT EXISTS automation_draft (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          source VARCHAR(32) NOT NULL,
          template_id VARCHAR(64) NOT NULL,
          template_version INTEGER NOT NULL,
          title VARCHAR(256) NOT NULL,
          user_goal TEXT,
          project_id BIGINT,
          workspace_id BIGINT,
          schedule_expr VARCHAR(128) NOT NULL,
          timezone VARCHAR(64) NOT NULL,
          allowed_sources_json JSONB NOT NULL DEFAULT '[]'::jsonb,
          proposed_workflow_json JSONB NOT NULL,
          risk_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
          status VARCHAR(32) NOT NULL DEFAULT 'draft',
          idempotency_key VARCHAR(128),
          workflow_id BIGINT,
          automation_id BIGINT,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          expires_at TIMESTAMPTZ NOT NULL DEFAULT now() + interval '30 days'
        );
        CREATE INDEX IF NOT EXISTS idx_automation_draft_user_status_updated
          ON automation_draft(user_id, status, updated_at DESC);
        CREATE UNIQUE INDEX IF NOT EXISTS uq_automation_draft_enable_key
          ON automation_draft(user_id, idempotency_key) WHERE idempotency_key IS NOT NULL;
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS automation_draft")
    op.execute("DROP TABLE IF EXISTS experience_onboarding_state")
