"""add projects goals and plans

Revision ID: 0013_projects_goals_and_plans
Revises: 0012_pending_operation_idempotency
Create Date: 2026-07-04
"""

from alembic import op


revision = "0013_projects_goals_and_plans"
down_revision = "0012_pending_operation_idempotency"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS project (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          name VARCHAR(256) NOT NULL,
          slug VARCHAR(256),
          description TEXT,
          status VARCHAR(32) NOT NULL DEFAULT 'active',
          current_stage VARCHAR(128),
          priority INTEGER NOT NULL DEFAULT 0,
          last_activity_at TIMESTAMPTZ,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          UNIQUE(user_id, name)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_project_user_status
        ON project(user_id, status, updated_at DESC)
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS project_link (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          project_id BIGINT NOT NULL REFERENCES project(id) ON DELETE CASCADE,
          target_type VARCHAR(64) NOT NULL,
          target_id VARCHAR(256) NOT NULL,
          relation VARCHAR(64) NOT NULL DEFAULT 'related',
          confidence DOUBLE PRECISION NOT NULL DEFAULT 0.8,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          UNIQUE(project_id, target_type, target_id, relation)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_project_link_project
        ON project_link(project_id, target_type, created_at DESC)
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS project_decision (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          project_id BIGINT NOT NULL REFERENCES project(id) ON DELETE CASCADE,
          title VARCHAR(256) NOT NULL,
          content TEXT NOT NULL,
          rationale TEXT,
          source_type VARCHAR(64),
          source_id VARCHAR(256),
          status VARCHAR(32) NOT NULL DEFAULT 'active',
          decided_at TIMESTAMPTZ,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS project_open_question (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          project_id BIGINT NOT NULL REFERENCES project(id) ON DELETE CASCADE,
          question TEXT NOT NULL,
          status VARCHAR(32) NOT NULL DEFAULT 'open',
          source_type VARCHAR(64),
          source_id VARCHAR(256),
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS goal (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          project_id BIGINT REFERENCES project(id) ON DELETE SET NULL,
          title VARCHAR(256) NOT NULL,
          description TEXT,
          horizon VARCHAR(32) NOT NULL DEFAULT 'week',
          status VARCHAR(32) NOT NULL DEFAULT 'active',
          progress DOUBLE PRECISION NOT NULL DEFAULT 0,
          due_at TIMESTAMPTZ,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_goal_user_status
        ON goal(user_id, status, updated_at DESC)
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS milestone (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          goal_id BIGINT NOT NULL REFERENCES goal(id) ON DELETE CASCADE,
          title VARCHAR(256) NOT NULL,
          description TEXT,
          status VARCHAR(32) NOT NULL DEFAULT 'pending',
          sort_order INTEGER NOT NULL DEFAULT 0,
          due_at TIMESTAMPTZ,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS goal_task_link (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          goal_id BIGINT NOT NULL REFERENCES goal(id) ON DELETE CASCADE,
          milestone_id BIGINT REFERENCES milestone(id) ON DELETE SET NULL,
          task_id VARCHAR(256) NOT NULL,
          relation VARCHAR(64) NOT NULL DEFAULT 'contributes_to',
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          UNIQUE(goal_id, task_id)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_plan (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          project_id BIGINT REFERENCES project(id) ON DELETE SET NULL,
          conversation_id VARCHAR(128),
          title VARCHAR(256) NOT NULL,
          objective TEXT NOT NULL,
          status VARCHAR(32) NOT NULL DEFAULT 'draft',
          requires_confirmation BOOLEAN NOT NULL DEFAULT true,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          completed_at TIMESTAMPTZ
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_plan_step (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          plan_id BIGINT NOT NULL REFERENCES agent_plan(id) ON DELETE CASCADE,
          sort_order INTEGER NOT NULL,
          title VARCHAR(256) NOT NULL,
          description TEXT,
          status VARCHAR(32) NOT NULL DEFAULT 'pending',
          tool_name VARCHAR(128),
          output_ref TEXT,
          error_code VARCHAR(64),
          error_message TEXT,
          started_at TIMESTAMPTZ,
          completed_at TIMESTAMPTZ,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS briefing (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          project_id BIGINT REFERENCES project(id) ON DELETE SET NULL,
          briefing_type VARCHAR(32) NOT NULL,
          title VARCHAR(256) NOT NULL,
          content TEXT NOT NULL,
          source_refs JSONB,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS briefing")
    op.execute("DROP TABLE IF EXISTS agent_plan_step")
    op.execute("DROP TABLE IF EXISTS agent_plan")
    op.execute("DROP TABLE IF EXISTS goal_task_link")
    op.execute("DROP TABLE IF EXISTS milestone")
    op.execute("DROP TABLE IF EXISTS goal")
    op.execute("DROP TABLE IF EXISTS project_open_question")
    op.execute("DROP TABLE IF EXISTS project_decision")
    op.execute("DROP TABLE IF EXISTS project_link")
    op.execute("DROP TABLE IF EXISTS project")
