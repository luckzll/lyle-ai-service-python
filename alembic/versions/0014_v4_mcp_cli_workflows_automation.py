"""add v4 mcp cli workflows automation

Revision ID: 0014_v4_mcp_cli_workflows_automation
Revises: 0013_projects_goals_and_plans
Create Date: 2026-07-09
"""

from alembic import op


revision = "0014_v4_mcp_cli_workflows_automation"
down_revision = "0013_projects_goals_and_plans"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_server (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          name VARCHAR(256) NOT NULL,
          server_type VARCHAR(64) NOT NULL,
          config_ref TEXT,
          status VARCHAR(32) NOT NULL DEFAULT 'disabled',
          last_health_status VARCHAR(32),
          last_health_message TEXT,
          last_checked_at TIMESTAMPTZ,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          UNIQUE(user_id, name)
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_mcp_server_user_status ON mcp_server(user_id, status, updated_at DESC)")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_tool (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          server_id BIGINT NOT NULL REFERENCES mcp_server(id) ON DELETE CASCADE,
          tool_name VARCHAR(256) NOT NULL,
          description TEXT,
          input_schema JSONB,
          output_schema JSONB,
          risk_level VARCHAR(32) NOT NULL DEFAULT 'read_only',
          enabled BOOLEAN NOT NULL DEFAULT true,
          last_seen_at TIMESTAMPTZ,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          UNIQUE(server_id, tool_name)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS tool_permission_policy (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          tool_id BIGINT REFERENCES mcp_tool(id) ON DELETE CASCADE,
          tool_name VARCHAR(256),
          operation VARCHAR(64) NOT NULL,
          resource_pattern TEXT,
          allow_manual BOOLEAN NOT NULL DEFAULT true,
          allow_automation BOOLEAN NOT NULL DEFAULT false,
          requires_confirmation BOOLEAN NOT NULL DEFAULT true,
          requires_second_confirmation BOOLEAN NOT NULL DEFAULT false,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS workspace (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          name VARCHAR(256) NOT NULL,
          root_path TEXT NOT NULL,
          status VARCHAR(32) NOT NULL DEFAULT 'active',
          default_shell VARCHAR(64) DEFAULT 'zsh',
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          UNIQUE(user_id, name)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS workspace_command_policy (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          workspace_id BIGINT NOT NULL REFERENCES workspace(id) ON DELETE CASCADE,
          command_pattern TEXT NOT NULL,
          policy VARCHAR(32) NOT NULL,
          risk_level VARCHAR(32) NOT NULL DEFAULT 'read_only',
          requires_confirmation BOOLEAN NOT NULL DEFAULT false,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS cli_command_run (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          workspace_id BIGINT REFERENCES workspace(id) ON DELETE SET NULL,
          project_id BIGINT,
          command TEXT NOT NULL,
          command_hash VARCHAR(128),
          risk_level VARCHAR(32) NOT NULL,
          status VARCHAR(32) NOT NULL DEFAULT 'pending',
          exit_code INTEGER,
          stdout_ref TEXT,
          stderr_ref TEXT,
          output_summary TEXT,
          started_at TIMESTAMPTZ,
          completed_at TIMESTAMPTZ,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS workflow_template (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          project_id BIGINT,
          name VARCHAR(256) NOT NULL,
          description TEXT,
          status VARCHAR(32) NOT NULL DEFAULT 'active',
          trigger_type VARCHAR(32) NOT NULL DEFAULT 'manual',
          created_from_plan_id BIGINT,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS workflow_step_template (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          workflow_id BIGINT NOT NULL REFERENCES workflow_template(id) ON DELETE CASCADE,
          sort_order INTEGER NOT NULL,
          title VARCHAR(256) NOT NULL,
          step_type VARCHAR(64) NOT NULL,
          tool_name VARCHAR(256),
          command_template TEXT,
          input_template JSONB,
          requires_confirmation BOOLEAN NOT NULL DEFAULT false,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS workflow_run (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          workflow_id BIGINT REFERENCES workflow_template(id) ON DELETE SET NULL,
          project_id BIGINT,
          trigger_type VARCHAR(32) NOT NULL,
          status VARCHAR(32) NOT NULL DEFAULT 'queued',
          input JSONB,
          output JSONB,
          error_code VARCHAR(64),
          error_message TEXT,
          started_at TIMESTAMPTZ,
          completed_at TIMESTAMPTZ,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS workflow_step_run (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          run_id BIGINT NOT NULL REFERENCES workflow_run(id) ON DELETE CASCADE,
          step_template_id BIGINT REFERENCES workflow_step_template(id) ON DELETE SET NULL,
          sort_order INTEGER NOT NULL,
          title VARCHAR(256) NOT NULL,
          step_type VARCHAR(64) NOT NULL,
          status VARCHAR(32) NOT NULL DEFAULT 'pending',
          input JSONB,
          output JSONB,
          tool_call_id VARCHAR(256),
          cli_run_id BIGINT,
          error_code VARCHAR(64),
          error_message TEXT,
          started_at TIMESTAMPTZ,
          completed_at TIMESTAMPTZ,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS automation_rule (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          workflow_id BIGINT NOT NULL REFERENCES workflow_template(id) ON DELETE CASCADE,
          name VARCHAR(256) NOT NULL,
          trigger_type VARCHAR(32) NOT NULL,
          schedule_expr VARCHAR(128),
          timezone VARCHAR(64),
          status VARCHAR(32) NOT NULL DEFAULT 'paused',
          next_run_at TIMESTAMPTZ,
          last_run_at TIMESTAMPTZ,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS automation_run (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          automation_id BIGINT NOT NULL REFERENCES automation_rule(id) ON DELETE CASCADE,
          workflow_run_id BIGINT REFERENCES workflow_run(id) ON DELETE SET NULL,
          status VARCHAR(32) NOT NULL DEFAULT 'queued',
          error_code VARCHAR(64),
          error_message TEXT,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          completed_at TIMESTAMPTZ
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS execution_audit (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(128) NOT NULL,
          project_id BIGINT,
          source_type VARCHAR(64) NOT NULL,
          source_id VARCHAR(256),
          executor_type VARCHAR(64) NOT NULL,
          executor_name VARCHAR(256) NOT NULL,
          operation VARCHAR(64) NOT NULL,
          risk_level VARCHAR(32) NOT NULL,
          status VARCHAR(32) NOT NULL,
          input_summary TEXT,
          output_summary TEXT,
          error_code VARCHAR(64),
          error_message TEXT,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_execution_audit_user_created ON execution_audit(user_id, created_at DESC)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS execution_audit")
    op.execute("DROP TABLE IF EXISTS automation_run")
    op.execute("DROP TABLE IF EXISTS automation_rule")
    op.execute("DROP TABLE IF EXISTS workflow_step_run")
    op.execute("DROP TABLE IF EXISTS workflow_run")
    op.execute("DROP TABLE IF EXISTS workflow_step_template")
    op.execute("DROP TABLE IF EXISTS workflow_template")
    op.execute("DROP TABLE IF EXISTS cli_command_run")
    op.execute("DROP TABLE IF EXISTS workspace_command_policy")
    op.execute("DROP TABLE IF EXISTS workspace")
    op.execute("DROP TABLE IF EXISTS tool_permission_policy")
    op.execute("DROP TABLE IF EXISTS mcp_tool")
    op.execute("DROP TABLE IF EXISTS mcp_server")
