"""add V4.2 result safety cleanup, result lookup and analytics indexes

Revision ID: 0024_v42_result_indexes
Revises: 0023_v41_product_events
"""

from alembic import op


revision = "0024_v42_result_indexes"
down_revision = "0023_v41_product_events"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Older summaries may contain masked-but-still-technical exception text. Remove it once so
    # the V4.2 detail API never exposes legacy raw errors; subsequent upserts use stable copy.
    op.execute(
        """
        UPDATE execution_result_summary
        SET conclusion = CASE status
              WHEN 'failed' THEN '执行未完成，请检查连接、权限或输入配置后重试。'
              WHEN 'awaiting_confirmation' THEN '执行已暂停，等待确认后可继续。'
              ELSE '部分步骤尚未完成，请查看状态后继续。'
            END,
            risks_json = '[]'::jsonb
        WHERE status IN ('failed', 'partial', 'awaiting_confirmation')
          AND source_type IN ('workflow_run', 'automation')
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_automation_run_user_workflow_created
          ON automation_run(user_id, workflow_run_id, created_at DESC)
          WHERE workflow_run_id IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_execution_result_user_created_id
          ON execution_result_summary(user_id, created_at DESC, id DESC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_experience_event_created_user_type
          ON experience_product_event(created_at DESC, anonymous_user_id, event_type)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_experience_event_created_user_type")
    op.execute("DROP INDEX IF EXISTS idx_execution_result_user_created_id")
    op.execute("DROP INDEX IF EXISTS idx_automation_run_user_workflow_created")
