"""Versioned memory schema and durable build queue."""
from alembic import op

revision = "0029_memory_schema"
down_revision = "0028_v66_memory_governance"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""CREATE TABLE IF NOT EXISTS memory_schema_state (
        user_id TEXT PRIMARY KEY, state_json JSONB NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now())""")
    op.execute("""CREATE INDEX IF NOT EXISTS idx_memory_schema_queue
        ON memory_schema_state ((state_json->>'status'), updated_at)""")


def downgrade():
    op.drop_table("memory_schema_state")
