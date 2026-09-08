"""Authoritative memory facts; legacy data is retained for audited migration."""
from alembic import op

revision = '0030_memory_facts'
down_revision = '0029_memory_schema'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("CREATE TABLE IF NOT EXISTS memory_fact_owner (user_id TEXT PRIMARY KEY, metadata JSONB NOT NULL DEFAULT '{}'::jsonb)")
    op.execute("""CREATE TABLE IF NOT EXISTS memory_fact (
        fact_id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES memory_fact_owner(user_id),
        schema_path TEXT NOT NULL, data JSONB NOT NULL)""")
    op.execute('CREATE INDEX IF NOT EXISTS idx_memory_fact_user_path ON memory_fact(user_id, schema_path)')


def downgrade():
    op.drop_table('memory_fact')
    op.drop_table('memory_fact_owner')
