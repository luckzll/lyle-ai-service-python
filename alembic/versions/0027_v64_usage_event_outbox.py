"""V6.4 reliable AI usage event outbox.

Revision ID: 0027_v64_usage_event_outbox
Revises: 0026_v63_result_dismissal
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0027_v64_usage_event_outbox"
down_revision = "0026_v63_result_dismissal"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "usage_event_outbox",
        sa.Column("event_id", sa.String(length=64), primary_key=True),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.String(length=255), nullable=True),
    )
    op.create_index(
        "idx_usage_event_outbox_pending",
        "usage_event_outbox",
        ["delivered_at", "next_attempt_at", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_usage_event_outbox_pending", table_name="usage_event_outbox")
    op.drop_table("usage_event_outbox")
