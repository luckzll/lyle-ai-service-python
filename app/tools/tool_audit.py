import json
import logging
from datetime import UTC, datetime

from app.core.config import settings
from app.core.database import postgres_connection

logger = logging.getLogger(__name__)


def audit_tool_call(
    *,
    tool_name: str,
    user_id: str,
    conversation_id: str | None,
    question: str,
    ok: bool,
    content: str,
    elapsed_ms: int,
) -> None:
    if not settings.tool_audit_enabled:
        return

    logger.info(
        "Tool call. tool=%s ok=%s user_id=%s conversation_id=%s elapsed_ms=%s",
        tool_name,
        ok,
        user_id,
        conversation_id,
        elapsed_ms,
    )

    if not settings.postgres_enabled:
        return

    payload = {
        "question": question[:1000],
        "content": content[:2000],
    }
    try:
        with postgres_connection() as conn:
            conn.execute(
                """
                INSERT INTO tool_audit_log
                  (tool_name, user_id, conversation_id, ok, elapsed_ms, payload, created_at)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
                """,
                (
                    tool_name,
                    user_id,
                    conversation_id or "",
                    ok,
                    elapsed_ms,
                    json.dumps(payload, ensure_ascii=False),
                    datetime.now(UTC),
                ),
            )
            conn.commit()
    except Exception:
        logger.exception("Failed to write tool audit log. tool=%s user_id=%s", tool_name, user_id)
