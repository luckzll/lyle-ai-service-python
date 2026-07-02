from datetime import UTC, datetime

from app.core.database import postgres_connection


def list_notifications(user_id: str, limit: int = 50) -> dict[str, object]:
    generate_due_notifications(user_id)
    with postgres_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, source_type, source_id, title, message, status, created_at, read_at
            FROM agent_notification
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (user_id, limit),
        ).fetchall()
    return {
        "items": [
            {
                "id": row[0],
                "sourceType": row[1],
                "sourceId": row[2],
                "title": row[3],
                "message": row[4],
                "status": row[5],
                "createdAt": row[6].isoformat(),
                "readAt": row[7].isoformat() if row[7] else None,
            }
            for row in rows
        ]
    }


def mark_notification_read(user_id: str, notification_id: int) -> dict[str, object] | None:
    with postgres_connection() as conn:
        row = conn.execute(
            """
            UPDATE agent_notification
            SET status = 'read', read_at = %s
            WHERE user_id = %s AND id = %s
            RETURNING id, status, read_at
            """,
            (datetime.now(UTC), user_id, notification_id),
        ).fetchone()
        conn.commit()
    if row is None:
        return None
    return {"id": row[0], "status": row[1], "readAt": row[2].isoformat()}


def generate_due_notifications(user_id: str) -> None:
    now = datetime.now(UTC)
    with postgres_connection() as conn:
        conn.execute(
            """
            INSERT INTO agent_notification (user_id, source_type, source_id, title, message, created_at)
            SELECT user_id, 'todo', id, title, '待办已到提醒时间：' || title, %s
            FROM agent_todo
            WHERE user_id = %s
              AND status = 'open'
              AND due_at IS NOT NULL
              AND due_at <= %s
            ON CONFLICT (user_id, source_type, source_id) DO NOTHING
            """,
            (now, user_id, now),
        )
        conn.execute(
            """
            INSERT INTO agent_notification (user_id, source_type, source_id, title, message, created_at)
            SELECT user_id, 'calendar', id, title, '日历事件已到提醒时间：' || title, %s
            FROM agent_calendar_event
            WHERE user_id = %s
              AND status = 'scheduled'
              AND starts_at IS NOT NULL
              AND starts_at <= %s
            ON CONFLICT (user_id, source_type, source_id) DO NOTHING
            """,
            (now, user_id, now),
        )
        conn.commit()


def generate_all_due_notifications() -> int:
    now = datetime.now(UTC)
    with postgres_connection() as conn:
        todo_rows = conn.execute(
            """
            INSERT INTO agent_notification (user_id, source_type, source_id, title, message, created_at)
            SELECT user_id, 'todo', id, title, '待办已到提醒时间：' || title, %s
            FROM agent_todo
            WHERE status = 'open'
              AND due_at IS NOT NULL
              AND due_at <= %s
            ON CONFLICT (user_id, source_type, source_id) DO NOTHING
            RETURNING id
            """,
            (now, now),
        ).fetchall()
        calendar_rows = conn.execute(
            """
            INSERT INTO agent_notification (user_id, source_type, source_id, title, message, created_at)
            SELECT user_id, 'calendar', id, title, '日历事件已到提醒时间：' || title, %s
            FROM agent_calendar_event
            WHERE status = 'scheduled'
              AND starts_at IS NOT NULL
              AND starts_at <= %s
            ON CONFLICT (user_id, source_type, source_id) DO NOTHING
            RETURNING id
            """,
            (now, now),
        ).fetchall()
        conn.commit()
    return len(todo_rows) + len(calendar_rows)
