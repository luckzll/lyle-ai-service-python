from app.core.config import settings
from app.core.database import postgres_connection


DEFAULT_SYSTEM_PROMPT = "你是 Lyle，一个清晰、可靠的 AI 智能体助手。"


def list_prompt_templates(limit: int = 50) -> list[dict[str, object]]:
    if not settings.postgres_enabled:
        return [_default_prompt_template()]
    with postgres_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, name, version, system_prompt, status, traffic_percent, created_at, updated_at
            FROM prompt_template
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            (max(1, min(limit, 100)),),
        ).fetchall()
    return [_row_to_prompt(item) for item in rows] or [_default_prompt_template()]


def create_prompt_template(payload: dict[str, object]) -> dict[str, object]:
    name = str(payload.get("name") or "Lyle Prompt").strip()[:128]
    version = str(payload.get("version") or settings.prompt_version).strip()[:128]
    system_prompt = str(payload.get("systemPrompt") or payload.get("system_prompt") or DEFAULT_SYSTEM_PROMPT).strip()
    status = str(payload.get("status") or "draft").strip()[:32]
    traffic_percent = _clamp_percent(payload.get("trafficPercent") or payload.get("traffic_percent") or 100)
    if not settings.postgres_enabled:
        return {
            **_default_prompt_template(),
            "name": name,
            "version": version,
            "systemPrompt": system_prompt,
            "status": status,
            "trafficPercent": traffic_percent,
            "persisted": False,
        }
    with postgres_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO prompt_template (name, version, system_prompt, status, traffic_percent, updated_at)
            VALUES (%s, %s, %s, %s, %s, now())
            ON CONFLICT (version)
            DO UPDATE SET
              name = EXCLUDED.name,
              system_prompt = EXCLUDED.system_prompt,
              status = EXCLUDED.status,
              traffic_percent = EXCLUDED.traffic_percent,
              updated_at = now()
            RETURNING id, name, version, system_prompt, status, traffic_percent, created_at, updated_at
            """,
            (name, version, system_prompt, status, traffic_percent),
        ).fetchone()
        conn.commit()
    return _row_to_prompt(row)


def activate_prompt_template(prompt_id: int) -> dict[str, object]:
    if not settings.postgres_enabled:
        return _default_prompt_template()
    with postgres_connection() as conn:
        conn.execute("UPDATE prompt_template SET status = 'archived', updated_at = now() WHERE status = 'active'")
        row = conn.execute(
            """
            UPDATE prompt_template
            SET status = 'active', traffic_percent = 100, updated_at = now()
            WHERE id = %s
            RETURNING id, name, version, system_prompt, status, traffic_percent, created_at, updated_at
            """,
            (prompt_id,),
        ).fetchone()
        conn.commit()
    if row is None:
        raise ValueError("Prompt template not found.")
    settings.prompt_version = str(row[2])
    return _row_to_prompt(row)


def get_active_system_prompt() -> tuple[str, str]:
    if not settings.postgres_enabled:
        return DEFAULT_SYSTEM_PROMPT, settings.prompt_version
    try:
        with postgres_connection() as conn:
            row = conn.execute(
                """
                SELECT version, system_prompt
                FROM prompt_template
                WHERE status = 'active'
                ORDER BY updated_at DESC
                LIMIT 1
                """
            ).fetchone()
        if row:
            return str(row[1]), str(row[0])
    except Exception:
        return DEFAULT_SYSTEM_PROMPT, settings.prompt_version
    return DEFAULT_SYSTEM_PROMPT, settings.prompt_version


def get_prompt_history(limit: int = 20) -> dict[str, object]:
    items = list_prompt_templates(limit)
    return {
        "items": items,
        "active": next((item for item in items if item.get("status") == "active"), items[0] if items else None),
    }


def _row_to_prompt(row) -> dict[str, object]:
    return {
        "id": row[0],
        "name": row[1],
        "version": row[2],
        "systemPrompt": row[3],
        "status": row[4],
        "trafficPercent": row[5],
        "createdAt": row[6].isoformat(),
        "updatedAt": row[7].isoformat(),
    }


def _default_prompt_template() -> dict[str, object]:
    return {
        "id": 0,
        "name": "Lyle 默认 Prompt",
        "version": settings.prompt_version,
        "systemPrompt": DEFAULT_SYSTEM_PROMPT,
        "status": "active",
        "trafficPercent": 100,
        "createdAt": "",
        "updatedAt": "",
        "persisted": False,
    }


def _clamp_percent(value: object) -> int:
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return 100
