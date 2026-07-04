from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException

from app.core.config import settings
from app.core.database import postgres_connection


PROJECT_TARGET_TYPES = {
    "conversation",
    "message",
    "memory_fragment",
    "knowledge_file",
    "knowledge_chunk",
    "todo",
    "calendar_event",
    "goal",
    "milestone",
    "decision",
    "open_question",
    "agent_plan",
}


def list_projects(user_id: str, status: str | None = "active", limit: int = 100) -> list[dict[str, Any]]:
    _ensure_postgres()
    limit = max(1, min(limit, 200))
    with postgres_connection() as conn:
        if status:
            rows = conn.execute(
                """
                SELECT id, name, slug, description, status, current_stage, priority,
                       last_activity_at, created_at, updated_at
                FROM project
                WHERE user_id = %s AND status = %s
                ORDER BY priority DESC, updated_at DESC
                LIMIT %s
                """,
                (user_id, status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, name, slug, description, status, current_stage, priority,
                       last_activity_at, created_at, updated_at
                FROM project
                WHERE user_id = %s
                ORDER BY priority DESC, updated_at DESC
                LIMIT %s
                """,
                (user_id, limit),
            ).fetchall()
    return [_project_row(row) for row in rows]


def create_project(user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_postgres()
    name = _required_text(payload.get("name"), "项目名称不能为空。")[:256]
    description = _optional_text(payload.get("description"))
    current_stage = _optional_text(payload.get("currentStage") or payload.get("current_stage"))
    status = _status(payload.get("status"), {"active", "archived"}, "active")
    priority = _int(payload.get("priority"), 0)
    slug = (_optional_text(payload.get("slug")) or _slugify(name))[:256]
    now = datetime.now(UTC)
    with postgres_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO project
              (user_id, name, slug, description, status, current_stage, priority,
               last_activity_at, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT(user_id, name) DO UPDATE SET
              description = EXCLUDED.description,
              status = EXCLUDED.status,
              current_stage = EXCLUDED.current_stage,
              priority = EXCLUDED.priority,
              last_activity_at = COALESCE(project.last_activity_at, EXCLUDED.last_activity_at),
              updated_at = EXCLUDED.updated_at
            RETURNING id, name, slug, description, status, current_stage, priority,
                      last_activity_at, created_at, updated_at
            """,
            (user_id, name, slug, description, status, current_stage, priority, now, now, now),
        ).fetchone()
        conn.commit()
    return _project_row(row)


def get_project(user_id: str, project_id: int) -> dict[str, Any]:
    _ensure_postgres()
    with postgres_connection() as conn:
        row = conn.execute(
            """
            SELECT id, name, slug, description, status, current_stage, priority,
                   last_activity_at, created_at, updated_at
            FROM project
            WHERE user_id = %s AND id = %s
            """,
            (user_id, project_id),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="项目不存在。")
    return _project_row(row)


def update_project(user_id: str, project_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_postgres()
    current = get_project(user_id, project_id)
    name = _optional_text(payload.get("name")) or current["name"]
    description = _optional_text(payload.get("description")) if "description" in payload else current.get("description")
    current_stage = (
        _optional_text(payload.get("currentStage") or payload.get("current_stage"))
        if "currentStage" in payload or "current_stage" in payload
        else current.get("currentStage")
    )
    status = _status(payload.get("status"), {"active", "archived"}, current["status"])
    priority = _int(payload.get("priority"), int(current.get("priority") or 0))
    slug = (_optional_text(payload.get("slug")) or str(current.get("slug") or _slugify(name)))[:256]
    with postgres_connection() as conn:
        row = conn.execute(
            """
            UPDATE project
            SET name = %s,
                slug = %s,
                description = %s,
                status = %s,
                current_stage = %s,
                priority = %s,
                updated_at = now()
            WHERE user_id = %s AND id = %s
            RETURNING id, name, slug, description, status, current_stage, priority,
                      last_activity_at, created_at, updated_at
            """,
            (name, slug, description, status, current_stage, priority, user_id, project_id),
        ).fetchone()
        conn.commit()
    if row is None:
        raise HTTPException(status_code=404, detail="项目不存在。")
    return _project_row(row)


def get_project_overview(user_id: str, project_id: int) -> dict[str, Any]:
    project = get_project(user_id, project_id)
    links = list_project_links(user_id, project_id, limit=200)
    link_counts: dict[str, int] = {}
    for link in links:
        target_type = str(link["targetType"])
        link_counts[target_type] = link_counts.get(target_type, 0) + 1
    decisions = _list_simple_project_rows(user_id, project_id, "project_decision", "title", "content", "active", 20)
    open_questions = _list_simple_project_rows(
        user_id,
        project_id,
        "project_open_question",
        "question",
        None,
        "open",
        20,
    )
    return {
        "project": project,
        "linkCounts": link_counts,
        "links": links[:20],
        "decisions": decisions,
        "openQuestions": open_questions,
        "nextStepSuggestions": _next_step_suggestions(project, link_counts, decisions, open_questions),
    }


def list_project_links(
    user_id: str,
    project_id: int,
    target_type: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    _ensure_postgres()
    _assert_project_exists(user_id, project_id)
    limit = max(1, min(limit, 300))
    with postgres_connection() as conn:
        if target_type:
            rows = conn.execute(
                """
                SELECT id, project_id, target_type, target_id, relation, confidence, created_at
                FROM project_link
                WHERE user_id = %s AND project_id = %s AND target_type = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (user_id, project_id, target_type, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, project_id, target_type, target_id, relation, confidence, created_at
                FROM project_link
                WHERE user_id = %s AND project_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (user_id, project_id, limit),
            ).fetchall()
    return [_link_row(row) for row in rows]


def create_project_link(user_id: str, project_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_postgres()
    _assert_project_exists(user_id, project_id)
    target_type = _required_text(payload.get("targetType") or payload.get("target_type"), "关联类型不能为空。")
    if target_type not in PROJECT_TARGET_TYPES:
        raise HTTPException(status_code=400, detail=f"不支持的关联类型：{target_type}")
    target_id = _required_text(payload.get("targetId") or payload.get("target_id"), "关联对象 ID 不能为空。")[:256]
    relation = (_optional_text(payload.get("relation")) or "related")[:64]
    confidence = _float(payload.get("confidence"), 0.8)
    confidence = min(1.0, max(0.0, confidence))
    with postgres_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO project_link (user_id, project_id, target_type, target_id, relation, confidence)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT(project_id, target_type, target_id, relation) DO UPDATE SET
              confidence = EXCLUDED.confidence
            RETURNING id, project_id, target_type, target_id, relation, confidence, created_at
            """,
            (user_id, project_id, target_type, target_id, relation, confidence),
        ).fetchone()
        conn.execute("UPDATE project SET last_activity_at = now(), updated_at = now() WHERE id = %s", (project_id,))
        conn.commit()
    return _link_row(row)


def delete_project_link(user_id: str, project_id: int, link_id: int) -> None:
    _ensure_postgres()
    with postgres_connection() as conn:
        row = conn.execute(
            "DELETE FROM project_link WHERE user_id = %s AND project_id = %s AND id = %s RETURNING id",
            (user_id, project_id, link_id),
        ).fetchone()
        conn.commit()
    if row is None:
        raise HTTPException(status_code=404, detail="项目关联不存在。")


def create_project_decision(user_id: str, project_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_postgres()
    _assert_project_exists(user_id, project_id)
    title = _required_text(payload.get("title"), "决策标题不能为空。")[:256]
    content = _required_text(payload.get("content"), "决策内容不能为空。")
    rationale = _optional_text(payload.get("rationale"))
    source_type = _optional_text(payload.get("sourceType") or payload.get("source_type"))
    source_id = _optional_text(payload.get("sourceId") or payload.get("source_id"))
    decided_at = _parse_datetime(payload.get("decidedAt") or payload.get("decided_at")) or datetime.now(UTC)
    with postgres_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO project_decision
              (user_id, project_id, title, content, rationale, source_type, source_id, decided_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id, title, content, status, created_at, updated_at
            """,
            (user_id, project_id, title, content, rationale, source_type, source_id, decided_at),
        ).fetchone()
        conn.execute("UPDATE project SET last_activity_at = now(), updated_at = now() WHERE id = %s", (project_id,))
        conn.commit()
    decision = {
        "id": row[0],
        "title": row[1],
        "content": row[2],
        "status": row[3],
        "createdAt": _iso(row[4]),
        "updatedAt": _iso(row[5]),
    }
    create_project_link(
        user_id,
        project_id,
        {
            "targetType": "decision",
            "targetId": str(decision["id"]),
            "relation": "records",
            "confidence": 1.0,
        },
    )
    return decision


def create_project_open_question(user_id: str, project_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_postgres()
    _assert_project_exists(user_id, project_id)
    question = _required_text(payload.get("question"), "开放问题不能为空。")
    source_type = _optional_text(payload.get("sourceType") or payload.get("source_type"))
    source_id = _optional_text(payload.get("sourceId") or payload.get("source_id"))
    with postgres_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO project_open_question (user_id, project_id, question, source_type, source_id)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, question, status, created_at, updated_at
            """,
            (user_id, project_id, question, source_type, source_id),
        ).fetchone()
        conn.execute("UPDATE project SET last_activity_at = now(), updated_at = now() WHERE id = %s", (project_id,))
        conn.commit()
    item = {
        "id": row[0],
        "question": row[1],
        "status": row[2],
        "createdAt": _iso(row[3]),
        "updatedAt": _iso(row[4]),
    }
    create_project_link(
        user_id,
        project_id,
        {
            "targetType": "open_question",
            "targetId": str(item["id"]),
            "relation": "tracks",
            "confidence": 1.0,
        },
    )
    return item


def update_project_open_question(user_id: str, question_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_postgres()
    status = _status(payload.get("status"), {"open", "resolved", "dismissed"}, "open")
    with postgres_connection() as conn:
        row = conn.execute(
            """
            UPDATE project_open_question
            SET status = %s, updated_at = now()
            WHERE user_id = %s AND id = %s
            RETURNING id, question, status, created_at, updated_at
            """,
            (status, user_id, question_id),
        ).fetchone()
        conn.commit()
    if row is None:
        raise HTTPException(status_code=404, detail="开放问题不存在。")
    return {
        "id": row[0],
        "question": row[1],
        "status": row[2],
        "createdAt": _iso(row[3]),
        "updatedAt": _iso(row[4]),
    }


def _list_simple_project_rows(
    user_id: str,
    project_id: int,
    table: str,
    title_column: str,
    content_column: str | None,
    status: str,
    limit: int,
) -> list[dict[str, Any]]:
    with postgres_connection() as conn:
        if content_column:
            rows = conn.execute(
                f"""
                SELECT id, {title_column}, {content_column}, status, created_at, updated_at
                FROM {table}
                WHERE user_id = %s AND project_id = %s AND status = %s
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (user_id, project_id, status, limit),
            ).fetchall()
            return [
                {
                    "id": row[0],
                    "title": row[1],
                    "content": row[2],
                    "status": row[3],
                    "createdAt": _iso(row[4]),
                    "updatedAt": _iso(row[5]),
                }
                for row in rows
            ]
        rows = conn.execute(
            f"""
            SELECT id, {title_column}, status, created_at, updated_at
            FROM {table}
            WHERE user_id = %s AND project_id = %s AND status = %s
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            (user_id, project_id, status, limit),
        ).fetchall()
        return [
            {
                "id": row[0],
                "question": row[1],
                "status": row[2],
                "createdAt": _iso(row[3]),
                "updatedAt": _iso(row[4]),
            }
            for row in rows
        ]


def _next_step_suggestions(
    project: dict[str, Any],
    link_counts: dict[str, int],
    decisions: list[dict[str, Any]],
    open_questions: list[dict[str, Any]],
) -> list[str]:
    suggestions: list[str] = []
    if not project.get("currentStage"):
        suggestions.append("补充项目当前阶段，方便后续生成每日重点和复盘。")
    if link_counts.get("todo", 0) == 0:
        suggestions.append("关联至少一个进行中任务，让项目页能展示下一步行动。")
    if link_counts.get("knowledge_file", 0) == 0:
        suggestions.append("关联项目资料或方案文档，作为项目事实来源。")
    if not decisions:
        suggestions.append("沉淀关键决策，避免后续规划反复摇摆。")
    if open_questions:
        suggestions.append("优先处理一个开放问题，降低项目推进阻塞。")
    return suggestions[:4]


def _assert_project_exists(user_id: str, project_id: int) -> None:
    with postgres_connection() as conn:
        row = conn.execute(
            "SELECT id FROM project WHERE user_id = %s AND id = %s",
            (user_id, project_id),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="项目不存在。")


def _project_row(row: Any) -> dict[str, Any]:
    return {
        "id": row[0],
        "name": row[1],
        "slug": row[2],
        "description": row[3],
        "status": row[4],
        "currentStage": row[5],
        "priority": row[6],
        "lastActivityAt": _iso(row[7]),
        "createdAt": _iso(row[8]),
        "updatedAt": _iso(row[9]),
    }


def _link_row(row: Any) -> dict[str, Any]:
    return {
        "id": row[0],
        "projectId": row[1],
        "targetType": row[2],
        "targetId": row[3],
        "relation": row[4],
        "confidence": row[5],
        "createdAt": _iso(row[6]),
    }


def _ensure_postgres() -> None:
    if not settings.postgres_enabled:
        raise HTTPException(status_code=503, detail="PostgreSQL 未启用，项目空间不可用。")


def _required_text(value: Any, message: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail=message)
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _status(value: Any, allowed: set[str], default: str) -> str:
    status = str(value or default).strip()
    if status not in allowed:
        raise HTTPException(status_code=400, detail=f"不支持的状态：{status}")
    return status


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_datetime(value: Any) -> datetime | str | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    text = str(value)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", value.strip().lower()).strip("-")
    return slug or "project"


def _iso(value: Any) -> str | None:
    return value.isoformat() if value else None
