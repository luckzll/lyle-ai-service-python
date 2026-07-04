from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import HTTPException

from app.core.config import settings
from app.core.database import postgres_connection
from app.goals.goal_store import list_goals
from app.projects.project_store import list_projects


def get_today_briefing(user_id: str, project_id: int | None = None) -> dict[str, Any]:
    _ensure_postgres()
    generated = generate_daily_briefing(user_id, project_id=project_id, persist=False)
    return generated


def generate_daily_briefing(user_id: str, project_id: int | None = None, persist: bool = True) -> dict[str, Any]:
    _ensure_postgres()
    projects = list_projects(user_id, status="active", limit=20)
    if project_id is not None:
        projects = [project for project in projects if int(project["id"]) == int(project_id)]
    focus_project = projects[0] if projects else None
    goals = list_goals(user_id, project_id=int(focus_project["id"]) if focus_project else None, status="active", limit=10)
    todos = _list_todos(user_id, limit=50)
    operations = _list_pending_operations(user_id, limit=20)
    overdue = [item for item in todos if item.get("isOverdue")]
    today = [item for item in todos if item.get("isToday")]
    suggestions = _suggestions(focus_project, goals, today, overdue, operations)
    title = "今日项目简报"
    content = "\n".join([
        f"当前项目：{focus_project.get('name') if focus_project else '未设置'}",
        f"进行中目标：{len(goals)} 个",
        f"今日到期任务：{len(today)} 个",
        f"逾期任务：{len(overdue)} 个",
        f"待确认操作：{len(operations)} 个",
        "建议下一步：" + ("；".join(suggestions) if suggestions else "先选择一个项目目标继续推进"),
    ])
    briefing = {
        "id": None,
        "projectId": focus_project.get("id") if focus_project else None,
        "briefingType": "daily",
        "title": title,
        "content": content,
        "todayFocus": suggestions[:3],
        "projects": projects[:5],
        "goals": goals[:5],
        "dueTasks": today[:5],
        "overdueTasks": overdue[:5],
        "pendingOperations": operations[:5],
        "createdAt": datetime.now(UTC).isoformat(),
    }
    if persist:
        with postgres_connection() as conn:
            row = conn.execute(
                """
                INSERT INTO briefing (user_id, project_id, briefing_type, title, content, source_refs)
                VALUES (%s, %s, 'daily', %s, %s, %s)
                RETURNING id, created_at
                """,
                (user_id, briefing["projectId"], title, content, json.dumps(_source_refs(briefing), ensure_ascii=False)),
            ).fetchone()
            conn.commit()
        briefing["id"] = row[0]
        briefing["createdAt"] = row[1].isoformat()
    return briefing


def generate_review_briefing(
    user_id: str,
    briefing_type: str,
    project_id: int | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    _ensure_postgres()
    if briefing_type not in {"evening", "weekly"}:
        raise HTTPException(status_code=400, detail="复盘类型仅支持 evening 或 weekly。")
    projects = list_projects(user_id, status="active", limit=20)
    if project_id is not None:
        projects = [project for project in projects if int(project["id"]) == int(project_id)]
    focus_project = projects[0] if projects else None
    focus_project_id = int(focus_project["id"]) if focus_project else None
    goals = list_goals(user_id, project_id=focus_project_id, status="active", limit=10)
    days = 7 if briefing_type == "weekly" else 1
    completed = _list_completed_todos(user_id, days=days, limit=20)
    overdue = [item for item in _list_todos(user_id, limit=50) if item.get("isOverdue")]
    decisions = _list_project_decisions(user_id, focus_project_id, days=days, limit=10)
    questions = _list_project_open_questions(user_id, focus_project_id, limit=10)
    suggestions = _review_suggestions(goals, overdue, questions)
    title = "每周项目复盘" if briefing_type == "weekly" else "晚间项目复盘"
    content = "\n".join([
        f"当前项目：{focus_project.get('name') if focus_project else '未设置'}",
        f"周期：最近 {days} 天",
        f"完成任务：{len(completed)} 个",
        f"新增/更新决策：{len(decisions)} 个",
        f"仍开放问题：{len(questions)} 个",
        f"逾期任务：{len(overdue)} 个",
        "下一步建议：" + ("；".join(suggestions) if suggestions else "选择一个目标，拆出下一步行动"),
    ])
    briefing = {
        "id": None,
        "projectId": focus_project.get("id") if focus_project else None,
        "briefingType": briefing_type,
        "title": title,
        "content": content,
        "todayFocus": suggestions[:3],
        "projects": projects[:5],
        "goals": goals[:5],
        "dueTasks": completed[:5],
        "overdueTasks": overdue[:5],
        "pendingOperations": questions[:5],
        "createdAt": datetime.now(UTC).isoformat(),
    }
    if persist:
        with postgres_connection() as conn:
            row = conn.execute(
                """
                INSERT INTO briefing (user_id, project_id, briefing_type, title, content, source_refs)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id, created_at
                """,
                (
                    user_id,
                    briefing["projectId"],
                    briefing_type,
                    title,
                    content,
                    json.dumps(_review_source_refs(completed, decisions, questions, briefing), ensure_ascii=False),
                ),
            ).fetchone()
            conn.commit()
        briefing["id"] = row[0]
        briefing["createdAt"] = row[1].isoformat()
    return briefing


def _list_todos(user_id: str, limit: int) -> list[dict[str, Any]]:
    now = datetime.now(UTC)
    with postgres_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, title, status, due_at, conversation_id, created_at, updated_at
            FROM agent_todo
            WHERE user_id = %s AND status = 'open'
            ORDER BY due_at ASC NULLS LAST, updated_at DESC
            LIMIT %s
            """,
            (user_id, limit),
        ).fetchall()
    items = []
    for row in rows:
        due_at = row[3]
        items.append({
            "id": row[0],
            "title": row[1],
            "status": row[2],
            "dueAt": due_at.isoformat() if due_at else None,
            "conversationId": row[4],
            "isToday": bool(due_at and due_at.date() == now.date()),
            "isOverdue": bool(due_at and due_at < now),
            "createdAt": row[5].isoformat(),
            "updatedAt": row[6].isoformat(),
        })
    return items


def _list_pending_operations(user_id: str, limit: int) -> list[dict[str, Any]]:
    with postgres_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, tool_name, operation_type, risk_level, summary, created_at, updated_at
            FROM agent_pending_operation
            WHERE user_id = %s AND status = 'pending'
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (user_id, limit),
        ).fetchall()
    return [
        {
            "id": row[0],
            "toolName": row[1],
            "operationType": row[2],
            "riskLevel": row[3],
            "summary": row[4],
            "createdAt": row[5].isoformat(),
            "updatedAt": row[6].isoformat(),
        }
        for row in rows
    ]


def _list_completed_todos(user_id: str, days: int, limit: int) -> list[dict[str, Any]]:
    since = datetime.now(UTC) - timedelta(days=days)
    with postgres_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, title, status, due_at, conversation_id, created_at, updated_at
            FROM agent_todo
            WHERE user_id = %s AND status IN ('done', 'completed') AND updated_at >= %s
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            (user_id, since, limit),
        ).fetchall()
    return [
        {
            "id": row[0],
            "title": row[1],
            "status": row[2],
            "dueAt": row[3].isoformat() if row[3] else None,
            "conversationId": row[4],
            "createdAt": row[5].isoformat(),
            "updatedAt": row[6].isoformat(),
        }
        for row in rows
    ]


def _list_project_decisions(user_id: str, project_id: int | None, days: int, limit: int) -> list[dict[str, Any]]:
    since = datetime.now(UTC) - timedelta(days=days)
    clauses = ["user_id = %s", "updated_at >= %s"]
    params: list[Any] = [user_id, since]
    if project_id is not None:
        clauses.append("project_id = %s")
        params.append(project_id)
    params.append(limit)
    with postgres_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT id, project_id, title, content, status, created_at, updated_at
            FROM project_decision
            WHERE {' AND '.join(clauses)}
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            tuple(params),
        ).fetchall()
    return [
        {
            "id": row[0],
            "projectId": row[1],
            "title": row[2],
            "content": row[3],
            "status": row[4],
            "createdAt": row[5].isoformat(),
            "updatedAt": row[6].isoformat(),
        }
        for row in rows
    ]


def _list_project_open_questions(user_id: str, project_id: int | None, limit: int) -> list[dict[str, Any]]:
    clauses = ["user_id = %s", "status = 'open'"]
    params: list[Any] = [user_id]
    if project_id is not None:
        clauses.append("project_id = %s")
        params.append(project_id)
    params.append(limit)
    with postgres_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT id, project_id, question, status, created_at, updated_at
            FROM project_open_question
            WHERE {' AND '.join(clauses)}
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            tuple(params),
        ).fetchall()
    return [
        {
            "id": row[0],
            "projectId": row[1],
            "summary": row[2],
            "toolName": "project_open_question",
            "operationType": "resolve",
            "riskLevel": "low",
            "status": row[3],
            "createdAt": row[4].isoformat(),
            "updatedAt": row[5].isoformat(),
        }
        for row in rows
    ]


def _suggestions(
    project: dict[str, Any] | None,
    goals: list[dict[str, Any]],
    today: list[dict[str, Any]],
    overdue: list[dict[str, Any]],
    operations: list[dict[str, Any]],
) -> list[str]:
    suggestions: list[str] = []
    if overdue:
        suggestions.append(f"先处理逾期任务：{overdue[0]['title']}")
    if today:
        suggestions.append(f"推进今日到期任务：{today[0]['title']}")
    if operations:
        suggestions.append(f"确认或拒绝待处理操作：{operations[0]['summary']}")
    if goals:
        suggestions.append(f"推进目标：{goals[0]['title']}")
    if project and not project.get("currentStage"):
        suggestions.append("补充项目当前阶段")
    return suggestions[:5]


def _review_suggestions(
    goals: list[dict[str, Any]],
    overdue: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> list[str]:
    suggestions: list[str] = []
    if overdue:
        suggestions.append(f"重新安排逾期任务：{overdue[0]['title']}")
    if questions:
        suggestions.append(f"优先关闭开放问题：{questions[0]['summary']}")
    if goals:
        suggestions.append(f"复盘目标进展：{goals[0]['title']}")
    if not suggestions:
        suggestions.append("记录本周期完成项，并为下个周期设定一个可验证目标")
    return suggestions[:5]


def _source_refs(briefing: dict[str, Any]) -> dict[str, Any]:
    return {
        "projectIds": [item["id"] for item in briefing.get("projects", [])],
        "goalIds": [item["id"] for item in briefing.get("goals", [])],
        "todoIds": [item["id"] for item in briefing.get("dueTasks", []) + briefing.get("overdueTasks", [])],
        "operationIds": [item["id"] for item in briefing.get("pendingOperations", [])],
    }


def _review_source_refs(
    completed: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    briefing: dict[str, Any],
) -> dict[str, Any]:
    refs = _source_refs(briefing)
    refs["completedTodoIds"] = [item["id"] for item in completed]
    refs["decisionIds"] = [item["id"] for item in decisions]
    refs["openQuestionIds"] = [item["id"] for item in questions]
    return refs


def _ensure_postgres() -> None:
    if not settings.postgres_enabled:
        raise HTTPException(status_code=503, detail="PostgreSQL 未启用，简报不可用。")
