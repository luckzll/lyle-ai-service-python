from __future__ import annotations

from typing import Any

from app.goals.goal_store import list_goals
from app.projects.project_store import get_project_overview


def build_project_context(user_id: str, project_id: int | None) -> dict[str, Any]:
    if not project_id:
        return {}
    try:
        overview = get_project_overview(user_id, project_id)
        goals = list_goals(user_id, project_id=project_id, status="active", limit=20)
    except Exception as exc:
        return {"error": type(exc).__name__}
    project = overview.get("project") or {}
    return {
        "project": project,
        "goals": goals,
        "linkCounts": overview.get("linkCounts") or {},
        "decisions": overview.get("decisions") or [],
        "openQuestions": overview.get("openQuestions") or [],
        "nextStepSuggestions": overview.get("nextStepSuggestions") or [],
        "summary": _context_summary(project, goals, overview),
    }


def _context_summary(project: dict[str, Any], goals: list[dict[str, Any]], overview: dict[str, Any]) -> str:
    lines = [
        f"项目：{project.get('name') or ''}",
        f"阶段：{project.get('currentStage') or '未设置'}",
    ]
    if goals:
        lines.append("目标：" + "；".join(str(item.get("title") or "") for item in goals[:3]))
    suggestions = overview.get("nextStepSuggestions") or []
    if suggestions:
        lines.append("下一步建议：" + "；".join(str(item) for item in suggestions[:3]))
    return "\n".join(line for line in lines if line.strip())
