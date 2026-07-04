from __future__ import annotations

from typing import Any

from app.projects.project_store import create_project_link, list_projects


def resolve_project(user_id: str, conversation_id: str | None, user_input: str) -> dict[str, Any]:
    text = (user_input or "").lower()
    try:
        projects = list_projects(user_id, status="active", limit=50)
    except Exception as exc:
        return {
            "projectId": None,
            "projectName": "",
            "confidence": 0.0,
            "requiresClarification": False,
            "reason": f"project_store_unavailable:{type(exc).__name__}",
        }
    if not projects:
        return {
            "projectId": None,
            "projectName": "",
            "confidence": 0.0,
            "requiresClarification": False,
            "reason": "没有可匹配的活跃项目",
        }

    scored: list[tuple[float, dict[str, Any], list[str]]] = []
    for project in projects:
        hits: list[str] = []
        name = str(project.get("name") or "")
        slug = str(project.get("slug") or "")
        description = str(project.get("description") or "")
        stage = str(project.get("currentStage") or "")
        for keyword in _project_keywords(name, slug, description, stage):
            if keyword and keyword.lower() in text:
                hits.append(keyword)
        if hits:
            score = min(0.95, 0.56 + len(set(hits)) * 0.13 + float(project.get("priority") or 0) * 0.02)
            scored.append((score, project, hits))

    if not scored:
        return {
            "projectId": None,
            "projectName": "",
            "confidence": 0.0,
            "requiresClarification": len(projects) > 1 and _looks_project_related(text),
            "reason": "输入像项目讨论，但没有命中项目名称或描述关键词" if _looks_project_related(text) else "未命中项目关键词",
            "candidates": [{"projectId": item["id"], "projectName": item["name"]} for item in projects[:5]],
        }

    scored.sort(key=lambda item: item[0], reverse=True)
    top_score, top_project, hits = scored[0]
    requires_clarification = len(scored) > 1 and top_score - scored[1][0] < 0.12
    result = {
        "projectId": top_project["id"],
        "projectName": top_project["name"],
        "confidence": round(top_score, 2),
        "requiresClarification": requires_clarification,
        "reason": f"命中项目关键词：{', '.join(sorted(set(hits))[:5])}",
        "candidates": [
            {"projectId": project["id"], "projectName": project["name"], "confidence": round(score, 2)}
            for score, project, _ in scored[:5]
        ],
    }
    if conversation_id and not requires_clarification and top_score >= 0.65:
        try:
            create_project_link(
                user_id,
                int(top_project["id"]),
                {
                    "targetType": "conversation",
                    "targetId": conversation_id,
                    "relation": "current_context",
                    "confidence": top_score,
                },
            )
            result["linkedConversation"] = True
        except Exception as exc:
            result["linkedConversation"] = False
            result["linkError"] = type(exc).__name__
    return result


def _project_keywords(name: str, slug: str, description: str, stage: str) -> list[str]:
    raw = [name, slug, stage]
    raw.extend(part for part in name.replace("-", " ").replace("_", " ").split(" ") if len(part) >= 2)
    raw.extend(part for part in slug.replace("-", " ").replace("_", " ").split(" ") if len(part) >= 2)
    for token in ("v1", "v2", "v3", "prd", "rag", "agent", "lyle"):
        haystack = " ".join([name, slug, description, stage]).lower()
        if token in haystack:
            raw.append(token)
    return [item.strip() for item in raw if item and item.strip()]


def _looks_project_related(text: str) -> bool:
    return any(keyword in text for keyword in ["项目", "版本", "目标", "里程碑", "任务", "规划", "复盘", "prd", "v2", "v3"])
