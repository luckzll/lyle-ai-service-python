from __future__ import annotations

import re
from typing import Any

from app.core.config import settings
from app.core.database import postgres_connection
from app.projects.project_store import create_project_link, get_project, list_project_links, list_projects


def should_resolve_project(
    message: str,
    project_id: int | None = None,
    conversation_id: str | None = None,
    user_id: str | None = None,
) -> bool:
    """Skip clearly independent turns without losing project follow-ups.

    This gate never selects or grants access to a project. The regular resolver
    still validates ownership, active status, ambiguity and conversation links.
    Unknown requests keep that resolver; only clear independent requests skip it.
    """
    if project_id is not None:
        return True
    text = (message or "").strip().lower().strip("。！？!? ")
    if not text:
        return False
    if _looks_project_related(text) or any(term in text for term in ("lyle", "进展", "进度")):
        return True
    if _is_project_context_followup(text):
        if not conversation_id:
            return _looks_project_followup(text)
        if not user_id:
            return True  # Missing identity must not silently discard context.
        return _has_active_conversation_project_link(user_id, conversation_id)
    if text in {"你好", "您好", "嗨", "hi", "hello", "在吗", "谢谢", "多谢", "你是谁"}:
        return False
    if re.fullmatch(r"(?:请)?(?:帮我)?(?:(?:计算|算)(?:一下|算)?)?\s*[0-9\s.+*/%()（）=×÷^−-]+(?:等于|是)?(?:多少|几|什么)?", text):
        return False
    explanation = bool(re.match(r"^(?:请)?(?:帮我)?(?:解释|介绍|科普|讲解|讲讲)(?:一下|下)?", text))
    explanation = explanation or text.startswith(("什么是", "what is ", "what are ", "explain "))
    external_search = any(term in text for term in ("联网搜索", "联网查", "网上搜", "网上查", "上网搜", "搜索一下", "搜一下", "搜一搜", "web search", "search the web"))
    if explanation or external_search:
        return _mentions_known_project(text, user_id)
    return True


def _mentions_known_project(text: str, user_id: str | None) -> bool:
    """A generic verb does not erase an explicitly named, owned active project.

    This bounded precheck only reads candidate metadata. Ownership, ambiguity,
    links and context are still handled by the existing resolver afterwards.
    """
    if not user_id or not settings.postgres_enabled:
        return False
    try:
        projects = list_projects(user_id, status="active", limit=50)
    except Exception:
        # Failed optimization must retain the normal resolver/error reporting.
        return True
    return any(
        keyword.lower() in text
        for project in projects
        for keyword in _project_keywords(
            str(project.get("name") or ""), str(project.get("slug") or ""),
            str(project.get("description") or ""), str(project.get("currentStage") or ""),
        )
    )


def _is_project_context_followup(text: str) -> bool:
    if _looks_project_followup(text):
        return True
    if text in {"好", "好的", "可以", "行", "嗯", "ok", "yes", "yes please", "好吧", "没问题"}:
        return True
    if any(term in text for term in ("它", "这个", "这件事", "这段", "这份", "这张", "那个", "这些", "上述", "前面", "刚才", "上次", "之前", "接着", "展开", "详细一点", "具体一点")):
        return True
    return len(text) <= 30 and bool(re.search(r"(?:改|调整|换成|做成|优化|实现|部署|发布|同步|执行|修改|补充)", text))


def _has_active_conversation_project_link(user_id: str, conversation_id: str) -> bool:
    """One bounded lookup instead of enumerating projects and each link list."""
    if not settings.postgres_enabled:
        return False
    try:
        with postgres_connection() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM project_link AS link
                JOIN project ON project.id = link.project_id AND project.user_id = link.user_id
                WHERE link.user_id = %s AND project.user_id = %s
                  AND link.target_type = 'conversation' AND link.target_id = %s
                  AND project.status = 'active'
                LIMIT 1
                """,
                (user_id, user_id, conversation_id),
            ).fetchone()
        return row is not None
    except Exception:
        # A failed optimization must not strip a previously associated project.
        return True


def resolve_project(user_id: str, conversation_id: str | None, user_input: str, project_id: int | None = None) -> dict[str, Any]:
    text = (user_input or "").lower()
    if project_id is not None:
        try:
            project = get_project(user_id, project_id)
        except Exception as exc:
            return {
                "projectId": None,
                "projectName": "",
                "confidence": 0.0,
                "requiresClarification": True,
                "reason": f"explicit_project_unavailable:{type(exc).__name__}",
            }
        result = {
            "projectId": project["id"],
            "projectName": project["name"],
            "confidence": 1.0,
            "requiresClarification": False,
            "reason": "当前项目页显式指定项目",
            "candidates": [{"projectId": project["id"], "projectName": project["name"], "confidence": 1.0}],
        }
        from app.memory.fact_store import enabled as facts_enabled
        if facts_enabled() and project.get('status') not in {None, 'active'}:
            return {**result, 'projectId': None, 'requiresClarification': True, 'reason': '项目已结束或归档，不自动沿用为当前任务'}
        if conversation_id:
            try:
                create_project_link(
                    user_id,
                    int(project["id"]),
                    {
                        "targetType": "conversation",
                        "targetId": conversation_id,
                        "relation": "current_context",
                        "confidence": 1.0,
                    },
                )
                result["linkedConversation"] = True
            except Exception as exc:
                result["linkedConversation"] = False
                result["linkError"] = type(exc).__name__
        return result

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
        linked_project = _project_from_conversation_link(user_id, conversation_id, projects, text)
        if linked_project:
            project, link = linked_project
            return {
                "projectId": project["id"],
                "projectName": project["name"],
                "confidence": round(float(link.get("confidence") or 0.8), 2),
                "requiresClarification": False,
                "reason": "沿用当前会话已关联项目",
                "candidates": [{"projectId": project["id"], "projectName": project["name"], "confidence": round(float(link.get("confidence") or 0.8), 2)}],
                "linkedConversation": True,
            }
        return {
            "projectId": None,
            "projectName": "",
            "confidence": 0.0,
            "requiresClarification": len(projects) > 1 and (_looks_project_related(text) or _is_project_context_followup(text)),
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
    return (
        any(keyword in text for keyword in ["项目", "版本", "目标", "里程碑", "任务", "规划", "复盘", "prd", "v2", "v3"])
        or bool(re.search(r"\b(?:project|milestone|roadmap|sprint)\b", text))
    )


def _looks_project_followup(text: str) -> bool:
    return any(keyword in text for keyword in ["这个目标", "该目标", "这个项目", "该项目", "基于", "继续", "拆", "计划", "推进", "下一步"])


def _project_from_conversation_link(
    user_id: str,
    conversation_id: str | None,
    projects: list[dict[str, Any]],
    text: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    from app.memory.fact_store import enabled as facts_enabled
    if not conversation_id or (not facts_enabled() and not _is_project_context_followup(text)):
        return None
    best: tuple[dict[str, Any], dict[str, Any]] | None = None
    linked_ids = set()
    for project in projects:
        try:
            links = list_project_links(user_id, int(project["id"]), target_type="conversation", limit=20)
        except Exception:
            continue
        for link in links:
            if str(link.get("targetId") or "") != str(conversation_id):
                continue
            linked_ids.add(project['id'])
            if best is None or float(link.get("confidence") or 0.0) > float(best[1].get("confidence") or 0.0):
                best = (project, link)
    if facts_enabled() and len(linked_ids) > 1:
        return None  # Multiple associated tasks need disambiguation, not a guess.
    return best
