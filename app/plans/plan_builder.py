from __future__ import annotations

from typing import Any


def build_plan_candidate(message: str, project_context: dict[str, Any]) -> dict[str, Any] | None:
    text = " ".join((message or "").split()).strip()
    if not _looks_like_long_task(text):
        return None
    resolver = project_context.get("resolver") if isinstance(project_context, dict) else {}
    project_id = resolver.get("projectId") if isinstance(resolver, dict) else None
    title = _title_from_text(text)
    steps = _steps_for_text(text)
    return {
        "type": "agent_plan",
        "projectId": project_id,
        "title": title,
        "objective": text[:500],
        "requiresConfirmation": True,
        "status": "pending",
        "confidence": 0.74,
        "steps": steps,
        "source": text[:240],
    }


def _looks_like_long_task(text: str) -> bool:
    if len(text) >= 80 and any(keyword in text for keyword in ["帮我", "整理", "拆", "规划", "实现", "开发", "方案"]):
        return True
    return any(
        keyword in text
        for keyword in [
            "拆成开发任务",
            "整理成prd",
            "整理成 PRD",
            "一步一步实现",
            "完整实现",
            "做一个方案",
            "生成执行计划",
            "推进计划",
            "三步推进计划",
            "拆一个三步",
            "拆一个计划",
            "拆解计划",
            "长期项目",
        ]
    )


def _title_from_text(text: str) -> str:
    title = text
    for prefix in ["帮我", "请", "继续", "把"]:
        title = title.replace(prefix, "", 1).strip()
    title = title.replace("，", " ").replace(",", " ").strip()
    return (title[:40] or "执行复杂任务计划").strip()


def _steps_for_text(text: str) -> list[dict[str, Any]]:
    if "PRD" in text.upper() or "方案" in text:
        return [
            {"title": "读取和梳理现有资料", "description": "确认输入文档、对话上下文和项目空间资料。"},
            {"title": "形成结构化草稿", "description": "输出目标、范围、用户场景、功能清单和验收标准。"},
            {"title": "拆解开发任务", "description": "按前端、后端、AI 服务和测试拆成可执行任务。"},
            {"title": "等待确认写入", "description": "涉及任务、决策或文档写入时先生成确认项。", "status": "awaiting_user"},
        ]
    if "实现" in text or "开发" in text:
        return [
            {"title": "定位相关模块", "description": "读取现有代码和接口约定。"},
            {"title": "设计最小可交付切片", "description": "拆分数据、API、前端和测试改动。"},
            {"title": "逐步实现并验证", "description": "每步完成后运行对应测试。"},
            {"title": "汇总结果和剩余风险", "description": "说明已完成内容、验证结果和后续项。"},
        ]
    return [
        {"title": "澄清目标和边界", "description": "确认要完成的产物、范围和限制。"},
        {"title": "拆解步骤", "description": "生成可执行步骤和依赖顺序。"},
        {"title": "执行低风险步骤", "description": "先完成读操作、整理和草稿输出。"},
        {"title": "确认高风险写操作", "description": "所有写入、外部变更或删除动作等待用户确认。", "status": "awaiting_user"},
    ]
