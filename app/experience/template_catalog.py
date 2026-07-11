from __future__ import annotations

from copy import deepcopy
from typing import Any


_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "id": "project_daily_briefing", "version": 1, "category": "project",
        "name": "项目每日简报", "problem": "每天快速看清重点、风险与下一步。",
        "sources": ["项目", "目标", "任务"], "writes": False, "estimatedDuration": "约 1 分钟",
        "defaultSchedule": "daily:09:00", "output": "今日重点、风险、建议",
        "steps": [("读取项目与目标", "project_snapshot"), ("整理今日重点与风险", "daily_summary")],
    },
    {
        "id": "weekly_project_review", "version": 1, "category": "project",
        "name": "项目每周复盘", "problem": "汇总本周进展、阻塞和下周建议。",
        "sources": ["项目", "任务", "已授权工作区"], "writes": False, "estimatedDuration": "约 2 分钟",
        "defaultSchedule": "weekly:fri:17:00", "output": "进展、阻塞、下周建议",
        "steps": [("检查项目进展与逾期事项", "project_progress"), ("检查工作区状态", "workspace_state"), ("生成每周复盘", "weekly_summary")],
    },
    {
        "id": "workspace_health_check", "version": 1, "category": "code",
        "name": "代码健康检查", "problem": "发现未提交改动、TODO 与测试风险。",
        "sources": ["已授权工作区 Git 状态", "TODO", "测试摘要"], "writes": False, "estimatedDuration": "约 2 分钟",
        "defaultSchedule": "weekly:fri:16:00", "output": "代码状态和需处理项",
        "steps": [("检查 Git 与未提交改动", "git_state"), ("检查 TODO 与测试状态", "code_risks"), ("生成代码健康结论", "health_summary")],
    },
    {
        "id": "overdue_and_stale_alert", "version": 1, "category": "project",
        "name": "逾期与停滞提醒", "problem": "及时发现逾期任务和长期无进展事项。",
        "sources": ["任务", "项目活动"], "writes": False, "estimatedDuration": "约 1 分钟",
        "defaultSchedule": "daily:10:00", "output": "逾期清单和恢复建议",
        "steps": [("检查逾期与停滞事项", "stale_items"), ("生成恢复建议", "recovery_summary")],
    },
    {
        "id": "document_change_summary", "version": 1, "category": "document",
        "name": "文档变化总结", "problem": "了解重要文档变化及待确认事项。",
        "sources": ["已连接文档来源", "已授权工作区"], "writes": False, "estimatedDuration": "约 2 分钟",
        "defaultSchedule": "weekly:fri:15:00", "output": "变化摘要和待确认事项", "availability": "coming_soon",
        "steps": [],
    },
)


def list_templates() -> list[dict[str, Any]]:
    return [{k: deepcopy(v) for k, v in item.items() if k != "steps"} for item in _TEMPLATES]


def get_template(template_id: str) -> dict[str, Any] | None:
    return next((deepcopy(item) for item in _TEMPLATES if item["id"] == template_id), None)


def workflow_payload(template: dict[str, Any], *, title: str, project_id: int | None) -> dict[str, Any]:
    return {
        "name": title,
        "description": f"由 Lyle 受控模板「{template['name']}」生成；全程只读。",
        "projectId": project_id,
        "triggerType": "manual",
        "steps": [
            {
                "title": step_title, "stepType": "summarize", "outputKey": output_key,
                "requiresConfirmation": False,
                "inputTemplate": {"templateId": template["id"], "readOnly": True},
            }
            for step_title, output_key in template.get("steps", [])
        ],
    }
