import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import re
from typing import Callable
from zoneinfo import ZoneInfo

from app.core.config import settings
from app.core.database import postgres_connection
from app.memory.overview import is_memory_overview_question
from app.memory.store import export_memory
from app.rag.reranker import get_reranker_status
from app.tools.risk_policy import ToolRiskPolicy, get_tool_risk_policy
from app.tools.file_parse_tool import parse_file_from_question
from app.tools.image_intent import is_image_generation_request
from app.tools.image_tool import analyze_image_from_question, edit_image_from_question, generate_image_from_question
from app.tools.task_tool import preview_calendar_event_from_question, preview_todo_from_question
from app.tools.tool_audit import audit_tool_call
from app.tools.weather_tool import get_weather_from_question
from app.tools.web_search_tool import search_web

LOCAL_TIMEZONE = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    handler: Callable[[str, str | None, str], dict[str, object]]
    risk_policy: ToolRiskPolicy


def list_available_tools() -> list[dict[str, object]]:
    allowed = _allowed_tool_names()
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "enabled": spec.name in allowed and _tool_switch_enabled(spec.name),
            "riskLevel": spec.risk_policy.risk_level,
            "requiresConfirmation": spec.risk_policy.requires_confirmation,
            "riskDescription": spec.risk_policy.description,
        }
        for spec in _tool_specs().values()
    ]


def select_tools(question: str) -> list[str]:
    lowered = question.lower()
    selected: list[str] = []
    memory_overview = is_memory_overview_question(question)
    if any(keyword in question for keyword in ["健康", "状态", "检查服务", "服务检查"]) or "health" in lowered:
        selected.append("health_check")
    if memory_overview or any(keyword in question for keyword in ["查看记忆", "记忆快照", "我的记忆", "当前记忆"]):
        selected.append("memory_snapshot")
    if any(keyword in question for keyword in ["记忆统计", "记忆数量", "数据库统计", "统计记忆"]):
        selected.append("memory_stats")
    work_summary_keywords = [
        "任务汇总",
        "任务总结",
        "工作汇总",
        "工作简报",
        "进展汇总",
        "进展总结",
        "任务版图",
        "工作版图",
        "有哪些任务",
        "几个任务",
        "待办数量",
        "逾期任务",
        "本周到期",
        "所有项目",
        "所有任务",
        "所有操作",
        "所有计划",
        "所有目标",
        "项目任务",
        "任务操作",
        "操作计划",
        "计划目标",
        "项目目标",
        "项目计划",
        "长任务计划",
        "目标进度",
        "当前任务",
        "当前项目",
        "本周安排",
    ]
    if memory_overview or any(keyword in question for keyword in work_summary_keywords):
        selected.append("work_summary")
    if any(keyword in question for keyword in ["天气", "气温", "温度", "下雨", "下雪"]) or "weather" in lowered:
        selected.append("weather")
    search_keywords = [
        "网页搜索",
        "联网搜索",
        "搜索一下",
        "网上搜",
        "上网搜",
        "网上查",
        "联网查",
        "查一下",
        "搜一下",
        "搜一搜",
        "去搜",
        "搜索",
        "最新",
        "实时",
        "新闻",
    ]
    if (
        any(keyword in question for keyword in search_keywords)
        or "web search" in lowered
        or _looks_like_external_web_question(question)
    ):
        if "weather" not in selected:
            selected.append("web_search")
    if any(keyword in question for keyword in ["解析文件", "读取文件", "分析文件", "file://", "s3://"]) or "parse file" in lowered:
        selected.append("file_parse")
    image_edit_keywords = [
        "改图",
        "修改一下",
        "改一下",
        "可以修改",
        "能不能修改",
        "修改图片",
        "编辑图片",
        "图生图",
        "重绘",
        "换背景",
        "去掉",
        "移除",
        "替换",
        "把背景",
        "把这张图",
        "把图片",
    ]
    has_image_ref = (
        "data:image/" in lowered
        or bool(re.search(r"https?://\S+\.(?:png|jpe?g|webp|gif)", question, re.IGNORECASE))
    )
    reference_generation = (
        has_image_ref
        and any(keyword in question for keyword in ["参考", "基于", "结合", "按照这些图", "用这些图"])
        and is_image_generation_request(question)
    )
    if has_image_ref and (any(keyword in question for keyword in image_edit_keywords) or reference_generation):
        selected.append("image_edit")
    elif (
        any(keyword in question for keyword in ["识别图片", "看图", "这张图", "图片里", "图里", "分析图片", "图片识别"])
        or "image analyze" in lowered
        or "data:image/" in lowered
    ):
        selected.append("image_analyze")
    if is_image_generation_request(question):
        if "image_edit" not in selected:
            selected.append("image_generate")
    if any(keyword in question for keyword in ["创建待办", "加个待办", "待办"]) or "todo" in lowered:
        selected.append("todo")
    if any(keyword in question for keyword in ["创建日历", "加个日程", "日历", "日程"]) or "calendar" in lowered:
        selected.append("calendar")
    if any(keyword in question for keyword in ["有哪些工具", "工具列表", "可用工具"]):
        selected.append("list_tools")
    return _dedupe(selected)


def _looks_like_external_web_question(question: str) -> bool:
    external_keywords = [
        "最近",
        "今天",
        "现在",
        "目前",
        "刚刚",
        "版本",
        "更新",
        "上线",
        "发布",
        "公告",
        "官方",
        "争议",
        "热搜",
        "舆情",
        "爆料",
        "价格",
        "榜单",
        "票房",
        "赛程",
        "比分",
        "天气",
        "政策",
        "法规",
        "新闻",
    ]
    if any(keyword in question for keyword in external_keywords):
        return True
    if re.search(r"\b\d+(?:\.\d+)+\b", question) and any(keyword in question for keyword in ["版", "争议", "更新", "改动"]):
        return True
    if re.search(r"\b(202[4-9]|203\d)\b", question) and any(keyword in question for keyword in ["发生", "变化", "趋势", "政策", "事件"]):
        return True
    return False


def execute_tool(
    tool_name: str,
    user_id: str,
    conversation_id: str | None,
    question: str,
) -> dict[str, object]:
    started_at = time.perf_counter()
    try:
        result = _execute_tool(tool_name, user_id, conversation_id, question)
    except Exception as exc:
        result = {
            "tool": tool_name,
            "ok": False,
            "content": f"工具调用异常：{exc}",
        }

    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    result = _normalise_tool_result(tool_name, result)
    result["elapsedMs"] = elapsed_ms
    audit_tool_call(
        tool_name=tool_name,
        user_id=user_id,
        conversation_id=conversation_id,
        question=question,
        ok=bool(result.get("ok")),
        content=str(result.get("content") or ""),
        elapsed_ms=elapsed_ms,
    )
    return result


def format_tool_context(tool_results: list[dict[str, object]]) -> str:
    if not tool_results:
        return ""
    lines = []
    for item in tool_results:
        status = "成功" if item.get("ok") else "失败"
        content = item.get("userMessage") or item.get("content")
        if item.get("tool") == "image_generate" and item.get("ok"):
            content = "图片已成功生成，最终图片会由系统自动附加到回答末尾。回答中不要输出图片 URL 或 Markdown。"
        if item.get("tool") == "image_edit" and item.get("ok"):
            content = "图片已成功修改，最终图片会由系统自动附加到回答末尾。回答中不要输出图片 URL 或 Markdown。"
        if item.get("tool") == "image_analyze" and item.get("ok"):
            content = f"图片识别结果：{content}\n回答时直接根据识别结果回应用户，不要解释图片输入格式或工具限制。"
        if item.get("requiresConfirmation"):
            operation = item.get("operation") if isinstance(item.get("operation"), dict) else {}
            content = (
                f"已生成待用户确认的操作预览：{operation.get('summary') or content}。"
                "回答时说明需要用户点击确认后才会执行，不要说已经完成。"
            )
        lines.append(f"- {item.get('tool')}({status})：{content}")
    return (
        "已完成的工具调用结果如下。回答时必须优先使用这些结果；"
        "不要再说“我将调用工具/我去搜索”，除非工具调用失败。\n"
        + "\n".join(lines)
    )


def _execute_tool(
    tool_name: str,
    user_id: str,
    conversation_id: str | None,
    question: str,
) -> dict[str, object]:
    specs = _tool_specs()
    spec = specs.get(tool_name)
    if spec is None:
        return {
            "tool": tool_name,
            "ok": False,
            "content": f"未知工具：{tool_name}",
        }
    if tool_name not in _allowed_tool_names():
        return {
            "tool": tool_name,
            "ok": False,
            "content": f"工具未在白名单中：{tool_name}",
        }
    if not _tool_switch_enabled(tool_name):
        return {
            "tool": tool_name,
            "ok": False,
            "content": f"工具未启用：{tool_name}",
        }
    result = spec.handler(user_id, conversation_id, question)
    if (
        spec.risk_policy.requires_confirmation
        and result.get("ok")
        and not result.get("requiresConfirmation")
    ):
        return {
            "tool": tool_name,
            "ok": False,
            "content": "写操作被安全策略拦截：该工具必须先生成待确认操作，不能直接执行。",
            "policyBlocked": True,
            "riskLevel": spec.risk_policy.risk_level,
            "requiresConfirmation": True,
        }
    return {"tool": tool_name, **result}


def _normalise_tool_result(tool_name: str, result: dict[str, object]) -> dict[str, object]:
    risk_policy = get_tool_risk_policy(tool_name)
    ok = bool(result.get("ok"))
    content = str(result.get("content") or result.get("userMessage") or "")
    operation_type = str(result.get("operationType") or _operation_type_for_tool(tool_name, result))
    requires_confirmation = bool(result.get("requiresConfirmation", risk_policy.requires_confirmation and not ok))
    error = result.get("error") if isinstance(result.get("error"), dict) else None
    if not ok and error is None:
        error = _classify_tool_error(content)
    normalized = {
        **result,
        "tool": str(result.get("tool") or tool_name),
        "ok": ok,
        "operationType": operation_type,
        "riskLevel": str(result.get("riskLevel") or risk_policy.risk_level),
        "requiresConfirmation": requires_confirmation,
        "retryable": bool(result.get("retryable", bool(error and error.get("retryable")))),
        "userMessage": content,
        "content": content,
        "result": _tool_result_payload(result),
        "error": None if ok else error,
    }
    return normalized


def _operation_type_for_tool(tool_name: str, result: dict[str, object]) -> str:
    operation = result.get("operation") if isinstance(result.get("operation"), dict) else {}
    if operation.get("operationType"):
        return str(operation["operationType"])
    return {
        "health_check": "health.check",
        "memory_snapshot": "memory.snapshot",
        "memory_stats": "memory.stats",
        "work_summary": "work.summary",
        "web_search": "web.search",
        "weather": "weather.read",
        "file_parse": "file.parse",
        "image_analyze": "image.analyze",
        "image_generate": "image.generate",
        "image_edit": "image.edit",
        "todo": "todo.preview",
        "calendar": "calendar.preview",
        "list_tools": "tools.list",
    }.get(tool_name, f"{tool_name}.run")


def _tool_result_payload(result: dict[str, object]) -> dict[str, object]:
    excluded = {
        "tool",
        "ok",
        "content",
        "operationType",
        "riskLevel",
        "requiresConfirmation",
        "retryable",
        "userMessage",
        "result",
        "error",
        "elapsedMs",
    }
    payload = result.get("result")
    if isinstance(payload, dict):
        return payload
    return {key: value for key, value in result.items() if key not in excluded}


def _classify_tool_error(message: str) -> dict[str, object]:
    text = message or ""
    if any(marker in text for marker in ["没有识别", "没有找到", "请说明", "请在问题中", "未知工具", "不支持"]):
        code, retryable = "PARAM_MISSING", False
    elif any(marker in text for marker in ["PostgreSQL", "数据库", "SQL"]):
        code, retryable = "DATABASE_ERROR", False
    elif any(marker in text for marker in ["白名单", "未启用", "拒绝", "权限", "不属于当前用户"]):
        code, retryable = "PERMISSION_DENIED", False
    elif any(marker in text.lower() for marker in ["timeout", "timed out", "超时"]):
        code, retryable = "TIMEOUT", True
    elif any(marker in text for marker in ["限流", "rate limit", "429"]):
        code, retryable = "RATE_LIMITED", True
    elif any(marker in text for marker in ["格式", "JSON", "解析失败"]):
        code, retryable = "MODEL_FORMAT_ERROR", True
    else:
        code, retryable = "UNKNOWN_ERROR", False
    return {
        "code": code,
        "message": text,
        "retryable": retryable,
    }


def _tool_specs() -> dict[str, ToolSpec]:
    return {
        "health_check": ToolSpec(
            name="health_check",
            description="检查 Python AI 服务内部工具系统、Reranker 状态。",
            handler=_health_check,
            risk_policy=get_tool_risk_policy("health_check"),
        ),
        "memory_snapshot": ToolSpec(
            name="memory_snapshot",
            description="查看当前用户的记忆快照，包括短期、长期摘要、结构化和向量召回记忆。",
            handler=_memory_snapshot,
            risk_policy=get_tool_risk_policy("memory_snapshot"),
        ),
        "memory_stats": ToolSpec(
            name="memory_stats",
            description="只读查询 PostgreSQL 记忆表统计，不执行用户传入 SQL。",
            handler=_memory_stats,
            risk_policy=get_tool_risk_policy("memory_stats"),
        ),
        "work_summary": ToolSpec(
            name="work_summary",
            description="只读查询 PostgreSQL 中的待办、项目、目标、计划和项目关联，生成实时工作简报。",
            handler=_work_summary,
            risk_policy=get_tool_risk_policy("work_summary"),
        ),
        "web_search": ToolSpec(
            name="web_search",
            description="网页搜索工具入口，需配置搜索 provider 后启用。",
            handler=_web_search,
            risk_policy=get_tool_risk_policy("web_search"),
        ),
        "weather": ToolSpec(
            name="weather",
            description="按需查询指定城市当前天气和短期预报。",
            handler=_weather,
            risk_policy=get_tool_risk_policy("weather"),
        ),
        "file_parse": ToolSpec(
            name="file_parse",
            description="临时解析 file://、s3:// 或本地绝对路径文件。",
            handler=_file_parse,
            risk_policy=get_tool_risk_policy("file_parse"),
        ),
        "image_analyze": ToolSpec(
            name="image_analyze",
            description="识别/分析用户上传的图片内容。",
            handler=_image_analyze,
            risk_policy=get_tool_risk_policy("image_analyze"),
        ),
        "image_generate": ToolSpec(
            name="image_generate",
            description="根据文本提示词生成图片。",
            handler=_image_generate,
            risk_policy=get_tool_risk_policy("image_generate"),
        ),
        "image_edit": ToolSpec(
            name="image_edit",
            description="根据用户上传的图片和编辑指令生成修改后的图片。",
            handler=_image_edit,
            risk_policy=get_tool_risk_policy("image_edit"),
        ),
        "todo": ToolSpec(
            name="todo",
            description="生成创建待办的操作预览，用户确认后才写入 PostgreSQL。",
            handler=_todo,
            risk_policy=get_tool_risk_policy("todo"),
        ),
        "calendar": ToolSpec(
            name="calendar",
            description="生成创建日程的操作预览，用户确认后才写入 PostgreSQL。",
            handler=_calendar,
            risk_policy=get_tool_risk_policy("calendar"),
        ),
        "list_tools": ToolSpec(
            name="list_tools",
            description="列出当前可用工具及启用状态。",
            handler=_list_tools,
            risk_policy=get_tool_risk_policy("list_tools"),
        ),
    }


def _health_check(user_id: str, conversation_id: str | None, question: str) -> dict[str, object]:
    reranker = get_reranker_status()
    return {
        "ok": bool(reranker.get("ready", True)),
        "content": f"Python AI 服务工具系统可用。Reranker={reranker.get('provider')} ready={reranker.get('ready')}",
        "reranker": reranker,
    }


def _memory_snapshot(user_id: str, conversation_id: str | None, question: str) -> dict[str, object]:
    memory = export_memory(user_id, conversation_id)
    return {
        "ok": True,
        "content": _summarize_memory_snapshot(memory),
        "raw": memory,
    }


def _memory_stats(user_id: str, conversation_id: str | None, question: str) -> dict[str, object]:
    if not settings.postgres_enabled:
        return {
            "ok": False,
            "content": "PostgreSQL 未启用，无法读取长期摘要和结构化记忆统计。",
        }

    with postgres_connection() as conn:
        summary_count = conn.execute(
            "SELECT COUNT(*) FROM memory_summary WHERE user_id = %s",
            (user_id,),
        ).fetchone()[0]
        structured_count = conn.execute(
            "SELECT COUNT(*) FROM structured_memory WHERE user_id = %s AND status = 'active'",
            (user_id,),
        ).fetchone()[0]
        disabled_structured_count = conn.execute(
            "SELECT COUNT(*) FROM structured_memory WHERE user_id = %s AND status <> 'active'",
            (user_id,),
        ).fetchone()[0]

    return {
        "ok": True,
        "content": (
            f"当前用户长期摘要 {summary_count} 条，"
            f"启用的结构化记忆 {structured_count} 条，"
            f"停用/过期结构化记忆 {disabled_structured_count} 条。"
        ),
        "stats": {
            "summaryCount": summary_count,
            "structuredCount": structured_count,
            "disabledStructuredCount": disabled_structured_count,
        },
    }


def _work_summary(user_id: str, conversation_id: str | None, question: str) -> dict[str, object]:
    if not settings.postgres_enabled:
        return {
            "ok": False,
            "content": "PostgreSQL 未启用，无法读取实时工作简报。",
        }

    now = datetime.now(UTC)
    today_start, today_end = _local_day_window(now)
    week_start, week_end = _local_week_window(now)
    with postgres_connection() as conn:
        todo_counts = conn.execute(
            """
            SELECT
              COUNT(*) FILTER (WHERE status = 'open') AS open_count,
              COUNT(*) FILTER (WHERE status = 'open' AND due_at >= %s AND due_at < %s) AS today_count,
              COUNT(*) FILTER (WHERE status = 'open' AND due_at IS NOT NULL AND due_at < %s) AS overdue_count,
              COUNT(*) FILTER (WHERE status = 'open' AND due_at >= %s AND due_at < %s) AS week_due_count
            FROM agent_todo
            WHERE user_id = %s
            """,
            (today_start, today_end, now, week_start, week_end, user_id),
        ).fetchone()
        week_due_rows = conn.execute(
            """
            SELECT id, title, due_at, conversation_id
            FROM agent_todo
            WHERE user_id = %s
              AND status = 'open'
              AND due_at >= %s
              AND due_at < %s
            ORDER BY due_at ASC NULLS LAST, updated_at DESC
            LIMIT 8
            """,
            (user_id, week_start, week_end),
        ).fetchall()
        todo_rows = conn.execute(
            """
            SELECT id, title, status, due_at, conversation_id, updated_at
            FROM agent_todo
            WHERE user_id = %s
            ORDER BY
              CASE WHEN status = 'open' THEN 0 ELSE 1 END,
              due_at ASC NULLS LAST,
              updated_at DESC
            LIMIT 10
            """,
            (user_id,),
        ).fetchall()
        project_counts = conn.execute(
            """
            SELECT
              COUNT(*) FILTER (WHERE status = 'active') AS active_count,
              COUNT(*) AS total_count
            FROM project
            WHERE user_id = %s
            """,
            (user_id,),
        ).fetchone()
        project_rows = conn.execute(
            """
            SELECT id, name, status, current_stage, priority,
                   last_activity_at, updated_at
            FROM project
            WHERE user_id = %s
            ORDER BY
              CASE WHEN status = 'active' THEN 0 ELSE 1 END,
              COALESCE(last_activity_at, updated_at) DESC
            LIMIT 10
            """,
            (user_id,),
        ).fetchall()
        current_project_rows = _current_project_rows(conn, user_id, conversation_id)
        goal_counts = conn.execute(
            """
            SELECT
              COUNT(*) FILTER (WHERE status = 'active') AS active_count,
              COALESCE(AVG(progress) FILTER (WHERE status = 'active'), 0) AS avg_progress,
              COUNT(*) FILTER (WHERE status = 'completed') AS completed_count
            FROM goal
            WHERE user_id = %s
            """,
            (user_id,),
        ).fetchone()
        goal_rows = conn.execute(
            """
            SELECT g.id, g.project_id, p.name AS project_name, g.title,
                   g.horizon, g.status, g.progress, g.due_at, g.updated_at
            FROM goal g
            LEFT JOIN project p ON p.id = g.project_id AND p.user_id = g.user_id
            WHERE g.user_id = %s
            ORDER BY
              CASE WHEN g.status = 'active' THEN 0 ELSE 1 END,
              g.updated_at DESC
            LIMIT 10
            """,
            (user_id,),
        ).fetchall()
        plan_counts = conn.execute(
            """
            SELECT
              COUNT(*) FILTER (WHERE status IN ('running', 'awaiting_confirmation')) AS active_count,
              COUNT(*) FILTER (WHERE status = 'running') AS running_count,
              COUNT(*) FILTER (WHERE status = 'awaiting_confirmation') AS awaiting_count,
              COUNT(*) FILTER (WHERE status = 'completed') AS completed_count
            FROM agent_plan
            WHERE user_id = %s
            """,
            (user_id,),
        ).fetchone()
        plan_rows = conn.execute(
            """
            SELECT ap.id, ap.project_id, p.name AS project_name, ap.title,
                   ap.objective, ap.status, ap.requires_confirmation, ap.updated_at
            FROM agent_plan ap
            LEFT JOIN project p ON p.id = ap.project_id AND p.user_id = ap.user_id
            WHERE ap.user_id = %s
            ORDER BY
              CASE
                WHEN ap.status = 'awaiting_confirmation' THEN 0
                WHEN ap.status = 'running' THEN 1
                ELSE 2
              END,
              ap.updated_at DESC
            LIMIT 10
            """,
            (user_id,),
        ).fetchall()
        operation_rows = conn.execute(
            """
            SELECT id, tool_name, operation_type, risk_level, summary,
                   status, conversation_id, updated_at
            FROM agent_pending_operation
            WHERE user_id = %s
            ORDER BY
              CASE WHEN status = 'pending' THEN 0 ELSE 1 END,
              updated_at DESC
            LIMIT 10
            """,
            (user_id,),
        ).fetchall()
        operation_counts = conn.execute(
            """
            SELECT
              COUNT(*) FILTER (WHERE status = 'pending') AS pending_count,
              COUNT(*) FILTER (WHERE status = 'confirmed') AS confirmed_count,
              COUNT(*) FILTER (WHERE status = 'rejected') AS rejected_count
            FROM agent_pending_operation
            WHERE user_id = %s
            """,
            (user_id,),
        ).fetchone()
        link_rows = conn.execute(
            """
            SELECT target_type, COUNT(*)
            FROM project_link
            WHERE user_id = %s
            GROUP BY target_type
            ORDER BY target_type
            """,
            (user_id,),
        ).fetchall()
        project_rollup_rows = conn.execute(
            """
            SELECT p.id, p.name,
                   COUNT(DISTINCT g.id) FILTER (WHERE g.status = 'active') AS active_goals,
                   COUNT(DISTINCT ap.id) FILTER (WHERE ap.status IN ('running', 'awaiting_confirmation')) AS active_plans,
                   COUNT(DISTINCT pl.id) AS links
            FROM project p
            LEFT JOIN goal g ON g.project_id = p.id AND g.user_id = p.user_id
            LEFT JOIN agent_plan ap ON ap.project_id = p.id AND ap.user_id = p.user_id
            LEFT JOIN project_link pl ON pl.project_id = p.id AND pl.user_id = p.user_id
            WHERE p.user_id = %s
            GROUP BY p.id, p.name, p.last_activity_at, p.updated_at
            ORDER BY COALESCE(p.last_activity_at, p.updated_at) DESC
            LIMIT 10
            """,
            (user_id,),
        ).fetchall()

    summary = {
        "todos": {
            "open": int(todo_counts[0] or 0),
            "today": int(todo_counts[1] or 0),
            "overdue": int(todo_counts[2] or 0),
            "weekDue": int(todo_counts[3] or 0),
            "weekDueItems": [
                {
                    "id": row[0],
                    "title": row[1],
                    "dueAt": row[2].isoformat() if row[2] else None,
                    "conversationId": row[3],
                }
                for row in week_due_rows
            ],
            "items": [
                {
                    "id": row[0],
                    "title": row[1],
                    "status": row[2],
                    "dueAt": row[3].isoformat() if row[3] else None,
                    "conversationId": row[4],
                    "updatedAt": row[5].isoformat() if row[5] else None,
                }
                for row in todo_rows
            ],
        },
        "projects": {
            "active": int(project_counts[0] or 0),
            "total": int(project_counts[1] or 0),
            "current": [
                {
                    "id": row[0],
                    "name": row[1],
                    "status": row[2],
                    "currentStage": row[3],
                    "confidence": float(row[4] or 0.0) if len(row) > 4 else None,
                }
                for row in current_project_rows
            ],
            "items": [
                {
                    "id": row[0],
                    "name": row[1],
                    "status": row[2],
                    "currentStage": row[3],
                    "priority": row[4],
                    "lastActivityAt": row[5].isoformat() if row[5] else None,
                    "updatedAt": row[6].isoformat() if row[6] else None,
                }
                for row in project_rows
            ],
            "rollup": [
                {
                    "id": row[0],
                    "name": row[1],
                    "activeGoals": int(row[2] or 0),
                    "activePlans": int(row[3] or 0),
                    "links": int(row[4] or 0),
                }
                for row in project_rollup_rows
            ],
        },
        "goals": {
            "active": int(goal_counts[0] or 0),
            "averageProgress": round(float(goal_counts[1] or 0.0), 3),
            "completed": int(goal_counts[2] or 0),
            "items": [
                {
                    "id": row[0],
                    "projectId": row[1],
                    "projectName": row[2],
                    "title": row[3],
                    "horizon": row[4],
                    "status": row[5],
                    "progress": float(row[6] or 0.0),
                    "dueAt": row[7].isoformat() if row[7] else None,
                    "updatedAt": row[8].isoformat() if row[8] else None,
                }
                for row in goal_rows
            ],
        },
        "plans": {
            "active": int(plan_counts[0] or 0),
            "running": int(plan_counts[1] or 0),
            "awaitingConfirmation": int(plan_counts[2] or 0),
            "completed": int(plan_counts[3] or 0),
            "items": [
                {
                    "id": row[0],
                    "projectId": row[1],
                    "projectName": row[2],
                    "title": row[3],
                    "objective": row[4],
                    "status": row[5],
                    "requiresConfirmation": bool(row[6]),
                    "updatedAt": row[7].isoformat() if row[7] else None,
                }
                for row in plan_rows
            ],
        },
        "operations": {
            "pending": int(operation_counts[0] or 0),
            "confirmed": int(operation_counts[1] or 0),
            "rejected": int(operation_counts[2] or 0),
            "items": [
                {
                    "id": row[0],
                    "toolName": row[1],
                    "operationType": row[2],
                    "riskLevel": row[3],
                    "summary": row[4],
                    "status": row[5],
                    "conversationId": row[6],
                    "updatedAt": row[7].isoformat() if row[7] else None,
                }
                for row in operation_rows
            ],
        },
        "projectLinks": {str(row[0]): int(row[1] or 0) for row in link_rows},
        "range": {
            "todayStart": today_start.isoformat(),
            "todayEnd": today_end.isoformat(),
            "weekStart": week_start.isoformat(),
            "weekEnd": week_end.isoformat(),
            "timezone": str(LOCAL_TIMEZONE),
        },
    }
    return {
        "ok": True,
        "content": _format_work_summary(summary),
        "summary": summary,
    }


def _web_search(user_id: str, conversation_id: str | None, question: str) -> dict[str, object]:
    return search_web(question)


def _weather(user_id: str, conversation_id: str | None, question: str) -> dict[str, object]:
    return get_weather_from_question(question)


def _file_parse(user_id: str, conversation_id: str | None, question: str) -> dict[str, object]:
    return parse_file_from_question(question, user_id=user_id)


def _image_analyze(user_id: str, conversation_id: str | None, question: str) -> dict[str, object]:
    return analyze_image_from_question(question)


def _image_generate(user_id: str, conversation_id: str | None, question: str) -> dict[str, object]:
    return generate_image_from_question(question)


def _image_edit(user_id: str, conversation_id: str | None, question: str) -> dict[str, object]:
    return edit_image_from_question(question)


def _todo(user_id: str, conversation_id: str | None, question: str) -> dict[str, object]:
    return preview_todo_from_question(user_id, conversation_id, question)


def _calendar(user_id: str, conversation_id: str | None, question: str) -> dict[str, object]:
    return preview_calendar_event_from_question(user_id, conversation_id, question)


def _list_tools(user_id: str, conversation_id: str | None, question: str) -> dict[str, object]:
    return {
        "ok": True,
        "content": "可用工具：" + "；".join(
            f"{item['name']}={item['description']} enabled={item['enabled']}"
            for item in list_available_tools()
        ),
        "tools": list_available_tools(),
    }


def _summarize_memory_snapshot(memory: dict[str, object]) -> str:
    summary = str(memory.get("summary") or "无长期摘要")
    structured = memory.get("structured") or []
    short_term = memory.get("short_term") or []
    retrieved = memory.get("retrieved") or []
    return (
        f"长期摘要：{summary}；"
        f"结构化记忆数量：{len(structured)}；"
        f"短期记忆数量：{len(short_term)}；"
        f"向量召回记忆数量：{len(retrieved)}。"
    )


def _allowed_tool_names() -> set[str]:
    return {
        item.strip()
        for item in settings.tool_allowed_names.split(",")
        if item.strip()
    }


def _tool_switch_enabled(tool_name: str) -> bool:
    if tool_name in {"memory_stats", "work_summary"}:
        return settings.tool_database_query_enabled
    if tool_name == "web_search":
        return settings.tool_web_search_enabled
    if tool_name == "weather":
        return settings.tool_weather_enabled
    if tool_name == "file_parse":
        return settings.tool_file_parse_enabled
    if tool_name == "image_analyze":
        return settings.tool_image_analyze_enabled
    if tool_name == "image_generate":
        return settings.tool_image_generate_enabled
    if tool_name == "image_edit":
        return settings.tool_image_edit_enabled
    if tool_name == "todo":
        return settings.tool_todo_enabled
    if tool_name == "calendar":
        return settings.tool_calendar_enabled
    return True


def _local_day_window(now: datetime) -> tuple[datetime, datetime]:
    local_now = now.astimezone(LOCAL_TIMEZONE)
    start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return start.astimezone(UTC), end.astimezone(UTC)


def _local_week_window(now: datetime) -> tuple[datetime, datetime]:
    local_now = now.astimezone(LOCAL_TIMEZONE)
    start = local_now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=local_now.weekday())
    end = start + timedelta(days=7)
    return start.astimezone(UTC), end.astimezone(UTC)


def _current_project_rows(conn, user_id: str, conversation_id: str | None):
    if conversation_id:
        rows = conn.execute(
            """
            SELECT p.id, p.name, p.status, p.current_stage, pl.confidence
            FROM project_link pl
            JOIN project p ON p.id = pl.project_id AND p.user_id = pl.user_id
            WHERE pl.user_id = %s
              AND pl.target_type = 'conversation'
              AND pl.target_id = %s
              AND p.status = 'active'
            ORDER BY pl.confidence DESC, pl.created_at DESC
            LIMIT 3
            """,
            (user_id, conversation_id),
        ).fetchall()
        if rows:
            return rows
    return conn.execute(
        """
        SELECT id, name, status, current_stage, NULL::double precision AS confidence
        FROM project
        WHERE user_id = %s AND status = 'active'
        ORDER BY COALESCE(last_activity_at, updated_at) DESC
        LIMIT 3
        """,
        (user_id,),
    ).fetchall()


def _format_work_summary(summary: dict[str, object]) -> str:
    todos = summary.get("todos") if isinstance(summary.get("todos"), dict) else {}
    projects = summary.get("projects") if isinstance(summary.get("projects"), dict) else {}
    goals = summary.get("goals") if isinstance(summary.get("goals"), dict) else {}
    plans = summary.get("plans") if isinstance(summary.get("plans"), dict) else {}
    operations = summary.get("operations") if isinstance(summary.get("operations"), dict) else {}
    links = summary.get("projectLinks") if isinstance(summary.get("projectLinks"), dict) else {}
    current_projects = projects.get("current") if isinstance(projects.get("current"), list) else []
    current_names = "、".join(str(item.get("name") or "") for item in current_projects if isinstance(item, dict) and item.get("name")) or "暂无"
    week_due_items = todos.get("weekDueItems") if isinstance(todos.get("weekDueItems"), list) else []
    week_due_preview = "；".join(
        str(item.get("title") or "")
        for item in week_due_items[:5]
        if isinstance(item, dict) and item.get("title")
    ) or "暂无"
    todo_preview = _preview_titles(todos.get("items"), "title")
    project_preview = _preview_titles(projects.get("items"), "name")
    goal_preview = _preview_titles(goals.get("items"), "title")
    plan_preview = _preview_titles(plans.get("items"), "title")
    operation_preview = _preview_titles(operations.get("items"), "summary")
    project_rollup = projects.get("rollup") if isinstance(projects.get("rollup"), list) else []
    rollup_preview = "；".join(
        f"{item.get('name')}：目标{item.get('activeGoals', 0)}，计划{item.get('activePlans', 0)}，关联{item.get('links', 0)}"
        for item in project_rollup[:5]
        if isinstance(item, dict) and item.get("name")
    ) or "暂无"
    return (
        "以下来自实时工作数据："
        f"打开待办 {todos.get('open', 0)} 个，今日到期 {todos.get('today', 0)} 个，"
        f"逾期 {todos.get('overdue', 0)} 个，本周到期 {todos.get('weekDue', 0)} 个。"
        f"活跃项目 {projects.get('active', 0)} 个，当前相关项目：{current_names}。"
        f"活跃目标 {goals.get('active', 0)} 个，平均进度 {round(float(goals.get('averageProgress', 0) or 0) * 100)}%。"
        f"长任务计划 {plans.get('active', 0)} 个，其中运行中 {plans.get('running', 0)} 个、待确认 {plans.get('awaitingConfirmation', 0)} 个。"
        f"待确认操作 {operations.get('pending', 0)} 个。"
        f"项目关联 {sum(int(value or 0) for value in links.values()) if isinstance(links, dict) else 0} 条。"
        f"本周到期预览：{week_due_preview}。"
        f"待办列表预览：{todo_preview}。"
        f"项目列表预览：{project_preview}。"
        f"目标列表预览：{goal_preview}。"
        f"计划列表预览：{plan_preview}。"
        f"操作列表预览：{operation_preview}。"
        f"按项目聚合：{rollup_preview}。"
    )


def _preview_titles(items: object, key: str, limit: int = 5) -> str:
    if not isinstance(items, list):
        return "暂无"
    values = [
        str(item.get(key) or "").strip()
        for item in items[:limit]
        if isinstance(item, dict) and str(item.get(key) or "").strip()
    ]
    return "；".join(values) if values else "暂无"


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
