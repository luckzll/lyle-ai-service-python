import time
from dataclasses import dataclass
import re
from typing import Callable

from app.core.config import settings
from app.core.database import postgres_connection
from app.memory.store import export_memory
from app.rag.reranker import get_reranker_status
from app.tools.risk_policy import ToolRiskPolicy, get_tool_risk_policy
from app.tools.file_parse_tool import parse_file_from_question
from app.tools.image_tool import analyze_image_from_question, edit_image_from_question, generate_image_from_question
from app.tools.task_tool import preview_calendar_event_from_question, preview_todo_from_question
from app.tools.tool_audit import audit_tool_call
from app.tools.weather_tool import get_weather_from_question
from app.tools.web_search_tool import search_web


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
    if any(keyword in question for keyword in ["健康", "状态", "检查服务", "服务检查"]) or "health" in lowered:
        selected.append("health_check")
    if any(keyword in question for keyword in ["查看记忆", "记忆快照", "我的记忆", "当前记忆"]):
        selected.append("memory_snapshot")
    if any(keyword in question for keyword in ["记忆统计", "记忆数量", "数据库统计", "统计记忆"]):
        selected.append("memory_stats")
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
    if has_image_ref and any(keyword in question for keyword in image_edit_keywords):
        selected.append("image_edit")
    elif (
        any(keyword in question for keyword in ["识别图片", "看图", "这张图", "图片里", "图里", "分析图片", "图片识别"])
        or "image analyze" in lowered
        or "data:image/" in lowered
    ):
        selected.append("image_analyze")
    image_generate_keywords = [
        "生成图片",
        "生图",
        "画一张",
        "画一个",
        "帮我画",
        "生成一张",
        "文生图",
        "再生成",
        "再来一张",
    ]
    image_subject_keywords = [
        "图片",
        "图",
        "照片",
        "海报",
        "头像",
        "壁纸",
        "猫",
        "狗",
        "小猫",
        "小狗",
        "小狗狗",
        "赛博朋克",
        "风格",
    ]
    if (
        any(keyword in question for keyword in image_generate_keywords)
        or "generate image" in lowered
        or (
            any(keyword in question for keyword in ["再来", "换", "改"])
            and any(keyword in question for keyword in image_subject_keywords)
        )
    ):
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
        content = item.get("content")
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
    if tool_name == "memory_stats":
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


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
