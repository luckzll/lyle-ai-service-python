from __future__ import annotations

import re


def is_memory_overview_question(question: str) -> bool:
    compact = re.sub(r"[\s，。,.!！？?：:]", "", question or "")
    return any(marker in compact for marker in [
        "你记得我什么",
        "你都记得我什么",
        "你对我有什么记忆",
        "你对我了解什么",
        "我的记忆概览",
        "查看我的记忆",
        "记忆概览",
    ])


def build_memory_overview_answer(
    question: str,
    memory: dict[str, object],
    tool_results: list[dict[str, object]],
) -> str | None:
    if not is_memory_overview_question(question):
        return None

    profile_lines = _profile_lines(memory)
    work_summary = _work_summary(tool_results)
    todo_lines = _todo_lines(work_summary)
    operation_lines = _operation_lines(work_summary)

    sections: list[str] = []
    if profile_lines:
        sections.append("已确认记忆：\n" + "\n".join(profile_lines))
    if todo_lines:
        sections.append("任务：\n" + "\n".join(todo_lines))
    if operation_lines:
        sections.append("操作历史：\n" + "\n".join(operation_lines))
    if not sections:
        return "目前没有可用的已确认记忆、任务或操作历史。"
    return "我目前记录到以下信息，并按各对象的真实状态展示：\n\n" + "\n\n".join(sections)


def _profile_lines(memory: dict[str, object]) -> list[str]:
    lines: list[str] = []
    seen_paths: set[str] = set()
    profiles = memory.get("profiles") if isinstance(memory.get("profiles"), list) else []
    for profile in profiles[:8]:
        if not isinstance(profile, dict) or str(profile.get("status") or "") != "active":
            continue
        schema_path = str(profile.get("schemaPath") or "")
        label = str(profile.get("label") or schema_path or "长期记忆")
        value = str(profile.get("summary") or profile.get("value") or "").strip()
        if not value:
            continue
        lines.append(f"- {label}：{value}（已确认记忆）")
        if schema_path:
            seen_paths.add(schema_path)

    structured = memory.get("structured") if isinstance(memory.get("structured"), list) else []
    for item in structured[:8]:
        if not isinstance(item, dict) or str(item.get("status") or "") != "active":
            continue
        key = str(item.get("schemaPath") or item.get("key") or "长期记忆")
        value = str(item.get("value") or "").strip()
        if not value or key in seen_paths or str(item.get("key") or "").startswith("pending:"):
            continue
        lines.append(f"- {key}：{value}（已确认记忆）")
        seen_paths.add(key)
    return lines


def _work_summary(tool_results: list[dict[str, object]]) -> dict[str, object]:
    for item in tool_results:
        if not isinstance(item, dict) or item.get("tool") != "work_summary" or not item.get("ok"):
            continue
        summary = item.get("summary")
        if isinstance(summary, dict):
            return summary
        result = item.get("result")
        if isinstance(result, dict) and isinstance(result.get("summary"), dict):
            return result["summary"]  # type: ignore[return-value]
    return {}


def _todo_lines(summary: dict[str, object]) -> list[str]:
    todos = summary.get("todos") if isinstance(summary.get("todos"), dict) else {}
    items = todos.get("items") if isinstance(todos.get("items"), list) else []
    labels = {
        "open": "进行中",
        "done": "已完成",
        "cancelled": "已取消",
    }
    return [
        f"- {str(item.get('title') or '未命名任务')}（{labels.get(str(item.get('status') or ''), str(item.get('status') or '状态未知'))}）"
        for item in items[:6]
        if isinstance(item, dict)
    ]


def _operation_lines(summary: dict[str, object]) -> list[str]:
    operations = summary.get("operations") if isinstance(summary.get("operations"), dict) else {}
    items = operations.get("items") if isinstance(operations.get("items"), list) else []
    labels = {
        "pending": "待确认，尚未执行",
        "confirmed": "已确认处理",
        "rejected": "已拒绝，未执行",
    }
    return [
        f"- {str(item.get('summary') or '未命名操作')}（{labels.get(str(item.get('status') or ''), str(item.get('status') or '状态未知'))}）"
        for item in items[:6]
        if isinstance(item, dict)
    ]
