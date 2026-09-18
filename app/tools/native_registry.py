"""Model-facing schemas and typed dispatch over the existing execution boundary."""
import json

from app.chat_tools import run_observed_tool

from app.tools.slots import ARGUMENT_MODELS, ClarifyArgs, slot_records
from app.tools import tool_registry as legacy
from app.tools.operation_confirmation import create_pending_operation


def tool_schemas() -> list[dict]:
    definitions = []
    for item in legacy.list_available_tools():
        if item["enabled"] and item["name"] in ARGUMENT_MODELS:
            definitions.append({"type": "function", "function": {
                "name": item["name"], "description": item["description"],
                "parameters": ARGUMENT_MODELS[item["name"]].model_json_schema(),
            }})
    definitions.append({"type": "function", "function": {
        "name": "request_clarification", "description": "缺少必填字段或目标有歧义时询问用户；禁止猜测收件人、文件、日期。",
        "parameters": ClarifyArgs.model_json_schema(),
    }})
    return definitions


def execute_native(name: str, raw: dict, user_id: str, conversation_id: str | None, question: str) -> dict:
    return run_observed_tool(name, lambda: _execute_native(name, raw, user_id, conversation_id, question))


def _execute_native(name: str, raw: dict, user_id: str, conversation_id: str | None, question: str) -> dict:
    available = {item["name"] for item in legacy.list_available_tools() if item["enabled"]}
    if name not in available or name not in ARGUMENT_MODELS:
        raise ValueError("工具未启用或不在白名单内。")
    args = ARGUMENT_MODELS[name].model_validate(raw)
    values = args.model_dump(mode="json")
    if name in {"todo", "calendar"}:
        if name == "calendar" and args.participants:
            return {"tool": name, "ok": False, "needsClarification": True,
                    "content": "内置日程暂不支持邀请参与者，请选择已连接的日历工具，或明确仅创建个人日程。"}
        payload = {"title": args.title}
        payload["dueAt" if name == "todo" else "startsAt"] = values.get("due_at" if name == "todo" else "starts_at")
        result = create_pending_operation(
            user_id=user_id, conversation_id=conversation_id, tool_name=name,
            operation_type=f"{name}.create", summary=json.dumps(payload, ensure_ascii=False),
            payload=payload, source_text=question,
        )
    elif name == "weather":
        from app.tools.weather_tool import get_weather
        result = get_weather(args.city, values["date"])
    elif name == "file_parse":
        from app.tools.file_parse_tool import parse_file
        result = parse_file(args.file_ref, user_id)
    else:
        text = values.get("query", values.get("prompt", question))
        if name == "web_search" and args.time_range:
            text += f" 时间范围：{args.time_range}"
        if name in {"image_analyze", "image_edit"}:
            # References must come from the user's request, not from a model invention.
            if any(ref not in question for ref in args.image_refs):
                raise ValueError("图片引用必须来自用户本轮提供的附件。")
            text = args.instruction + "\n" + "\n".join(args.image_refs)
        result = legacy.execute_tool(name, user_id, conversation_id, text)
    result = legacy._normalise_tool_result(name, result)
    result["slots"] = slot_records(args, question)
    if name in {"todo", "calendar", "weather", "file_parse"}:
        legacy.audit_tool_call(tool_name=name, user_id=user_id, conversation_id=conversation_id,
                               question=question, ok=bool(result.get("ok")),
                               content=str(result.get("content") or ""), elapsed_ms=0)
    return result
