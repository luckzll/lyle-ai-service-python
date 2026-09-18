"""Request-local factual tool lifecycle events, with no arguments or results.

LangGraph/AnyIO copy ContextVars into workers. The shared display state owns
ordering and limits; the nested-call marker suppresses native-to-legacy wrappers.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import logging
import re
from threading import RLock
from time import perf_counter
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

MAX_TOOL_CALLS = 128
_TOOL_NAME = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_MCP_COMPONENT = re.compile(r"[A-Za-z0-9_.:-]+\Z")
_TITLES = {
    "weather": "查询天气", "web_search": "搜索网页", "knowledge_search": "检索知识库",
    "file_parse": "读取文件", "chat_attachment_parse": "解析附件", "health_check": "检查服务状态",
    "memory_snapshot": "查看记忆", "memory_stats": "统计记忆", "work_summary": "汇总工作信息",
    "list_tools": "查看可用工具", "todo": "处理待办", "calendar": "处理日程",
    "image_analyze": "分析图片", "image_generate": "生成图片", "image_edit": "编辑图片",
    "request_clarification": "确认所需信息",
    "attachment_list": "查看上传目录", "attachment_read": "读取上传文件", "attachment_search": "检索上传文件",
}
_SUMMARIES = {
    "running": "正在处理工具请求",
    "completed": "已获得工具结果",
    "failed": "工具调用未成功，请查看最终说明",
    "awaiting_confirmation": "操作预览已准备，等待确认后执行",
    "needs_clarification": "需要补充信息，尚未执行目标操作",
    "interrupted": "调用已中断，尚未确认执行结果",
}
logger = logging.getLogger(__name__)


def mcp_display_name(selected: dict) -> str:
    """Expose only a registered identifier, never MCP descriptions or payloads."""
    identity = str(selected.get("id", ""))
    base = f"mcp_{identity}" if re.fullmatch(r"[0-9]{1,64}", identity) else "mcp_tool"
    name = selected.get("toolName")
    if isinstance(name, str) and _MCP_COMPONENT.fullmatch(name):
        return f"{base}:{name}"[:128]
    return base


class ToolDisplay:
    def __init__(self, callback: Callable[[dict[str, object]], None]):
        self.callback = callback
        self.lock = RLock()
        self.sequence = 0
        self.count = 0
        self.active: dict[str, ToolInvocation] = {}
        self.closed = False

    def _emit(self, call: ToolInvocation, status: str) -> None:
        self.sequence += 1
        event: dict[str, object] = {
            "sequence": self.sequence, "callId": call.call_id, "tool": call.tool,
            "title": _TITLES.get(call.tool, "调用外部工具" if call.tool.startswith("mcp_") else "调用工具"),
            "status": status, "summary": _SUMMARIES[status],
        }
        if status != "running":
            event["durationMs"] = round(min(86_400_000, max(0, (perf_counter() - call.started) * 1000)), 3)
        try:
            self.callback(event)
        except Exception:
            logger.warning("Chat tool observer unavailable; task continues.")

    def start(self, tool: str) -> ToolInvocation:
        with self.lock:
            if self.closed or self.count >= MAX_TOOL_CALLS:
                return ToolInvocation()
            # Invalid model/extension names must never cross the display boundary.
            name = tool if isinstance(tool, str) and _TOOL_NAME.fullmatch(tool) else "unavailable_tool"
            call = ToolInvocation(self, uuid4().hex, name, perf_counter())
            self.count += 1
            self.active[call.call_id] = call
            self._emit(call, "running")
            return call

    def finish(self, call: ToolInvocation, status: str) -> None:
        with self.lock:
            if self.closed or call.call_id not in self.active or status not in _SUMMARIES or status == "running":
                return
            self.active.pop(call.call_id)
            self._emit(call, status)

    def close(self, status: str) -> None:
        with self.lock:
            try:
                for call in list(self.active.values()):
                    self.finish(call, status)
            finally:
                self.closed = True
                self.active.clear()


@dataclass
class ToolInvocation:
    display: ToolDisplay | None = None
    call_id: str = ""
    tool: str = ""
    started: float = 0

    def finish_status(self, status: str) -> None:
        if self.display is not None:
            self.display.finish(self, status)

    def finish(self, result: object) -> None:
        value = result if isinstance(result, dict) else {}
        operation = value.get("operation") if isinstance(value.get("operation"), dict) else {}
        if value.get("needsClarification"):
            status = "needs_clarification"
        elif value.get("requiresConfirmation") and not value.get("policyBlocked") and (
            value.get("ok") or operation.get("status") == "pending"
        ):
            status = "awaiting_confirmation"
        elif operation.get("status") == "failed":
            status = "failed"
        else:
            status = "completed" if value.get("ok") else "failed"
        self.finish_status(status)


_display: ContextVar[ToolDisplay | None] = ContextVar("chat_tool_display", default=None)
_nested: ContextVar[bool] = ContextVar("chat_tool_nested", default=False)


@contextmanager
def tool_scope(callback: Callable[[dict[str, object]], None]) -> Iterator[None]:
    display = ToolDisplay(callback)
    token = _display.set(display)
    nested_token = _nested.set(False)
    terminal = "interrupted"
    try:
        yield
    except Exception:
        terminal = "failed"
        raise
    finally:
        try:
            display.close(terminal)
        finally:
            _nested.reset(nested_token)
            _display.reset(token)


def start_tool_call(tool: str) -> ToolInvocation:
    display = _display.get()
    return display.start(tool) if display is not None and not _nested.get() else ToolInvocation()


@contextmanager
def tool_invocation(tool: str) -> Iterator[ToolInvocation]:
    call = start_tool_call(tool)
    token = _nested.set(True)
    try:
        yield call
    except ValidationError:
        call.finish_status("needs_clarification")
        raise
    except BaseException as error:
        call.finish_status("failed" if isinstance(error, Exception) else "interrupted")
        raise
    finally:
        try:
            # Covers an abandoned execution path without inventing success.
            call.finish_status("interrupted")
        finally:
            _nested.reset(token)


def run_observed_tool(tool: str, operation: Callable[[], Any]) -> Any:
    with tool_invocation(tool) as call:
        result = operation()
        call.finish(result)
        return result
