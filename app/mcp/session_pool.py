"""Bounded MCP actors. Each SDK context is entered and exited by the same task.

Connections are isolated by tenant, server configuration, and conversation.
Calls are serialized; failed/uncertain calls are never replayed automatically.
"""
import asyncio
from concurrent.futures import Future, TimeoutError as FutureTimeout
from contextlib import AsyncExitStack
import hashlib
import json
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import monotonic

from fastapi import HTTPException

_lock = Lock()
_actors = {}
MAX_SESSIONS = 32
IDLE_SECONDS = 300


class _Actor:
    def __init__(self, key, server):
        self.key, self.server = key, dict(server)
        self.queue = Queue(maxsize=8)
        self.stopped = Event()
        self.thread = Thread(target=self._run, name="lyle-mcp-session", daemon=True)

    def close(self):
        self.stopped.set()
        try:
            self.queue.put_nowait(None)
        except Full:
            pass

    def _run(self):
        try:
            asyncio.run(self._serve())
        except Exception:
            pass  # No SDK stderr, credentials, arguments or result bodies in logs.
        finally:
            self.stopped.set()
            with _lock:
                if _actors.get(self.key) is self:
                    del _actors[self.key]
            while True:
                try:
                    item = self.queue.get_nowait()
                except Empty:
                    break
                if item and not item[2].done():
                    item[2].set_exception(HTTPException(502, "MCP 会话已关闭；不会自动重试未确认结果。"))

    async def _serve(self):
        from app.mcp.client_manager import _session
        async with AsyncExitStack() as stack:
            # asyncio.timeout keeps context entry and exit in this same SDK task.
            async with asyncio.timeout(30):
                session = await stack.enter_async_context(_session(self.server))
            while not self.stopped.is_set():
                try:
                    item = await asyncio.to_thread(self.queue.get, True, IDLE_SECONDS)
                except Empty:
                    break
                if item is None:
                    break
                name, arguments, future = item
                if not future.set_running_or_notify_cancel():
                    continue
                if self.stopped.is_set():
                    future.set_exception(HTTPException(409, "连接授权已失效，工具未执行。"))
                    break
                try:
                    async with asyncio.timeout(60):
                        result = await session.call_tool(name, arguments=arguments)
                    future.set_result(result)
                except BaseException:
                    future.set_exception(HTTPException(502, "MCP 调用中断或结果未知；请核对执行记录，不会自动重试。"))
                    break


def call(server, tenant_scope, conversation_scope, name, arguments):
    from app.mcp.client_manager import parse_config_ref
    fingerprint = hashlib.sha256(json.dumps(parse_config_ref(server), sort_keys=True).encode()).hexdigest()
    owner = (str(tenant_scope), str(server["id"]), str(conversation_scope))
    key = (*owner, fingerprint)
    with _lock:
        for old_key, old_actor in list(_actors.items()):
            if old_key[:3] == owner and old_key != key:
                old_actor.close()
        actor = _actors.get(key)
        if actor is None or actor.stopped.is_set():
            if len(_actors) >= MAX_SESSIONS:
                raise HTTPException(503, "MCP 并发连接已满，请稍后重试。")
            actor = _Actor(key, server)
            _actors[key] = actor
            actor.thread.start()
        future = Future()
        try:
            actor.queue.put_nowait((name, arguments, future))
        except Full:
            raise HTTPException(429, "当前对话的 MCP 工具队列已满。")
    try:
        return future.result(timeout=90)
    except FutureTimeout:
        future.cancel()  # Queued requests cannot start after their caller timed out.
        actor.close()
        raise HTTPException(504, "MCP 调用超时，执行结果未知；不会自动重试。")


def close_server(server_id):
    with _lock:
        for key, actor in list(_actors.items()):
            if key[1] == str(server_id):
                actor.close()


def close_all():
    with _lock:
        actors = list(_actors.values())
        for actor in actors:
            actor.close()
    deadline = monotonic() + 5
    for actor in actors:
        actor.thread.join(timeout=max(0, deadline - monotonic()))
