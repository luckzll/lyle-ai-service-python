"""Bounded bridge between blocking agent work and a live SSE response."""
from collections import deque
from collections.abc import Callable, Iterator
from contextvars import copy_context
from threading import BoundedSemaphore, Condition, Thread

from fastapi import HTTPException
from starlette.responses import StreamingResponse

HEARTBEAT_SECONDS = 10.0
_WORKERS = BoundedSemaphore(32)
_END = object()


class _HeartbeatFrames(Iterator[str]):
    """Own the producer lifecycle independently of the iterator's consumption.

    Starlette does not close synchronous iterators on disconnect. ``stop`` must
    therefore be called by the response lifecycle, not a generator finally block.
    A provider already in flight keeps its concurrency slot until it returns.
    """

    def __init__(self, producer, error_frame, interval):
        self._semaphore = _WORKERS
        if not self._semaphore.acquire(blocking=False):
            raise HTTPException(status_code=503, detail="当前生成任务较多，请稍后重试。")
        self._producer = producer
        self._error_frame = error_frame
        self._interval = HEARTBEAT_SECONDS if interval is None else interval
        self._context = copy_context()
        self._condition = Condition()
        self._frames = deque()
        self._started = False
        self._stopped = False
        self._released = False

    def __iter__(self):
        return self

    def __next__(self) -> str:
        with self._condition:
            if self._stopped:
                raise StopIteration
            if not self._started:
                self._started = True
                try:
                    Thread(target=lambda: self._context.run(self._work),
                           name="chat-sse-producer", daemon=True).start()
                except BaseException:
                    self._stopped = True
                    self._release_slot()
                    raise
                return ": keep-alive\n\n"
            available = self._condition.wait_for(
                lambda: self._stopped or bool(self._frames), timeout=self._interval,
            )
            if self._stopped:
                raise StopIteration
            if not available:
                return ": keep-alive\n\n"
            frame = self._frames.popleft()
            self._condition.notify_all()
            if frame is _END:
                self._stopped = True
                raise StopIteration
            return frame

    def _publish(self, frame):
        with self._condition:
            # No polling: consuming a frame or stopping the response wakes us.
            self._condition.wait_for(lambda: self._stopped or len(self._frames) < 128)
            if self._stopped:
                raise GeneratorExit
            self._frames.append(frame)
            self._condition.notify_all()

    def _work(self):
        try:
            self._producer(self._publish)
        except GeneratorExit:
            pass
        except Exception as exc:
            try:
                self._publish(self._error_frame(exc))
            except GeneratorExit:
                pass
        finally:
            try:
                self._publish(_END)
            except GeneratorExit:
                pass
            finally:
                self._release_slot()

    def _release_slot(self):
        with self._condition:
            if not self._released:
                self._released = True
                self._semaphore.release()

    def stop(self):
        with self._condition:
            self._stopped = True
            self._frames.clear()
            self._condition.notify_all()
            if not self._started:
                self._release_slot()

    close = stop

    def __del__(self):
        # Fallback for an abandoned response that was never submitted to ASGI.
        # The actual ASGI disconnect path explicitly stops in __call__ below.
        if hasattr(self, "_condition"):
            self.stop()


def heartbeat_frames(producer: Callable[[Callable[[str], None]], None],
                     error_frame: Callable[[Exception], str], *,
                     interval: float | None = None) -> _HeartbeatFrames:
    """Keep preparation/model work off SSE; retain finite bounded backpressure."""
    return _HeartbeatFrames(producer, error_frame, interval)


class HeartbeatStreamingResponse(StreamingResponse):
    """Stop producers on real ASGI disconnect, send failure, or normal completion."""

    def __init__(self, content: _HeartbeatFrames, *args, **kwargs):
        self._heartbeat_frames = content
        try:
            super().__init__(content, *args, **kwargs)
        except BaseException:
            content.stop()
            raise

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Safe even while a worker thread is inside __next__: unlike closing
            # a running generator, stop only changes shared state and notifies.
            self._heartbeat_frames.stop()

    async def listen_for_disconnect(self, receive):
        try:
            await super().listen_for_disconnect(receive)
        finally:
            # Wake a consumer currently waiting for a heartbeat immediately,
            # before Starlette waits for its cancelled threadpool task to exit.
            self._heartbeat_frames.stop()

    async def stream_response(self, send):
        try:
            await super().stream_response(send)
        finally:
            self._heartbeat_frames.stop()
