import asyncio
from contextvars import ContextVar
from threading import BoundedSemaphore, Event

from fastapi import HTTPException
import pytest
from starlette.background import BackgroundTask

from app import streaming_response as bridge
from app.streaming_response import HeartbeatStreamingResponse, heartbeat_frames


@pytest.fixture
def slots(monkeypatch):
    semaphore = BoundedSemaphore(1)
    monkeypatch.setattr(bridge, "_WORKERS", semaphore)
    return semaphore


def _assert_slot_free(slots):
    assert slots.acquire(timeout=1)
    slots.release()


def test_never_consumed_bridge_close_releases_reserved_slot_once(slots):
    called = Event()
    frames = heartbeat_frames(lambda publish: called.set(), lambda exc: "error")
    with pytest.raises(HTTPException) as error:
        heartbeat_frames(lambda publish: None, lambda exc: "error")
    assert error.value.status_code == 503
    frames.stop()
    frames.close()
    assert list(frames) == []
    assert not called.is_set()
    _assert_slot_free(slots)


def test_normal_completion_preserves_order_context_and_background(slots):
    context = ContextVar("test_request", default=None)
    context.set("request-a")
    seen_contexts, messages, background = [], [], []

    def produce(publish):
        seen_contexts.append(context.get())
        publish("event: delta\ndata: first\n\n")
        publish("event: done\ndata: final\n\n")

    response = HeartbeatStreamingResponse(heartbeat_frames(produce, lambda exc: "error"),
                                         background=BackgroundTask(lambda: background.append("done")))

    async def run():
        async def receive():
            await asyncio.Event().wait()
        async def send(message):
            messages.append(message)
        await response({"type": "http"}, receive, send)

    asyncio.run(run())
    bodies = [message["body"] for message in messages if message["type"] == "http.response.body"]
    assert bodies == [b": keep-alive\n\n", b"event: delta\ndata: first\n\n", b"event: done\ndata: final\n\n", b""]
    assert seen_contexts == ["request-a"]
    assert background == ["done"]
    _assert_slot_free(slots)


def test_producer_error_is_sent_then_stream_completes(slots):
    def produce(publish):
        publish("first")
        raise TimeoutError("private provider detail")

    frames = heartbeat_frames(produce, lambda error: "event: error\ndata: safe timeout\n\n")
    assert list(frames) == [": keep-alive\n\n", "first", "event: error\ndata: safe timeout\n\n"]
    _assert_slot_free(slots)


def test_waiting_provider_emits_heartbeat_and_keeps_slot_until_call_returns(slots):
    release_call, entered, producer_finished = Event(), Event(), Event()

    def produce(publish):
        try:
            entered.set()
            assert release_call.wait(2)
            publish("late result")
        finally:
            producer_finished.set()

    frames = heartbeat_frames(produce, lambda exc: "error", interval=.01)
    assert next(frames) == ": keep-alive\n\n"
    assert entered.wait(1)
    assert next(frames) == ": keep-alive\n\n"
    frames.stop()
    assert not slots.acquire(blocking=False)
    release_call.set()
    assert producer_finished.wait(1)
    _assert_slot_free(slots)


def test_backpressure_is_bounded_and_stop_wakes_blocked_publish(slots):
    full, finished, published_extra = Event(), Event(), Event()

    def produce(publish):
        try:
            for index in range(128):
                publish(str(index))
            full.set()
            publish("overflow")
            published_extra.set()
        finally:
            finished.set()

    frames = heartbeat_frames(produce, lambda exc: "error")
    assert next(frames) == ": keep-alive\n\n"
    assert full.wait(1)
    assert not published_extra.wait(.03)
    frames.stop()
    assert finished.wait(1)
    assert not published_extra.is_set()
    _assert_slot_free(slots)


def test_real_asgi_disconnect_stops_full_queue_without_closing_iterator(slots):
    full, finished = Event(), Event()

    def produce(publish):
        try:
            for index in range(128):
                publish(str(index))
            full.set()
            publish("blocked frame")
        finally:
            finished.set()

    frames = heartbeat_frames(produce, lambda exc: "error")
    response = HeartbeatStreamingResponse(frames)

    async def run():
        body_sent = asyncio.Event()
        async def send(message):
            if message["type"] == "http.response.body":
                body_sent.set()
                # Simulate a slow/disconnected downstream send.
                await asyncio.Event().wait()
        async def receive():
            await body_sent.wait()
            assert await asyncio.to_thread(full.wait, 1)
            return {"type": "http.disconnect"}
        await asyncio.wait_for(response({"type": "http"}, receive, send), timeout=1.5)

    asyncio.run(run())
    # Keep both response and frames alive: success cannot depend on CPython GC.
    assert response._heartbeat_frames is frames
    assert finished.wait(1)
    assert list(frames) == []
    _assert_slot_free(slots)


def test_disconnect_before_first_body_never_starts_producer(slots):
    called = Event()
    frames = heartbeat_frames(lambda publish: called.set(), lambda exc: "error")
    response = HeartbeatStreamingResponse(frames)

    async def run():
        headers_sent = asyncio.Event()
        async def send(message):
            headers_sent.set()
            await asyncio.Event().wait()
        async def receive():
            await headers_sent.wait()
            return {"type": "http.disconnect"}
        await asyncio.wait_for(response({"type": "http"}, receive, send), timeout=1)

    asyncio.run(run())
    assert not called.is_set()
    assert list(frames) == []
    _assert_slot_free(slots)


def test_asgi_send_failure_releases_unconsumed_reservation(slots):
    frames = heartbeat_frames(lambda publish: None, lambda exc: "error")
    response = HeartbeatStreamingResponse(frames)

    async def run():
        async def receive():
            await asyncio.Event().wait()
        async def send(message):
            raise OSError("downstream closed")
        with pytest.raises(BaseExceptionGroup):
            await response({"type": "http"}, receive, send)

    asyncio.run(run())
    _assert_slot_free(slots)


def test_thread_start_failure_releases_reservation(monkeypatch, slots):
    class FailedThread:
        def __init__(self, **kwargs):
            pass
        def start(self):
            raise RuntimeError("thread unavailable")
    monkeypatch.setattr(bridge, "Thread", FailedThread)
    frames = heartbeat_frames(lambda publish: None, lambda exc: "error")
    with pytest.raises(RuntimeError):
        next(frames)
    frames.stop()
    _assert_slot_free(slots)
