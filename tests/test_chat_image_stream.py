from __future__ import annotations

import asyncio
import json
import time

from app.api.chat import _image_chat_stream
from app.schemas.chat import ChatRequest


def _event_sequence(body: str) -> list[tuple[str, dict[str, object]]]:
    events: list[tuple[str, dict[str, object]]] = []
    for block in body.split("\n\n"):
        event = ""
        data = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line.removeprefix("event: ").strip()
            elif line.startswith("data: "):
                data += line.removeprefix("data: ")
        if event and data:
            events.append((event, json.loads(data)))
    return events


async def _response_body(response) -> str:
    chunks: list[str] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
    return "".join(chunks)


def test_image_batch_stream_emits_started_items_and_completion_incrementally(monkeypatch):
    monkeypatch.setattr("app.api.chat.audit_tool_call", lambda **kwargs: None)

    def fake_execute(spec, index):
        time.sleep(0.01 * (3 - index))
        if index == 1:
            raise RuntimeError("provider rate limited")
        return {
            "index": index,
            "status": "COMPLETED",
            "imageUrl": f"https://example.com/result-{index}.png",
            "storageUsed": "remote",
            "storageError": "",
        }

    monkeypatch.setattr("app.tools.image_tool._execute_image_item", fake_execute)
    response = _image_chat_stream(
        ChatRequest(
            userId="batch-user",
            conversationId="batch-conversation",
            turnId="turn-batch-1",
            message="生成 3 张不同构图的赛博朋克猫",
            knowledgeBaseIds=[],
            stream=True,
            aiConfig={},
        ),
        "image_generate",
    )

    events = _event_sequence(asyncio.run(_response_body(response)))
    event_names = [event for event, _ in events]
    assert event_names.index("image.batch.started") < event_names.index("image.item.completed")
    item_events = [
        data
        for event, data in events
        if event in {"image.item.completed", "image.item.failed"}
    ]
    assert [item["index"] for item in item_events] == [2, 1, 0]
    assert item_events[0]["imageUrl"] == "https://example.com/result-2.png"
    assert item_events[1]["status"] == "FAILED"

    completed = next(data for event, data in events if event == "image.batch.completed")
    assert completed == {
        "turnId": "turn-batch-1",
        "batchId": completed["batchId"],
        "status": "PARTIALLY_COMPLETED",
        "count": 3,
        "completedCount": 2,
        "failedCount": 1,
    }
    phases = [
        data["phase"]
        for event, data in events
        if event == "turn.status"
    ]
    assert phases[0] == "THINKING"
    assert "GENERATING" in phases
    assert "RESPONDING" in phases
    assert phases[-1] == "COMPLETED"
