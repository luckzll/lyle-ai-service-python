from __future__ import annotations

import asyncio
import json
import time
from uuid import uuid4

import pytest

from app.api.chat import _image_chat_stream, chat_stream
from app.core.config import settings
from app.memory import store
from app.schemas.chat import ChatAttachment, ChatAttachmentStorage, ChatRequest
from app.usage.context import activate_model_call_context, set_model_call_context
from app.usage.manual import begin_external_call, finish_external_call


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


@pytest.mark.parametrize('tool_name', ['image_analyze', 'image_generate', 'image_edit'])
def test_image_stream_returns_pending_memories_from_user_text(monkeypatch, tool_name):
    for flag in ('postgres_enabled', 'redis_enabled', 'qdrant_enabled', 'milvus_enabled'):
        monkeypatch.setattr(settings, flag, False)
    monkeypatch.setattr(settings, 'structured_memory_extractor', 'rule')
    monkeypatch.setattr('app.api.chat.audit_tool_call', lambda **kwargs: None)
    monkeypatch.setattr('app.api.chat.analyze_image_from_question', lambda text: {
        'ok': True, 'content': '图片中的人物没有戴帽子。',
    })
    monkeypatch.setattr('app.tools.image_tool._execute_image_item', lambda spec, index: {
        'index': index, 'status': 'COMPLETED', 'imageUrl': 'https://example.com/result.png',
        'storageUsed': 'remote', 'storageError': '',
    })
    user_id = 'image-memory-' + uuid4().hex
    request = ChatRequest(
        userId=user_id, conversationId='image-memory-conversation', sourceMessageId='image-memory-source',
        message='请记住：我不喜欢帽子。请处理这张图片 https://example.com/source.png',
    )
    response = _image_chat_stream(
        request, tool_name, attachment_context='附件原文：我的职业是医生。',
        usage_context=activate_model_call_context(request),
    )
    try:
        events = _event_sequence(asyncio.run(_response_body(response)))
    finally:
        set_model_call_context(None)

    meta = next(data for event, data in events if event == 'meta')
    candidates = meta.get('candidateMemories', [])
    assert len(candidates) == 1
    assert candidates[0]['schemaPath'] == 'preferences.visual.dislikes'
    assert candidates[0]['value'] == '帽子'
    assert candidates[0]['status'] == 'pending'
    assert ';message:image-memory-source' in candidates[0]['source']
    assert '医生' not in str(candidates)
    snapshot = store.export_memory(user_id, request.conversation_id)
    assert any(item['key'] == candidates[0]['key'] for item in snapshot['structured_all'])
    assert snapshot['profiles'] == []
    answer = ''.join(str(data['text']) for event, data in events if event == 'delta')
    expected_content = '图片中的人物没有戴帽子' if tool_name == 'image_analyze' else 'https://example.com/result.png'
    assert expected_content in answer


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
    request = ChatRequest(
        userId="batch-user",
        conversationId="batch-conversation",
        turnId="turn-batch-1",
        message="生成 3 张不同构图的赛博朋克猫",
        knowledgeBaseIds=[],
        stream=True,
        aiConfig={},
    )
    usage_context = activate_model_call_context(request)
    response = _image_chat_stream(
        request,
        "image_generate",
        usage_context=usage_context,
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
    set_model_call_context(None)


@pytest.mark.parametrize(
    ("tool_name", "message", "usage_source"),
    [
        ("image_analyze", "分析 https://example.com/source.png", "PROVIDER_REPORTED"),
        ("image_generate", "生成一张赛博朋克猫", "ESTIMATED"),
        ("image_edit", "把 https://example.com/source.png 的背景改成蓝色", "ESTIMATED"),
    ],
)
def test_image_stream_reports_bound_model_usage_before_done(
    monkeypatch,
    tool_name: str,
    message: str,
    usage_source: str,
):
    monkeypatch.setattr("app.api.chat._stream_image_tool", lambda _message: tool_name)
    monkeypatch.setattr("app.api.chat.audit_tool_call", lambda **kwargs: None)
    monkeypatch.setattr("app.usage.manual.report_usage_event", lambda _event: None)

    def record_usage() -> None:
        ticket = begin_external_call(estimated_input_tokens=11, estimated_cost_micros=17)
        assert ticket is not None
        finish_external_call(
            ticket,
            call_type="LLM_VISION" if tool_name == "image_analyze" else "IMAGE_GENERATION",
            task_type=tool_name,
            provider="test-provider",
            model="test-model",
            input_tokens=11,
            output_tokens=5,
            usage_source=usage_source,
            estimated_cost_micros=17,
        )

    if tool_name == "image_analyze":
        def fake_analyze(_question: str) -> dict[str, object]:
            record_usage()
            return {"ok": True, "content": "图片分析完成。"}

        monkeypatch.setattr("app.api.chat.analyze_image_from_question", fake_analyze)
    else:
        def fake_execute(_spec, index):
            record_usage()
            return {
                "index": index,
                "status": "COMPLETED",
                "imageUrl": f"https://example.com/{tool_name}.png",
                "storageUsed": "remote",
                "storageError": "",
            }

        # Keep the real iterator: it copies the generator ContextVar into the
        # image worker thread, which is part of the regression being tested.
        monkeypatch.setattr("app.tools.image_tool._execute_image_item", fake_execute)

    request = ChatRequest(
        userId="usage-user",
        conversationId="usage-conversation",
        turnId=f"turn-{tool_name}",
        message=message,
        knowledgeBaseIds=[],
        stream=True,
        aiConfig={},
    )
    response = chat_stream(request)
    # StreamingResponse can consume the generator in another context. Clearing
    # the caller context proves the generator explicitly restores its own one.
    set_model_call_context(None)
    try:
        events = _event_sequence(asyncio.run(_response_body(response)))
    finally:
        set_model_call_context(None)

    event_names = [event for event, _ in events]
    assert event_names.count("meta") == 1
    calls = [data for event, data in events if event == "tool.call"]
    assert [item["status"] for item in calls] == ["running", "completed"]
    assert calls[0]["callId"] == calls[1]["callId"]
    assert all(item["tool"] == tool_name and item["turnId"] == f"turn-{tool_name}" for item in calls)
    assert event_names.index("meta") < event_names.index("done")
    meta = next(data for event, data in events if event == "meta")
    assert meta["debugVersion"] == "chat-debug-v1"
    assert meta["usage"] == {
        "promptTokens": 11,
        "completionTokens": 5,
        "callCount": 1,
        "estimatedCostMicros": 17,
        "measurementSource": usage_source,
    }


@pytest.mark.parametrize(
    ("tool_name", "message"),
    [
        ("image_analyze", "分析 https://example.com/source.png"),
        ("image_generate", "生成一张赛博朋克猫"),
        ("image_edit", "把 https://example.com/source.png 的背景改成蓝色"),
    ],
)
def test_image_stream_fatal_error_does_not_claim_complete_debug_snapshot(
    monkeypatch,
    tool_name: str,
    message: str,
):
    monkeypatch.setattr("app.api.chat._stream_image_tool", lambda _message: tool_name)
    monkeypatch.setattr("app.api.chat.audit_tool_call", lambda **kwargs: None)

    if tool_name == "image_analyze":
        def fail_analyze(_question: str):
            raise RuntimeError("vision provider unavailable")

        monkeypatch.setattr("app.api.chat.analyze_image_from_question", fail_analyze)
    else:
        def fail_batch(_spec):
            raise RuntimeError("image provider unavailable")
            yield  # pragma: no cover - keeps this a generator

        monkeypatch.setattr("app.api.chat.iter_image_batch_results", fail_batch)

    response = chat_stream(ChatRequest(
        userId="failed-user",
        conversationId="failed-conversation",
        turnId=f"failed-{tool_name}",
        message=message,
        knowledgeBaseIds=[],
        stream=True,
        aiConfig={},
    ))
    try:
        events = _event_sequence(asyncio.run(_response_body(response)))
    finally:
        set_model_call_context(None)

    event_names = [event for event, _ in events]
    assert "error" in event_names
    calls = [data for event, data in events if event == "tool.call"]
    assert [item["status"] for item in calls] == ["running", "failed"]
    assert "provider unavailable" not in json.dumps(calls)
    assert "meta" not in event_names
    assert event_names[-1] == "done"
    assert not any(
        event == "turn.status" and data.get("phase") == "COMPLETED"
        for event, data in events
    )
    assert any(
        event == "turn.status" and data.get("phase") == "FAILED"
        for event, data in events
    )
    assert "chat-debug-v1" not in json.dumps(events, ensure_ascii=False)


def test_mixed_image_and_file_stream_includes_parsed_file_in_image_request(monkeypatch, tmp_path):
    image_path = tmp_path / "reference.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    file_path = tmp_path / "requirements.txt"
    file_path.write_text("验收口径：图片中的版本号必须与 V6.2 一致。", encoding="utf-8")
    captured: dict[str, str] = {}

    def fake_analyze(question: str):
        captured["question"] = question
        return {"ok": True, "content": "图片版本号为 V6.2，与文件一致。"}

    monkeypatch.setattr("app.api.chat.analyze_image_from_question", fake_analyze)
    monkeypatch.setattr("app.api.chat.audit_tool_call", lambda **kwargs: None)
    response = chat_stream(ChatRequest(
        userId="mixed-user",
        conversationId="mixed-conversation",
        turnId="mixed-turn",
        message="请分析图片并对照文件内容",
        attachments=[
            _chat_attachment(image_path, "image-1", "image", "image/png"),
            _chat_attachment(file_path, "file-1", "file", "text/plain"),
        ],
    ))

    events = _event_sequence(asyncio.run(_response_body(response)))

    assert str(image_path) in captured["question"]
    assert "验收口径：图片中的版本号必须与 V6.2 一致。" in captured["question"]
    meta = next(data for event, data in events if event == "meta")
    assert [item["tool"] for item in meta["toolResults"]] == [
        "chat_attachment_parse",
        "chat_attachment_parse",
        "image_analyze",
    ]
    assert meta["toolResults"][0]["attachmentId"] == "image-1"
    assert meta["toolResults"][0]["status"] == "image"
    assert meta["toolResults"][1]["attachmentId"] == "file-1"
    assert meta["toolResults"][1]["extractedChars"] > 0


def _chat_attachment(path, attachment_id: str, kind: str, mime_type: str) -> ChatAttachment:
    return ChatAttachment(
        id=attachment_id,
        kind=kind,
        name=path.name,
        mimeType=mime_type,
        size=path.stat().st_size,
        storage=ChatAttachmentStorage(
            storagePath=path.as_uri(),
            storageProvider="LOCAL",
            originalName=path.name,
            contentType=mime_type,
            fileSize=path.stat().st_size,
        ),
    )
