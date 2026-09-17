from concurrent.futures import ThreadPoolExecutor
import json
from threading import Barrier
from types import SimpleNamespace

import httpx
from langchain_core.messages import AIMessageChunk, HumanMessage
import pytest

from app.chat_reasoning import (
    MAX_REASONING_CHARS, emit_reasoning, finish_reasoning, reasoning_scope,
)
from app.core.config import settings
from app.llm import client
from app.llm.providers.reasoning_chat import ReasoningChatOpenAI
from app.usage.callbacks import UsageCallbackHandler
from app.usage.context import ModelCallContext, set_model_call_context, usage_totals


def _stream(chunks):
    events = []
    with reasoning_scope(events.append):
        for chunk in chunks:
            emit_reasoning(chunk)
        finish_reasoning()
    return events


def test_actual_openai_compatible_wire_retains_reasoning_answer_and_provider_usage(monkeypatch):
    requests = []
    delivered = []
    monkeypatch.setattr("app.usage.callbacks.report_usage_event", delivered.append)
    context = ModelCallContext(user_id="reasoning-user", conversation_id=None,
                               turn_id="reasoning-turn", trace_id="reasoning-trace",
                               reservation_id=None, route_version="test")
    set_model_call_context(context)
    frames = [
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "",
                                               "reasoning_content": "先检查现有实现。"}}]},
        {"choices": [{"index": 0, "delta": {"content": "答案"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 22, "completion_tokens": 15,
                                     "total_tokens": 37}},
    ]

    def respond(request):
        requests.append(json.loads(request.content))
        body = "".join("data: " + json.dumps({"id": "test", "object": "chat.completion.chunk",
                        "created": 1, "model": "test", **frame}) + "\n\n" for frame in frames)
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text=body + "data: [DONE]\n\n")

    try:
        with httpx.Client(transport=httpx.MockTransport(respond)) as http:
            model = ReasoningChatOpenAI(model="test", api_key="test", http_client=http,
                streaming=True, stream_usage=True, max_retries=0,
                callbacks=[UsageCallbackHandler("chat", "deepseek", "test")])
            chunks = list(model.stream([HumanMessage(content="test question")]))
        assert "".join(chunk.content for chunk in chunks) == "答案"
        assert "".join(chunk.additional_kwargs.get("reasoning_content", "") for chunk in chunks) == "先检查现有实现。"
        assert usage_totals(context)["prompt_tokens"] == 22
        assert usage_totals(context)["completion_tokens"] == 15
        assert len(delivered) == 1
        assert requests[0]["stream_options"]["include_usage"] is True
        assert "reasoning" not in requests[0] and "reasoning_effort" not in requests[0]
    finally:
        set_model_call_context(None)


def test_client_keeps_reasoning_and_encrypted_payload_out_of_answer(monkeypatch):
    monkeypatch.setattr(client, "is_chat_model_configured", lambda: True)
    monkeypatch.setattr(client, "get_system_prompt", lambda: ("test", "test"))
    chunks = [
        AIMessageChunk(content="", additional_kwargs={"reasoning_content": "已检查资料。"}),
        AIMessageChunk(content="回答"),
        AIMessageChunk(content=[{"type": "reasoning", "encrypted_content": "private-data"},
                                {"type": "text", "text": "内容"}]),
    ]
    monkeypatch.setattr(client, "build_chat_task_model", lambda **kwargs: SimpleNamespace(stream=lambda _: iter(chunks)))
    events = []
    with reasoning_scope(events.append):
        answer = "".join(client.stream_answer("question", []))
    assert answer == "回答内容"
    assert "".join(item["delta"] for item in events) == "已检查资料。"
    assert "private-data" not in json.dumps(events)


def test_responses_summary_is_distinguished_and_only_plain_answer_text_is_yielded(monkeypatch):
    monkeypatch.setattr(client, "is_chat_model_configured", lambda: True)
    monkeypatch.setattr(client, "get_system_prompt", lambda: ("test", "test"))
    chunks = [AIMessageChunk(content=[
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "检查相关实现。"}],
         "encrypted_content": "encrypted"}, {"type": "text", "text": "正文"},
        {"type": "function_call", "arguments": "private-tool-arguments"},
    ])]
    monkeypatch.setattr(client, "build_chat_task_model", lambda **kwargs: SimpleNamespace(stream=lambda _: iter(chunks)))
    events = []
    with reasoning_scope(events.append):
        assert list(client.stream_answer("question", [])) == ["正文"]
    assert [(item["kind"], item["delta"]) for item in events] == [("summary", "检查相关实现。")]


def test_absent_reasoning_produces_no_display_and_answer_is_unchanged(monkeypatch):
    monkeypatch.setattr(client, "is_chat_model_configured", lambda: True)
    monkeypatch.setattr(client, "get_system_prompt", lambda: ("test", "test"))
    monkeypatch.setattr(client, "build_chat_task_model", lambda **kwargs: SimpleNamespace(
        stream=lambda _: iter([AIMessageChunk(content="hello "), AIMessageChunk(content="world")])) )
    events = []
    with reasoning_scope(events.append):
        assert list(client.stream_answer("question", [])) == ["hello ", "world"]
    assert events == []


@pytest.mark.parametrize("chunk", [
    AIMessageChunk(content="", additional_kwargs={
        "reasoning": {"summary": [{"type": "summary_text", "text": "可见摘要"}]}}),
    AIMessageChunk(content=[{"type": "reasoning", "reasoning": "可见摘要"}],
                   response_metadata={"output_version": "v1", "model_provider": "openai"}),
])
def test_installed_langchain_responses_summary_formats(chunk):
    assert list(client._reasoning_deltas(chunk)) == [("summary", "可见摘要")]


@pytest.mark.parametrize("secret", [
    "sk-abcdefghijklmno12345", "api_key=opaque-credential", "Bearer opaque-credential",
    "Authorization: Bearer opaque-credential", "postgresql://user:opaque-credential@localhost/db",
    "-----BEGIN PRIVATE KEY-----\nopaque-credential\n-----END PRIVATE KEY-----",
])
def test_credentials_split_at_every_boundary_are_masked(secret):
    prefix, suffix = "公开信息。" * 30 + " ", " " + "继续处理。" * 30
    for index in range(1, len(secret)):
        events = _stream([prefix + secret[:index], secret[index:] + suffix])
        text = "".join(item["delta"] for item in events)
        assert "opaque-credential" not in text and "abcdefghijklmno12345" not in text
        assert secret not in text
        assert prefix in text and suffix in text


def test_opaque_configured_secret_is_masked_before_any_fragment_is_published(monkeypatch):
    secret = "opaque-provider-key-that-has-no-known-prefix"
    monkeypatch.setattr(settings, "deepseek_api_key", secret)
    text = "".join(item["delta"] for item in _stream(list("公开。" * 100 + secret + " 完成。" * 100)))
    assert secret not in text
    assert "opaque-provider" not in text
    assert "***" in text


def test_token_adjacent_to_chinese_text_is_masked():
    events = _stream(list("公开内容。" * 30 + "sk-abcdefghijklmno12345继续处理。" * 10))
    text = "".join(item["delta"] for item in events)
    assert "abcdefghijklmno12345" not in text
    assert "继续处理。" in text


def test_incomplete_private_key_is_not_flushed_on_eof_or_cap():
    for tail in ["key-material", "key-material" * 2000]:
        text = "".join(item["delta"] for item in _stream(["公开内容。", "-----BEGIN PRIVATE KEY-----\n", tail]))
        assert "key-material" not in text
        assert "PRIVATE KEY" not in text


def test_display_is_bounded_sequenced_and_fixed_kind():
    events = []
    with reasoning_scope(events.append):
        emit_reasoning("摘要。" * 100, "summary")
        emit_reasoning("should be ignored", "reasoning")
        for _ in range(100):
            emit_reasoning("公开摘要。" * 100, "summary")
        finish_reasoning()
    assert 0 < sum(len(item["delta"]) for item in events) <= MAX_REASONING_CHARS
    assert [item["sequence"] for item in events] == list(range(1, len(events) + 1))
    assert all(item["kind"] == "summary" and item["step"] == "writer_agent" for item in events)
    assert "should be ignored" not in json.dumps(events)


def test_concurrent_request_scopes_cannot_mix_reasoning():
    barrier = Barrier(2)

    def run(text):
        events = []
        with reasoning_scope(events.append):
            emit_reasoning(text)
            barrier.wait(timeout=3)
            finish_reasoning()
        emit_reasoning("outside the scope")
        return events

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, ["request A", "request B"]))
    assert [[event["delta"] for event in items] for items in results] == [["request A"], ["request B"]]
    assert all(items[0]["sequence"] == 1 for items in results)


def test_observer_failure_does_not_fail_model_stream():
    with reasoning_scope(lambda _: (_ for _ in ()).throw(RuntimeError("private"))):
        emit_reasoning("思考摘要。")
        finish_reasoning()
