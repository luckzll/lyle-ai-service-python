"""Regressions for the production image transform and screenshot follow-up."""
import asyncio
import json

import httpx
import pytest

from app.api.chat import _stream_image_tool, chat_stream
from app.schemas.chat import ChatRequest
from app.tools import image_tool
from app.tools.image_intent import is_image_generation_request
from app.usage.callbacks import UsageBudgetExceeded
from app.usage.context import ModelCallContext, set_model_call_context


@pytest.fixture(autouse=True)
def reset_context():
    set_model_call_context(None)
    yield
    set_model_call_context(None)


@pytest.mark.parametrize("question", [
    "给我整理成一个图片", "请帮我整理成一张图片", "把上述内容整理成一张架构图",
    "把这些要点汇总成一张海报", "帮我做成一张流程图",
])
def test_transform_requests_invoke_real_image_tools(question):
    assert is_image_generation_request(question)
    assert _stream_image_tool(question) == "image_generate"
    assert _stream_image_tool(question + "\n图片1：https://example.com/source.png") == "image_edit"


@pytest.mark.parametrize("question", [
    "解释如何把内容整理成图片", "把这段话整理成一个段落", "为什么图片没有展示出来什么原因",
    "整理成图片需要多少钱？", "不要把内容整理成图片", "给我整理图片里的文字",
])
def test_image_mentions_do_not_trigger_generation(question):
    assert not is_image_generation_request(question)


def context(limit=16000):
    return ModelCallContext(user_id="regression", conversation_id="c", turn_id="t",
                            trace_id="test", reservation_id=None, route_version="test",
                            max_input_tokens=limit)


def payload(image, text="为什么图片没有展示出来什么原因"):
    return {"model": "test-vision", "messages": [{"role": "user", "content": [
        {"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": image}},
    ]}]}


def test_large_screenshot_reaches_provider_and_records_provider_usage(monkeypatch):
    request = payload("data:image/png;base64," + "A" * 1_000_000)
    calls = []
    transport = httpx.MockTransport(lambda req: (
        calls.append(json.loads(req.content)),
        httpx.Response(200, json={"choices": [{"message": {"content": "截图分析"}}],
                                 "usage": {"prompt_tokens": 1500, "completion_tokens": 12}}),
    )[1])
    client_type = httpx.Client
    monkeypatch.setattr(image_tool.httpx, "Client", lambda **kwargs: client_type(transport=transport, **kwargs))
    monkeypatch.setattr(image_tool, "_image_api_key", lambda: "test-key")
    monkeypatch.setattr("app.usage.manual.report_usage_event", lambda event: None)
    usage = context()
    set_model_call_context(usage)

    result = image_tool._post_json("/chat/completions", request)

    assert result["usage"]["prompt_tokens"] == 1500
    assert calls == [request]  # The image is still sent intact to the provider.
    assert usage.input_tokens == 1500
    assert usage.output_tokens == 12
    assert usage.pending_input_tokens == 0
    assert usage.events[0]["usageSource"] == "PROVIDER_REPORTED"
    assert image_tool._estimate_image_request_tokens(request) == image_tool._estimate_image_request_tokens(
        payload("https://example.com/screenshot.png"))


def test_actual_text_and_image_budget_still_enforced():
    set_model_call_context(context(limit=100))
    with pytest.raises(UsageBudgetExceeded, match="输入上下文"):
        image_tool._post_json("/chat/completions", payload("https://example.com/screenshot.png"))
    set_model_call_context(context())
    with pytest.raises(UsageBudgetExceeded, match="输入上下文"):
        image_tool._post_json("/chat/completions", payload("data:image/png;base64,AAAA", "文" * 16001))


async def body(response):
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
    return "".join(chunks)


def test_budget_error_is_explicit_failed_terminal_stream(monkeypatch):
    def fail(_question):
        raise UsageBudgetExceeded("本轮输入上下文超过上限。")

    monkeypatch.setattr("app.api.chat.analyze_image_from_question", fail)
    monkeypatch.setattr("app.api.chat.audit_tool_call", lambda **kwargs: None)
    response = chat_stream(ChatRequest(userId="regression", conversationId="c", turnId="t",
                                      message="分析图片 https://example.com/screenshot.png"))
    frames = asyncio.run(body(response))
    assert "本轮输入上下文超过上限。" in frames
    assert '"phase": "FAILED"' in frames
    assert '"phase": "COMPLETED"' not in frames
    assert "event: error" in frames
    assert frames.rstrip().endswith('data: {"turnId": "t"}')
    assert frames.index("event: error") < frames.index("event: done")


def test_vision_answer_cannot_render_invented_image_artifact(monkeypatch):
    monkeypatch.setattr("app.api.chat.analyze_image_from_question", lambda question: {
        "ok": True, "content": "分析结果\n![](https://space.coze.cn/s/fake/)"})
    monkeypatch.setattr("app.api.chat.audit_tool_call", lambda **kwargs: None)
    response = chat_stream(ChatRequest(userId="regression", conversationId="c",
                                      message="分析图片 https://example.com/screenshot.png"))
    frames = asyncio.run(body(response))
    assert "分析结果" in frames
    assert "space.coze.cn" not in frames
