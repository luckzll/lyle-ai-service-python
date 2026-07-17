from __future__ import annotations

import httpx
import pytest

from app.core.config import settings


@pytest.fixture(autouse=True)
def isolate_unit_tests_from_external_services(monkeypatch: pytest.MonkeyPatch):
    """Make the default test environment hermetic.

    Individual contract tests may explicitly opt into a configured provider and
    replace its transport. Any unmocked outbound HTTP call is a test failure,
    never a paid model request or a multi-minute timeout.
    """

    for field in (
        "deepseek_api_key",
        "openai_api_key",
        "qwen_api_key",
        "ark_api_key",
        "tool_web_search_api_key",
        "connector_github_token",
    ):
        monkeypatch.setattr(settings, field, "")

    for field in (
        "chat_model_provider",
        "intent_model_provider",
        "memory_extract_model_provider",
        "summary_model_provider",
        "judge_model_provider",
    ):
        monkeypatch.setattr(settings, field, "deepseek")

    # Keep infrastructure feature flags unchanged: several integration tests
    # intentionally exercise their configured paths with fakes. Credentials are
    # cleared above and the transport guard below prevents accidental remote I/O.

    original_sync_request = httpx.HTTPTransport.handle_request
    original_async_request = httpx.AsyncHTTPTransport.handle_async_request
    loopback_hosts = {"127.0.0.1", "localhost", "::1"}

    def block_sync_request(transport, request: httpx.Request):
        if request.url.host in loopback_hosts:
            return original_sync_request(transport, request)
        raise AssertionError(
            f"Unexpected outbound HTTP request in test: {request.method} {request.url}"
        )

    async def block_async_request(transport, request: httpx.Request):
        if request.url.host in loopback_hosts:
            return await original_async_request(transport, request)
        raise AssertionError(
            f"Unexpected outbound HTTP request in test: {request.method} {request.url}"
        )

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", block_sync_request)
    monkeypatch.setattr(
        httpx.AsyncHTTPTransport,
        "handle_async_request",
        block_async_request,
    )
