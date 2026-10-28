import json

import httpx
import pytest

from app.agents import tool_agent
from app.agents.supervisor_agent import run_supervisor_agent
from app.chat_execution_plan import is_self_contained
from app.core.config import settings
from app.tools import web_search_tool
from app.tools.tool_registry import select_tools


PAGE_URL = "https://www.deepseek.com/harness/"


def test_pasted_page_url_routes_to_external_tool():
    question = f"介绍一下这个网站：{PAGE_URL}"
    assert "web_search" in select_tools(question)
    assert not is_self_contained(question)
    assert run_supervisor_agent(question, ["unrelated-kb"])["use_tools"]
    assert "web_search" not in select_tools("帮我看图 https://example.com/a.png 里面有什么")


def test_page_url_uses_tavily_extract_and_bounded_source_content(monkeypatch):
    monkeypatch.setattr(settings, "tool_web_search_provider", "tavily")
    monkeypatch.setattr(settings, "tool_web_search_api_key", "test-key")
    monkeypatch.setattr(settings, "tool_web_search_cache_ttl_seconds", 0)
    real_client = httpx.Client
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"results": [{"url": PAGE_URL, "raw_content": "页面正文" * 4000}], "failed_results": []})

    monkeypatch.setattr(web_search_tool.httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs))
    result = web_search_tool.search_web(f"总结 {PAGE_URL}")

    assert result["ok"]
    assert result["results"][0]["url"] == PAGE_URL
    assert len(result["results"][0]["snippet"]) <= 12020
    assert "已截断" in result["content"]
    assert len(calls) == 1
    assert str(calls[0].url) == "https://api.tavily.com/extract"
    assert calls[0].headers["Authorization"] == "Bearer test-key"
    assert calls[0].read().decode().find(PAGE_URL) >= 0


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/private", "http://169.254.169.254/latest/meta-data/",
    "http://localhost:8080/", "http://service.internal/", "http://user:pass@example.com/",
    "https://example.com:8443/",
])
def test_private_or_credentialed_urls_are_rejected_before_network(monkeypatch, url):
    monkeypatch.setattr(settings, "tool_web_search_provider", "tavily")
    monkeypatch.setattr(settings, "tool_web_search_api_key", "test-key")
    monkeypatch.setattr(web_search_tool, "_extract_tavily", lambda value: pytest.fail("must not call provider"))
    result = web_search_tool.search_web(f"读取 {url}")
    assert not result["ok"]
    assert "公开" in result["content"]


def test_extract_reports_empty_success_as_failure(monkeypatch):
    monkeypatch.setattr(settings, "tool_web_search_provider", "tavily")
    monkeypatch.setattr(settings, "tool_web_search_api_key", "test-key")
    monkeypatch.setattr(settings, "tool_web_search_cache_ttl_seconds", 0)
    real_client = httpx.Client
    monkeypatch.setattr(web_search_tool.httpx, "Client", lambda **kwargs: real_client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"results": [], "failed_results": [{"url": PAGE_URL, "error": "blocked"}]})),
        **kwargs,
    ))
    result = web_search_tool.search_web(PAGE_URL)
    assert not result["ok"]
    assert "未能读取" in result["content"]
    assert "上传截图" in result["content"]
    assert result["error"]["code"] == "PAGE_CONTENT_UNAVAILABLE"
    assert result["pageUrl"] == PAGE_URL
    assert web_search_tool.unavailable_page_answer(PAGE_URL, [{"tool": "web_search", **result}])


@pytest.mark.parametrize("other_context,extra,question", [
    (True, [], PAGE_URL),
    (False, [{"tool": "web_search", "ok": True}], PAGE_URL),
    (False, [{"tool": "todo", "ok": False}], PAGE_URL),
    (False, [], PAGE_URL + " https://example.com/second"),
    (False, [], "https://example.com/different"),
])
def test_missing_page_does_not_replace_other_material_or_tool_work(other_context, extra, question):
    result = {"tool": "web_search", "ok": False, "pageUrl": PAGE_URL,
              "error": {"code": "PAGE_CONTENT_UNAVAILABLE"}}
    assert web_search_tool.unavailable_page_answer(
        question, [result, *extra], has_other_context=other_context,
    ) is None


def test_nonstreaming_graph_also_handles_missing_page_without_a_model(monkeypatch):
    from app.graphs import multi_agent_graph as graph

    monkeypatch.setattr(graph, "run_writer_agent", lambda *args: pytest.fail("missing page has no source to analyze"))
    monkeypatch.setattr("app.agents.reviewer_agent.invoke_reviewer_chain",
                        lambda *args: pytest.fail("fixed guidance needs no model review"))
    monkeypatch.setattr(settings, "reviewer_enabled", True)
    state = {
        "user_id": "page-user", "conversation_id": "page-conversation", "question": PAGE_URL,
        "contexts": [], "memory": {}, "memory_context": "", "tool_context": "",
        "supplemental_context": "", "review_feedback": "", "review_attempts": 0,
        "agent_trace": [], "rag_sufficiency": {},
        "tool_results": [{"tool": "web_search", "ok": False, "pageUrl": PAGE_URL,
                          "error": {"code": "PAGE_CONTENT_UNAVAILABLE"}}],
    }
    written = graph.writer_agent_node(state)
    assert "未能读取" in written["answer"]
    reviewed = graph.reviewer_agent_node(written)
    assert reviewed["review"]["approved"]
    assert reviewed["review"]["modelReviewUsed"] is False


def test_github_trending_request_calls_web_tool_without_model_planning(monkeypatch):
    question = "给我看今天 GitHub Trending 热门榜单"
    assert select_tools(question) == ["web_search"]
    assert run_supervisor_agent(question, [])["use_tools"]
    monkeypatch.setattr(settings, "tool_agent_enabled", True)
    calls = []
    monkeypatch.setattr(tool_agent, "execute_tool", lambda name, user, conversation, text: (
        calls.append((name, text)) or {"tool": name, "ok": True, "content": "今日项目：example/repo"}
    ))
    monkeypatch.setattr("app.agents.native_tool_loop.run_native_loop", lambda *args: pytest.fail("model should not bypass explicit web request"))

    result = tool_agent.run_tool_agent("u1", "c1", question, True)
    assert calls == [("web_search", question)]
    assert result["decision"]["router"] == "direct_web_search"
    assert "example/repo" in result["tool_context"]
    assert not tool_agent._direct_web_search_request("离线说明 GitHub 最新榜单")


def test_github_trending_extracts_relevant_repository_chunks(monkeypatch):
    monkeypatch.setattr(settings, "tool_web_search_provider", "tavily")
    monkeypatch.setattr(settings, "tool_web_search_api_key", "test-key")
    monkeypatch.setattr(settings, "tool_web_search_cache_ttl_seconds", 0)
    real_client = httpx.Client
    requests = []

    def handler(request):
        requests.append(json.loads(request.read()))
        return httpx.Response(200, json={"results": [{"url": "https://github.com/trending", "raw_content": "## owner / repo\n5 stars today"}]})

    monkeypatch.setattr(web_search_tool.httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs))
    result = web_search_tool.search_web("今天 GitHub Trending 热门榜单")
    assert result["ok"]
    assert result["results"][0]["url"] == "https://github.com/trending"
    assert "[owner/repo](https://github.com/owner/repo)" in result["content"]
    assert requests[0]["urls"] == "https://github.com/trending"
    assert requests[0]["chunks_per_source"] == 5
    assert web_search_tool._github_trending_url("本周 GitHub 热榜") == "https://github.com/trending?since=weekly"


def test_web_search_timeout_and_empty_results_are_failures(monkeypatch):
    monkeypatch.setattr(settings, "tool_web_search_provider", "tavily")
    monkeypatch.setattr(settings, "tool_web_search_api_key", "test-key")
    monkeypatch.setattr(settings, "tool_web_search_cache_ttl_seconds", 0)
    monkeypatch.setattr(web_search_tool, "_search_tavily", lambda query: [])
    assert not web_search_tool.search_web("最新新闻")["ok"]
    monkeypatch.setattr(web_search_tool, "_search_tavily", lambda query: (_ for _ in ()).throw(httpx.ConnectTimeout("timeout")))
    result = web_search_tool.search_web("最新新闻")
    assert not result["ok"]
    assert "超时" in result["content"]
