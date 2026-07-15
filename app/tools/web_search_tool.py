from urllib.parse import urlparse
from time import time

import httpx

from app.core.config import settings

_SEARCH_CACHE: dict[str, tuple[float, dict[str, object]]] = {}


def search_web(query: str) -> dict[str, object]:
    provider = settings.tool_web_search_provider.strip().lower()
    if not provider:
        return {
            "ok": False,
            "content": "网页搜索 provider 未配置。支持 tavily、serpapi、bing。",
            "results": [],
        }
    if not settings.tool_web_search_api_key:
        return {
            "ok": False,
            "content": f"网页搜索 provider={provider} 已配置，但缺少 API Key。",
            "results": [],
        }

    cache_key = f"{provider}:{settings.tool_web_search_top_k}:{query.strip().lower()}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return {**cached, "cached": True}

    if provider == "tavily":
        results = _search_tavily(query)
    elif provider == "serpapi":
        results = _search_serpapi(query)
    elif provider == "bing":
        results = _search_bing(query)
    else:
        return {
            "ok": False,
            "content": f"不支持的网页搜索 provider：{provider}。支持 tavily、serpapi、bing。",
            "results": [],
        }

    results = _trim_results(_dedupe_results(results))[: settings.tool_web_search_top_k]
    payload = {
        "ok": True,
        "content": _format_results(results),
        "results": results,
        "provider": provider,
        "cached": False,
    }
    _set_cached(cache_key, payload)
    return payload


def _search_tavily(query: str) -> list[dict[str, str]]:
    with httpx.Client(timeout=15) as client:
        response = client.post(
            settings.tool_web_search_base_url.strip() or "https://api.tavily.com/search",
            json={
                "api_key": settings.tool_web_search_api_key,
                "query": query,
                "max_results": settings.tool_web_search_top_k,
                "search_depth": "basic",
                "include_answer": False,
            },
        )
        response.raise_for_status()
        data = response.json()
    return [
        {
            "title": str(item.get("title") or ""),
            "url": str(item.get("url") or ""),
            "snippet": str(item.get("content") or ""),
            "source": _domain(str(item.get("url") or "")),
        }
        for item in data.get("results", [])[: settings.tool_web_search_top_k]
    ]


def _search_serpapi(query: str) -> list[dict[str, str]]:
    with httpx.Client(timeout=15) as client:
        response = client.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google",
                "q": query,
                "api_key": settings.tool_web_search_api_key,
                "num": settings.tool_web_search_top_k,
            },
        )
        response.raise_for_status()
        data = response.json()
    return [
        {
            "title": str(item.get("title") or ""),
            "url": str(item.get("link") or ""),
            "snippet": str(item.get("snippet") or ""),
            "source": _domain(str(item.get("link") or "")),
        }
        for item in data.get("organic_results", [])[: settings.tool_web_search_top_k]
    ]


def _search_bing(query: str) -> list[dict[str, str]]:
    with httpx.Client(timeout=15) as client:
        response = client.get(
            "https://api.bing.microsoft.com/v7.0/search",
            headers={"Ocp-Apim-Subscription-Key": settings.tool_web_search_api_key},
            params={"q": query, "count": settings.tool_web_search_top_k},
        )
        response.raise_for_status()
        data = response.json()
    values = data.get("webPages", {}).get("value", [])
    return [
        {
            "title": str(item.get("name") or ""),
            "url": str(item.get("url") or ""),
            "snippet": str(item.get("snippet") or ""),
            "source": _domain(str(item.get("url") or "")),
        }
        for item in values[: settings.tool_web_search_top_k]
    ]


def _format_results(results: list[dict[str, str]]) -> str:
    if not results:
        return "网页搜索没有返回结果。"
    lines = []
    for index, item in enumerate(results, start=1):
        lines.append(f"{index}. {item['title']} - {item['snippet']} ({item['url']})")
    return "网页搜索结果：\n" + "\n".join(lines)


def _domain(url: str) -> str:
    return urlparse(url).netloc


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed._replace(fragment="", query="").geturl().rstrip("/")


def _dedupe_results(results: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    for item in results:
        url_key = _normalize_url(item.get("url", ""))
        title_key = item.get("title", "").strip().lower()
        if url_key and url_key in seen_urls:
            continue
        if title_key and title_key in seen_titles:
            continue
        if url_key:
            seen_urls.add(url_key)
        if title_key:
            seen_titles.add(title_key)
        deduped.append(item)
    return deduped


def _trim_results(results: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            **item,
            "title": _trim_text(item.get("title", ""), 120),
            "snippet": _trim_text(item.get("snippet", ""), settings.tool_web_search_snippet_max_chars),
        }
        for item in results
    ]


def _trim_text(text: str, max_chars: int) -> str:
    collapsed = " ".join(str(text or "").split())
    if max_chars <= 0 or len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 1].rstrip() + "…"


def _get_cached(cache_key: str) -> dict[str, object] | None:
    ttl = settings.tool_web_search_cache_ttl_seconds
    if ttl <= 0:
        return None
    cached = _SEARCH_CACHE.get(cache_key)
    if cached is None:
        return None
    expires_at, payload = cached
    if expires_at < time():
        _SEARCH_CACHE.pop(cache_key, None)
        return None
    return payload


def _set_cached(cache_key: str, payload: dict[str, object]) -> None:
    ttl = settings.tool_web_search_cache_ttl_seconds
    if ttl <= 0:
        return
    _SEARCH_CACHE[cache_key] = (time() + ttl, payload)
