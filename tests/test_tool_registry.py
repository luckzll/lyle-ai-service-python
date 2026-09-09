import threading
import time

from app.core.config import settings
from app.api.chat import _image_generate_answer, _stream_image_tool, _strip_unbacked_image_markdown
from app.agents.tool_agent import decide_tool_use
from app.tools import image_tool
from app.tools.image_intent import is_image_generation_request
from app.tools.tool_registry import execute_tool, list_available_tools, select_tools
from app.tools import weather_tool
from app.tools.web_search_tool import search_web


def test_select_tools_detects_memory_stats_and_web_search():
    assert "memory_stats" in select_tools("帮我看一下记忆统计")
    assert "work_summary" in select_tools("我现在有几个要完成的任务有几个项目目标给我做一下汇总")
    assert "work_summary" in select_tools("看我所有项目任务操作计划目标")
    assert "web_search" in select_tools("联网搜索一下最新消息")
    assert "web_search" in select_tools("你去网上搜一下他们6.0版本的争议")
    assert "web_search" in select_tools("他们6.0版本的争议是什么")
    assert "web_search" in select_tools("这个游戏最近有什么官方公告")
    assert "weather" in select_tools("今天上海天气怎么样")
    assert "web_search" not in select_tools("今天上海天气怎么样")
    assert "file_parse" in select_tools("解析文件 file:///tmp/a.txt")
    assert "image_analyze" in select_tools("帮我看图 https://example.com/a.png 里面有什么")
    assert "image_analyze" in select_tools("请确认收到验收图片\n图片1：/tmp/v62-acceptance.png")
    assert "image_generate" in select_tools("帮我生成一张赛博朋克猫")
    assert "image_edit" in select_tools("把这张图背景换成雪山 图片：https://example.com/a.png")
    assert "image_edit" in select_tools(
        "参考这两张图生成 4 张海报 图片1：https://example.com/a.png 图片2：https://example.com/b.png"
    )
    assert "image_generate" in select_tools("再来个小狗狗，相比于小猫我跟喜欢小狗")
    assert "image_generate" in select_tools("换成一只小狗")
    assert "todo" in select_tools("创建待办：写测试")
    assert "calendar" in select_tools("创建日历：2026-07-01 10:00 开会")


def test_select_tools_does_not_web_search_for_plain_chat():
    assert "web_search" not in select_tools("随便聊聊")
    assert "web_search" not in select_tools("你记得我喜欢什么吗")
    assert "image_generate" not in select_tools("把这段话改成英文")


def test_image_generation_concept_discussion_never_starts_billable_tool():
    questions = [
        "请用一句话说明批量生图的价值",
        "批量生图有哪些应用",
        "如何使用文生图功能",
        "生图模型是什么意思",
        "你支持批量生图吗",
        "生成 4 张图片的成本是多少？",
    ]

    for question in questions:
        assert is_image_generation_request(question) is False
        assert "image_generate" not in select_tools(question)
        assert _stream_image_tool(question) is None

    assert image_tool.parse_image_count("请用一句话说明批量生图的价值") == 1


def test_explicit_image_generation_commands_still_route_to_image_tool():
    questions = [
        "帮我生成一张赛博朋克猫",
        "生成 4 张不同构图的猫",
        "请批量生图：猫",
        "生图：猫",
        "画一只猫",
        "再来个小狗狗，相比于小猫我更喜欢小狗",
        "换成一只小狗",
        "请生成一张关于批量生图价值的海报",
    ]

    for question in questions:
        assert is_image_generation_request(question) is True
        assert "image_generate" in select_tools(question)
        assert _stream_image_tool(question) == "image_generate"

    assert image_tool.parse_image_count("生成 4 张不同构图的猫") == 4
    assert image_tool.parse_image_count("请批量生图：猫") == 4


def test_llm_router_cannot_override_image_generation_billing_guard(monkeypatch):
    monkeypatch.setattr(settings, "tool_agent_enabled", True)
    monkeypatch.setattr(settings, "tool_router_llm_enabled", True)
    monkeypatch.setattr(settings, "tool_router_confidence_threshold", 0.65)
    monkeypatch.setattr("app.agents.tool_agent._route_tools_with_context", lambda *args: {
        "selected_tools": ["image_generate"],
        "tool_inputs": {"image_generate": "生成 4 张批量生图价值海报"},
        "reason": "误判为文生图",
        "confidence": 0.99,
        "needs_clarification": False,
        "router": "llm",
    })

    decision = decide_tool_use("u1", "c1", "请用一句话说明批量生图的价值", True)

    assert decision["should_call_tools"] is False
    assert decision["selected_tools"] == []


def test_image_edit_excludes_image_generate(monkeypatch):
    monkeypatch.setattr(settings, "tool_agent_enabled", True)
    monkeypatch.setattr(settings, "tool_router_llm_enabled", True)
    monkeypatch.setattr(settings, "tool_router_confidence_threshold", 0.65)
    monkeypatch.setattr("app.agents.tool_agent._route_tools_with_context", lambda *args: {
        "selected_tools": ["image_generate"],
        "tool_inputs": {},
        "reason": "误判为文生图",
        "confidence": 0.9,
        "needs_clarification": False,
        "router": "llm",
    })

    decision = decide_tool_use("u1", "c1", "把这张图背景换成雪山 图片：https://example.com/a.png", True)

    assert decision["should_call_tools"] is True
    assert decision["selected_tools"] == ["image_edit"]


def test_image_followup_edit_with_recent_image_ref_prefers_edit():
    selected = select_tools("不喜欢戴帽子，可以修改一下图片吗\n图片：https://example.com/source.png")

    assert selected == ["image_edit"]


def test_plain_dislike_with_image_ref_does_not_trigger_image_edit():
    selected = select_tools("我不喜欢帽子\n图片：https://example.com/source.png")

    assert "image_edit" not in selected
    assert "image_analyze" in selected


def test_execute_tool_respects_allowed_tool_names(monkeypatch):
    monkeypatch.setattr(settings, "tool_audit_enabled", False)
    monkeypatch.setattr(settings, "tool_allowed_names", "health_check")

    result = execute_tool("memory_snapshot", "u1", "c1", "查看记忆")

    assert result["ok"] is False
    assert "白名单" in result["content"]
    assert result["operationType"] == "memory.snapshot"
    assert result["riskLevel"] == "read"
    assert result["retryable"] is False
    assert result["userMessage"] == result["content"]
    assert result["error"]["code"] == "PERMISSION_DENIED"


def test_memory_stats_reports_postgres_disabled(monkeypatch):
    monkeypatch.setattr(settings, "tool_audit_enabled", False)
    monkeypatch.setattr(settings, "tool_allowed_names", "memory_stats")
    monkeypatch.setattr(settings, "tool_database_query_enabled", True)
    monkeypatch.setattr(settings, "postgres_enabled", False)

    result = execute_tool("memory_stats", "u1", "c1", "记忆统计")

    assert result["ok"] is False
    assert "PostgreSQL 未启用" in result["content"]
    assert result["operationType"] == "memory.stats"
    assert result["error"]["code"] == "DATABASE_ERROR"


def test_work_summary_reports_postgres_disabled(monkeypatch):
    monkeypatch.setattr(settings, "tool_audit_enabled", False)
    monkeypatch.setattr(settings, "tool_allowed_names", "work_summary")
    monkeypatch.setattr(settings, "tool_database_query_enabled", True)
    monkeypatch.setattr(settings, "postgres_enabled", False)

    result = execute_tool("work_summary", "u1", "c1", "任务汇总")

    assert result["ok"] is False
    assert "PostgreSQL 未启用" in result["content"]
    assert result["operationType"] == "work.summary"
    assert result["riskLevel"] == "read"
    assert result["error"]["code"] == "DATABASE_ERROR"


def test_work_summary_content_includes_detail_previews():
    content = execute_tool.__globals__["_format_work_summary"]({
        "todos": {
            "open": 2,
            "today": 1,
            "overdue": 0,
            "weekDue": 1,
            "weekDueItems": [{"title": "周报"}],
            "items": [{"title": "写测试"}],
        },
        "projects": {
            "active": 1,
            "current": [{"name": "V3"}],
            "items": [{"name": "V3"}],
            "rollup": [{"name": "V3", "activeGoals": 2, "activePlans": 1, "links": 3}],
        },
        "goals": {"active": 2, "averageProgress": 0.5, "items": [{"title": "完成验证"}]},
        "plans": {"active": 1, "running": 1, "awaitingConfirmation": 0, "items": [{"title": "三步计划"}]},
        "operations": {"pending": 1, "items": [{"summary": "创建待办"}]},
        "projectLinks": {"conversation": 3},
    })

    assert "待办列表预览：写测试" in content
    assert "项目列表预览：V3" in content
    assert "目标列表预览：完成验证" in content
    assert "计划列表预览：三步计划" in content
    assert "操作列表预览：创建待办" in content
    assert "按项目聚合：V3：目标2，计划1，关联3" in content


def test_web_search_requires_explicit_enablement(monkeypatch):
    monkeypatch.setattr(settings, "tool_audit_enabled", False)
    monkeypatch.setattr(settings, "tool_allowed_names", "web_search")
    monkeypatch.setattr(settings, "tool_web_search_enabled", False)

    result = execute_tool("web_search", "u1", "c1", "联网搜索")

    assert result["ok"] is False
    assert "未启用" in result["content"]


def test_list_available_tools_marks_enabled_state(monkeypatch):
    monkeypatch.setattr(settings, "tool_allowed_names", "health_check,list_tools,file_parse,todo,calendar")
    monkeypatch.setattr(settings, "tool_database_query_enabled", False)
    monkeypatch.setattr(settings, "tool_web_search_enabled", False)
    monkeypatch.setattr(settings, "tool_weather_enabled", True)
    monkeypatch.setattr(settings, "tool_file_parse_enabled", True)
    monkeypatch.setattr(settings, "tool_image_analyze_enabled", True)
    monkeypatch.setattr(settings, "tool_image_generate_enabled", True)
    monkeypatch.setattr(settings, "tool_image_edit_enabled", True)
    monkeypatch.setattr(settings, "tool_todo_enabled", True)
    monkeypatch.setattr(settings, "tool_calendar_enabled", True)

    tools = {item["name"]: item for item in list_available_tools()}

    assert tools["health_check"]["enabled"] is True
    assert tools["memory_stats"]["enabled"] is False
    assert tools["work_summary"]["enabled"] is False
    assert tools["web_search"]["enabled"] is False
    assert tools["weather"]["enabled"] is False
    assert tools["file_parse"]["enabled"] is True
    assert tools["image_analyze"]["enabled"] is False
    assert tools["image_generate"]["enabled"] is False
    assert tools["image_edit"]["enabled"] is False
    assert tools["todo"]["enabled"] is True
    assert tools["todo"]["riskLevel"] == "write"
    assert tools["todo"]["requiresConfirmation"] is True
    assert tools["calendar"]["enabled"] is True
    assert tools["calendar"]["riskLevel"] == "write"
    assert tools["calendar"]["requiresConfirmation"] is True
    assert tools["health_check"]["riskLevel"] == "read"
    assert tools["health_check"]["requiresConfirmation"] is False


def test_write_tool_is_blocked_when_it_does_not_return_confirmation(monkeypatch):
    monkeypatch.setattr(settings, "tool_audit_enabled", False)
    monkeypatch.setattr(settings, "tool_allowed_names", "todo")
    monkeypatch.setattr(settings, "tool_todo_enabled", True)
    monkeypatch.setattr("app.tools.tool_registry._todo", lambda *args: {
        "ok": True,
        "content": "错误地直接执行了写入。",
    })

    result = execute_tool("todo", "u1", "c1", "创建待办：写测试")

    assert result["ok"] is False
    assert result["policyBlocked"] is True
    assert result["requiresConfirmation"] is True
    assert result["operationType"] == "todo.preview"
    assert result["riskLevel"] == "write"
    assert result["error"]["code"] == "UNKNOWN_ERROR"
    assert "安全策略拦截" in result["content"]


def test_file_parse_tool_reads_local_text_file(monkeypatch, tmp_path):
    target = tmp_path / "note.txt"
    target.write_text("hello lyle", encoding="utf-8")
    monkeypatch.setattr(settings, "tool_audit_enabled", False)
    monkeypatch.setattr(settings, "tool_allowed_names", "file_parse")
    monkeypatch.setattr(settings, "tool_file_parse_enabled", True)
    monkeypatch.setattr(settings, "tool_file_parse_allowed_dirs", str(tmp_path))

    result = execute_tool("file_parse", "u1", "c1", f"解析文件 {target}")

    assert result["ok"] is True
    assert "hello lyle" in result["content"]
    assert result["operationType"] == "file.parse"
    assert result["riskLevel"] == "read"
    assert result["error"] is None
    assert result["result"]["file"]["title"] == "note.txt"


def test_execute_tool_normalises_confirmation_preview(monkeypatch):
    monkeypatch.setattr(settings, "tool_audit_enabled", False)
    monkeypatch.setattr(settings, "tool_allowed_names", "todo")
    monkeypatch.setattr(settings, "tool_todo_enabled", True)
    monkeypatch.setattr("app.tools.tool_registry._todo", lambda *args: {
        "ok": True,
        "content": "已生成待确认操作：创建待办。用户确认后才会执行。",
        "requiresConfirmation": True,
        "operation": {
            "id": 1,
            "operationType": "todo.create",
            "summary": "创建待办",
            "status": "pending",
        },
    })

    result = execute_tool("todo", "u1", "c1", "创建待办：写测试")

    assert result["ok"] is True
    assert result["operationType"] == "todo.create"
    assert result["requiresConfirmation"] is True
    assert result["riskLevel"] == "write"
    assert result["userMessage"].startswith("已生成待确认操作")
    assert result["result"]["operation"]["id"] == 1


def test_web_search_reports_missing_api_key(monkeypatch):
    monkeypatch.setattr(settings, "tool_web_search_provider", "tavily")
    monkeypatch.setattr(settings, "tool_web_search_api_key", "")

    result = search_web("OpenAI news")

    assert result["ok"] is False
    assert "API Key" in result["content"]


def test_weather_tool_formats_city_weather(monkeypatch):
    monkeypatch.setattr(weather_tool, "_WEATHER_CACHE", {})
    monkeypatch.setattr(weather_tool, "_geocode", lambda location: {
        "name": "上海",
        "country": "中国",
        "admin1": "上海市",
        "latitude": 31.23,
        "longitude": 121.47,
        "timezone": "Asia/Shanghai",
    })
    monkeypatch.setattr(weather_tool, "_fetch_weather", lambda latitude, longitude: {
        "current": {
            "weather_code": 1,
            "temperature_2m": 29,
            "apparent_temperature": 31,
            "relative_humidity_2m": 70,
            "precipitation": 0,
            "wind_speed_10m": 12,
        },
        "current_units": {
            "temperature_2m": "°C",
            "wind_speed_10m": "km/h",
        },
        "daily": {
            "time": ["2026-07-01"],
            "weather_code": [1],
            "temperature_2m_max": [33],
            "temperature_2m_min": [26],
            "precipitation_probability_max": [20],
        },
    })

    result = weather_tool.get_weather_from_question("今天上海天气怎么样")

    assert result["ok"] is True
    assert result["provider"] == "open-meteo"
    assert "中国，上海市，上海 当前天气" in result["content"]
    assert "29 °C" in result["content"]


def test_web_search_trims_long_snippets(monkeypatch):
    monkeypatch.setattr(settings, "tool_web_search_provider", "tavily")
    monkeypatch.setattr(settings, "tool_web_search_api_key", "fake-key")
    monkeypatch.setattr(settings, "tool_web_search_top_k", 1)
    monkeypatch.setattr(settings, "tool_web_search_snippet_max_chars", 12)
    monkeypatch.setattr("app.tools.web_search_tool._search_tavily", lambda query: [{
        "title": "A very long but valid title",
        "url": "https://example.com/a",
        "snippet": "这是一段非常非常长的搜索结果摘要",
        "source": "example.com",
    }])

    result = search_web("测试搜索裁剪")

    assert result["ok"] is True
    assert result["results"][0]["snippet"].endswith("…")
    assert len(result["results"][0]["snippet"]) == 12


def test_image_generate_tool_returns_image_markdown(monkeypatch):
    monkeypatch.setattr(settings, "ark_api_key", "fake-key")
    monkeypatch.setattr(settings, "image_model_name", "doubao-seedream-test")
    monkeypatch.setattr(settings, "generated_image_storage", "local")
    calls = []

    def fake_post_json(path, payload, timeout=None):
        calls.append(payload)
        return {"data": [{"url": "https://example.com/generated.png"}]}

    monkeypatch.setattr(image_tool, "_post_json", fake_post_json)

    result = image_tool.generate_image_from_question("帮我生成一张赛博朋克猫")

    assert result["ok"] is True
    assert result["prompt"] == "赛博朋克猫"
    assert calls[0]["watermark"] is False
    assert "![生成图片](https://example.com/generated.png)" in result["content"]


def test_image_generate_batch_runs_model_requests_concurrently(monkeypatch):
    monkeypatch.setattr(settings, "ark_api_key", "fake-key")
    monkeypatch.setattr(settings, "image_model_name", "doubao-seedream-test")
    monkeypatch.setattr(settings, "generated_image_storage", "local")
    monkeypatch.setattr(settings, "image_batch_max_count", 8)
    monkeypatch.setattr(settings, "image_batch_concurrency", 8)
    lock = threading.Lock()
    active = 0
    max_active = 0
    sequence = 0

    def fake_post_json(path, payload, timeout=None):
        nonlocal active, max_active, sequence
        with lock:
            active += 1
            max_active = max(max_active, active)
            sequence += 1
            current = sequence
        time.sleep(0.03)
        with lock:
            active -= 1
        return {"data": [{"url": f"https://example.com/generated-{current}.png"}]}

    monkeypatch.setattr(image_tool, "_post_json", fake_post_json)

    result = image_tool.generate_image_from_question("生成 4 张不同构图的赛博朋克猫")

    assert result["ok"] is True
    assert result["requestedCount"] == 4
    assert result["completedCount"] == 4
    assert len(result["imageUrls"]) == 4
    assert result["prompt"] == "不同构图的赛博朋克猫"
    assert max_active > 1
    assert result["batchStatus"] == "COMPLETED"


def test_image_generate_retries_provider_rate_limit(monkeypatch):
    monkeypatch.setattr(settings, "ark_api_key", "fake-key")
    monkeypatch.setattr(settings, "generated_image_storage", "local")
    monkeypatch.setattr(settings, "image_batch_retry_attempts", 2)
    monkeypatch.setattr(settings, "image_batch_retry_base_seconds", 0)
    calls = 0

    def fake_post_json(path, payload, timeout=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("图片模型请求失败：HTTP 429 rate limited")
        return {"data": [{"url": "https://example.com/retried.png"}]}

    monkeypatch.setattr(image_tool, "_post_json", fake_post_json)

    result = image_tool.generate_image_from_question("生成一张重试测试图")

    assert calls == 2
    assert result["batchStatus"] == "COMPLETED"
    assert result["imageUrl"] == "https://example.com/retried.png"


def test_image_edit_passes_all_reference_images_to_every_variation(monkeypatch):
    monkeypatch.setattr(settings, "ark_api_key", "fake-key")
    monkeypatch.setattr(settings, "image_model_name", "doubao-seedream-test")
    monkeypatch.setattr(settings, "generated_image_storage", "local")
    calls = []

    def fake_post_image_edit_json(payload, image_urls):
        calls.append(image_urls)
        return {"data": [{"url": f"https://example.com/edited-{len(calls)}.png"}]}

    monkeypatch.setattr(image_tool, "_post_image_edit_json", fake_post_image_edit_json)
    result = image_tool.edit_image_from_question(
        "参考这些图片生成 3 张海报 "
        "图片1：https://example.com/a.png 图片2：https://example.com/b.png"
    )

    assert result["ok"] is True
    assert result["completedCount"] == 3
    assert result["sourceImageRefs"] == [
        "https://example.com/a.png",
        "https://example.com/b.png",
    ]
    assert calls == [
        ["https://example.com/a.png", "https://example.com/b.png"],
        ["https://example.com/a.png", "https://example.com/b.png"],
        ["https://example.com/a.png", "https://example.com/b.png"],
    ]


def test_image_count_prefers_generated_output_count_over_reference_count(monkeypatch):
    monkeypatch.setattr(settings, "image_batch_max_count", 8)

    spec = image_tool.build_image_batch_spec(
        "参考这两张图的构图，生成四张不同风格的海报 "
        "图片1：https://example.com/a.png 图片2：https://example.com/b.png",
        "edit",
    )

    assert spec["count"] == 4
    assert spec["referenceImageRefs"] == [
        "https://example.com/a.png",
        "https://example.com/b.png",
    ]


def test_image_analyze_sends_all_images_in_one_vision_request(monkeypatch):
    monkeypatch.setattr(settings, "ark_api_key", "fake-key")
    captured = {}

    def fake_post_json(path, payload, timeout=None):
        captured.update(payload)
        return {"choices": [{"message": {"content": "两张图的共同点"}}]}

    monkeypatch.setattr(image_tool, "_post_json", fake_post_json)
    result = image_tool.analyze_image_from_question(
        "对比这两张图 图片1：https://example.com/a.png 图片2：https://example.com/b.png"
    )

    image_parts = [
        item for item in captured["messages"][0]["content"]
        if item["type"] == "image_url"
    ]
    assert result["ok"] is True
    assert result["imageRefs"] == ["https://example.com/a.png", "https://example.com/b.png"]
    assert [item["image_url"]["url"] for item in image_parts] == [
        "https://example.com/a.png",
        "https://example.com/b.png",
    ]


def test_image_generate_uploads_to_tos(monkeypatch):
    monkeypatch.setattr(settings, "ark_api_key", "fake-key")
    monkeypatch.setattr(settings, "image_model_name", "doubao-seedream-test")
    monkeypatch.setattr(settings, "generated_image_storage", "tos")
    monkeypatch.setattr(settings, "tos_access_key_id", "fake-ak")
    monkeypatch.setattr(settings, "tos_secret_access_key", "fake-sk")
    monkeypatch.setattr(settings, "tos_endpoint", "tos-cn-shanghai.ivolces.com")
    monkeypatch.setattr(settings, "tos_region", "cn-shanghai")
    monkeypatch.setattr(settings, "tos_bucket", "common-prod-object")
    monkeypatch.setattr(settings, "tos_save_path_head", "zl-dev/")
    monkeypatch.setattr(settings, "tos_public_url_head", "https://common-prod-object.tos-cn-shanghai.volces.com/")
    monkeypatch.setattr(
        image_tool,
        "_post_json",
        lambda path, payload, timeout=None: {"data": [{"b64_json": "/9j/4AAQSkZJRgABAQAAAQABAAD/2w=="}]},
    )

    uploaded = []

    def fake_sdk_upload(image_bytes, object_key):
        uploaded.append((image_bytes, object_key))

    monkeypatch.setattr(image_tool, "_upload_generated_image_to_tos_sdk", fake_sdk_upload)

    result = image_tool.generate_image_from_question("帮我生成一张赛博朋克猫")

    assert result["ok"] is True
    assert result["imageUrl"].startswith("https://common-prod-object.tos-cn-shanghai.volces.com/zl-dev/")
    assert uploaded
    assert uploaded[0][1].startswith("zl-dev/")


def test_image_generate_falls_back_to_tosutil(monkeypatch):
    import sys
    monkeypatch.setattr(settings, "ark_api_key", "fake-key")
    monkeypatch.setattr(settings, "image_model_name", "doubao-seedream-test")
    monkeypatch.setattr(settings, "generated_image_storage", "tos")
    monkeypatch.setattr(settings, "tos_access_key_id", "")
    monkeypatch.setattr(settings, "tos_secret_access_key", "")
    monkeypatch.setattr(settings, "tos_endpoint", "")
    monkeypatch.setattr(settings, "tosutil_path", sys.executable)
    monkeypatch.setattr(settings, "tos_bucket", "common-prod-object")
    monkeypatch.setattr(settings, "tos_save_path_head", "zl-dev/")
    monkeypatch.setattr(settings, "tos_public_url_head", "https://common-prod-object.tos-cn-shanghai.volces.com/")
    monkeypatch.setattr(
        image_tool,
        "_post_json",
        lambda path, payload, timeout=None: {"data": [{"b64_json": "/9j/4AAQSkZJRgABAQAAAQABAAD/2w=="}]},
    )

    commands = []

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):
        commands.append(command)
        return Completed()

    monkeypatch.setattr(image_tool.subprocess, "run", fake_run)

    result = image_tool.generate_image_from_question("帮我生成一张赛博朋克猫")

    assert result["ok"] is True
    assert result["imageUrl"].startswith("https://common-prod-object.tos-cn-shanghai.volces.com/zl-dev/")
    assert commands[0][0] == sys.executable
    assert commands[0][1] == "cp"
    assert commands[0][3].startswith("tos://common-prod-object/zl-dev/")


def test_image_edit_tool_returns_image_markdown(monkeypatch):
    monkeypatch.setattr(settings, "ark_api_key", "fake-key")
    monkeypatch.setattr(settings, "image_model_name", "doubao-seedream-test")
    monkeypatch.setattr(settings, "generated_image_storage", "local")
    calls = []

    def fake_post_image_edit_json(payload, image_urls):
        calls.append((payload, image_urls))
        return {"data": [{"url": "https://example.com/edited.png"}]}

    monkeypatch.setattr(image_tool, "_post_image_edit_json", fake_post_image_edit_json)

    result = image_tool.edit_image_from_question("把背景换成雪山 图片：https://example.com/source.png")

    assert result["ok"] is True
    assert result["prompt"] == "把背景换成雪山"
    assert result["sourceImageRef"] == "https://example.com/source.png"
    assert "![修改后图片](https://example.com/edited.png)" in result["content"]
    assert calls[0][0]["watermark"] is False
    assert calls[0][1] == ["https://example.com/source.png"]


def test_image_analyze_requires_image_ref(monkeypatch):
    monkeypatch.setattr(settings, "ark_api_key", "fake-key")

    result = image_tool.analyze_image_from_question("帮我看看这张图")

    assert result["ok"] is False
    assert "上传图片" in result["content"]
    assert "http" not in result["content"].lower()
    assert "base64" not in result["content"].lower()


def test_image_generate_answer_uses_tool_artifact():
    answer = _image_generate_answer("给我生成一张风景海报", [{
        "tool": "image_generate",
        "ok": True,
        "content": "图片已生成。\n\n![生成图片](https://example.com/image.jpg)",
    }])

    assert answer == "图片已生成。\n\n![生成图片](https://example.com/image.jpg)"


def test_image_generate_answer_blocks_fake_image_when_tool_missing():
    answer = _image_generate_answer("给我生成一张风景海报", [])

    assert answer is not None
    assert "不会伪造图片链接" in answer


def test_strip_unbacked_image_markdown_without_image_tool():
    answer = _strip_unbacked_image_markdown(
        "安排好了\n\n![假图](http://127.0.0.1:8000/generated-images/fake.jpg)",
        [],
    )

    assert "generated-images" not in answer
    assert "安排好了" in answer
