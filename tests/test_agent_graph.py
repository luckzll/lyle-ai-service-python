from app.chains.intent_chain import classify_intent_by_rule
from app.core.config import settings


def test_classify_memory_intent():
    assert classify_intent_by_rule("你还记得我喜欢什么吗", []) == "memory"


def test_classify_knowledge_intent_when_knowledge_base_selected():
    assert classify_intent_by_rule("这份文档讲了什么", ["kb1"]) == "knowledge"


def test_multi_agent_graph_runs_agents(monkeypatch):
    from app.graphs.multi_agent_graph import run_multi_agent_graph

    monkeypatch.setattr(
        "app.graphs.multi_agent_graph.run_supervisor_agent",
        lambda question, knowledge_base_ids: {
            "intent": "knowledge",
            "use_memory": True,
            "use_rag": True,
            "use_tools": False,
            "next_agents": ["memory_agent", "rag_agent", "writer_agent"],
        },
    )
    monkeypatch.setattr(
        "app.graphs.multi_agent_graph.run_memory_agent",
        lambda user_id, conversation_id, question: {
            "memory": {"summary": "用户在做 AI 应用"},
            "memory_context": "长期摘要：用户在做 AI 应用",
        },
    )
    monkeypatch.setattr(
        "app.graphs.multi_agent_graph.run_rag_agent",
        lambda question, knowledge_base_ids, use_rag: {
            "contexts": [{"doc_id": "d1", "title": "方案", "chunk": "多 Agent 方案"}],
            "queries": [question, "多 Agent 方案"],
            "sufficiency": {"sufficient": True, "score": 0.9, "reason": "ok"},
            "skipped": False,
        },
    )
    monkeypatch.setattr(
        "app.graphs.multi_agent_graph.run_writer_agent",
        lambda question, contexts, memory_context, tool_context="", review_feedback="": f"answer: {memory_context} / {contexts[0]['title']}",
    )
    monkeypatch.setattr(
        "app.graphs.multi_agent_graph.run_reviewer_agent",
        lambda question, answer, contexts, memory_context: {
            "approved": True,
            "score": 0.9,
            "reason": "ok",
            "rewrite_instruction": "",
            "skipped": False,
        },
    )

    result = run_multi_agent_graph("u1", "c1", "继续方案", ["kb1"])

    assert result["intent"] == "knowledge"
    assert result["answer"] == "answer: 长期摘要：用户在做 AI 应用 / 方案"
    assert [item["agent"] for item in result["agent_trace"]] == [
        "supervisor_agent",
        "memory_agent",
        "rag_agent",
        "gather_agent",
        "writer_agent",
        "reviewer_agent",
    ]
    assert result["agent_trace"][3]["receivedAgents"] == ["memory_agent", "rag_agent"]
    assert result["agent_trace"][1]["memorySummary"]["summaryChars"] == len("用户在做 AI 应用")
    assert result["agent_trace"][1]["memorySummary"]["summaryPreview"] == "用户在做 AI 应用"


def test_multi_agent_graph_includes_supplemental_context_in_writer(monkeypatch):
    from app.graphs.multi_agent_graph import run_multi_agent_graph

    monkeypatch.setattr(
        "app.graphs.multi_agent_graph.run_supervisor_agent",
        lambda question, knowledge_base_ids: {
            "intent": "chat",
            "use_memory": True,
            "use_rag": False,
            "use_tools": False,
            "next_agents": ["memory_agent", "writer_agent"],
        },
    )
    monkeypatch.setattr(
        "app.graphs.multi_agent_graph.run_memory_agent",
        lambda user_id, conversation_id, question: {"memory": {}, "memory_context": "长期记忆"},
    )

    captured = {}

    def fake_writer(question, contexts, memory_context, tool_context="", review_feedback=""):
        captured["tool_context"] = tool_context
        return "已使用项目上下文"

    monkeypatch.setattr("app.graphs.multi_agent_graph.run_writer_agent", fake_writer)
    def fake_reviewer(question, answer, contexts, memory_context, **kwargs):
        captured["review_context"] = memory_context
        return {"approved": True, "score": 1.0, "reason": "ok", "skipped": True}

    monkeypatch.setattr("app.graphs.multi_agent_graph.run_reviewer_agent", fake_reviewer)

    result = run_multi_agent_graph(
        "u1",
        "c1",
        "下一步是什么？",
        [],
        "[项目上下文]\n项目：知识库体验优化",
    )

    assert result["answer"] == "已使用项目上下文"
    assert "项目：知识库体验优化" in captured["tool_context"]
    assert "项目：知识库体验优化" in captured["review_context"]
    assert "项目：知识库体验优化" in result["agent_trace"][-2]["writerInput"]["toolContext"]


def test_multi_agent_graph_rewrites_when_review_fails(monkeypatch):
    from app.graphs.multi_agent_graph import run_multi_agent_graph

    monkeypatch.setattr(settings, "reviewer_max_retry", 1)
    monkeypatch.setattr(
        "app.graphs.multi_agent_graph.run_supervisor_agent",
        lambda question, knowledge_base_ids: {
            "intent": "chat",
            "use_memory": True,
            "use_rag": False,
            "use_tools": False,
            "next_agents": ["memory_agent", "writer_agent"],
        },
    )
    monkeypatch.setattr(
        "app.graphs.multi_agent_graph.run_memory_agent",
        lambda user_id, conversation_id, question: {"memory": {}, "memory_context": ""},
    )
    monkeypatch.setattr(
        "app.graphs.multi_agent_graph.run_rag_agent",
        lambda question, knowledge_base_ids, use_rag: {
            "contexts": [],
            "queries": [],
            "sufficiency": {"sufficient": True},
            "skipped": True,
        },
    )

    answers = iter(["bad answer", "good answer"])

    monkeypatch.setattr(
        "app.graphs.multi_agent_graph.run_tool_agent",
        lambda user_id, conversation_id, question, use_tools, knowledge_base_ids=None: {
            "tool_results": [],
            "tool_context": "",
            "selected_tools": [],
            "skipped": True,
        },
    )

    def fake_writer(question, contexts, memory_context, tool_context="", review_feedback=""):
        if review_feedback:
            assert "请补充" in review_feedback
        return next(answers)

    reviews = iter([
        {
            "approved": False,
            "score": 0.3,
            "reason": "太短",
            "rewrite_instruction": "请补充关键细节",
            "skipped": False,
        },
        {
            "approved": True,
            "score": 0.9,
            "reason": "ok",
            "rewrite_instruction": "",
            "skipped": False,
        },
    ])

    monkeypatch.setattr("app.graphs.multi_agent_graph.run_writer_agent", fake_writer)
    monkeypatch.setattr(
        "app.graphs.multi_agent_graph.run_reviewer_agent",
        lambda question, answer, contexts, memory_context: next(reviews),
    )

    result = run_multi_agent_graph("u1", "c1", "解释一下", [])

    assert result["answer"] == "good answer"
    assert result["review"]["approved"] is True
    assert [item["agent"] for item in result["agent_trace"]].count("writer_agent") == 2
    assert [item["agent"] for item in result["agent_trace"]].count("reviewer_agent") == 2


def test_multi_agent_graph_runs_tool_agent(monkeypatch):
    from app.graphs.multi_agent_graph import run_multi_agent_graph

    monkeypatch.setattr(
        "app.graphs.multi_agent_graph.run_supervisor_agent",
        lambda question, knowledge_base_ids: {
            "intent": "chat",
            "use_memory": True,
            "use_rag": False,
            "use_tools": True,
            "next_agents": ["memory_agent", "tool_agent", "writer_agent"],
        },
    )
    monkeypatch.setattr(
        "app.graphs.multi_agent_graph.run_memory_agent",
        lambda user_id, conversation_id, question: {"memory": {}, "memory_context": ""},
    )
    monkeypatch.setattr(
        "app.graphs.multi_agent_graph.run_rag_agent",
        lambda question, knowledge_base_ids, use_rag: {
            "contexts": [],
            "queries": [],
            "sufficiency": {"sufficient": True},
            "skipped": True,
        },
    )
    monkeypatch.setattr(
        "app.graphs.multi_agent_graph.run_tool_agent",
        lambda user_id, conversation_id, question, use_tools, knowledge_base_ids=None: {
            "tool_results": [{"tool": "health_check", "ok": True, "content": "ok"}],
            "tool_context": "工具调用结果：health_check ok",
            "selected_tools": ["health_check"],
            "skipped": False,
        },
    )

    def fake_writer(question, contexts, memory_context, tool_context="", review_feedback=""):
        assert "health_check" in tool_context
        return "工具正常"

    monkeypatch.setattr("app.graphs.multi_agent_graph.run_writer_agent", fake_writer)
    monkeypatch.setattr(
        "app.graphs.multi_agent_graph.run_reviewer_agent",
        lambda question, answer, contexts, memory_context: {
            "approved": True,
            "score": 0.9,
            "reason": "ok",
            "rewrite_instruction": "",
            "skipped": False,
        },
    )

    result = run_multi_agent_graph("u1", "c1", "检查服务健康", [])

    assert result["answer"] == "工具正常"
    assert result["tool_results"][0]["tool"] == "health_check"
    assert any(item["agent"] == "tool_agent" and item["toolCount"] == 1 for item in result["agent_trace"])


def test_agent_self_decisions():
    from app.agents.memory_agent import decide_memory_recall
    from app.agents.rag_agent import decide_rag_retrieval
    from app.agents.tool_agent import decide_tool_use

    old_router_enabled = settings.tool_router_llm_enabled
    settings.tool_router_llm_enabled = False
    try:
        assert decide_memory_recall("你好")["should_recall"] is False
        assert decide_memory_recall("你记得我喜欢什么吗")["depth"] == "full"
        assert decide_rag_retrieval("这份文档讲了什么", ["kb1"], True)["should_retrieve"] is True
        assert decide_rag_retrieval("这份文档讲了什么", [], True)["should_retrieve"] is False
        assert decide_tool_use("u1", "c1", "联网搜索一下最新消息", True)["should_call_tools"] is True
        assert decide_tool_use("u1", "c1", "随便聊聊", True)["should_call_tools"] is False
    finally:
        settings.tool_router_llm_enabled = old_router_enabled


def test_tool_router_can_rewrite_followup_image_generation(monkeypatch):
    from app.agents.tool_agent import decide_tool_use

    monkeypatch.setattr(settings, "tool_router_llm_enabled", True)
    monkeypatch.setattr(
        "app.agents.tool_agent.recall_memory",
        lambda user_id, conversation_id, question, **kwargs: {
            "short_term": [
                {"role": "user", "content": "帮我生成一张赛博朋克猫"},
                {"role": "assistant", "content": "图片已生成。"},
            ]
        },
    )
    monkeypatch.setattr(
        "app.agents.tool_agent.route_tools_with_llm",
        lambda question, context: {
            "should_call_tools": True,
            "selected_tools": ["image_generate"],
            "tool_inputs": {"image_generate": "延续上一张赛博朋克猫风格，生成一只赛博朋克小狗"},
            "reason": "用户要求基于上一张图续作。",
            "confidence": 0.9,
            "needs_clarification": False,
            "router": "llm",
        },
    )

    decision = decide_tool_use("u1", "c1", "再来个小狗狗", True)

    assert decision["should_call_tools"] is True
    assert decision["selected_tools"] == ["image_generate"]
    assert decision["tool_inputs"]["image_generate"] == "延续上一张赛博朋克猫风格，生成一只赛博朋克小狗"


def test_explicit_todo_rule_skips_llm_tool_router(monkeypatch):
    from app.agents.tool_agent import decide_tool_use

    monkeypatch.setattr(settings, "tool_agent_enabled", True)
    monkeypatch.setattr(settings, "tool_router_llm_enabled", True)

    def fail_route(question: str, context: str = ""):
        raise AssertionError("explicit todo rule should not call LLM router")

    monkeypatch.setattr("app.agents.tool_agent.route_tools_with_llm", fail_route)

    decision = decide_tool_use("u1", "c1", "创建待办：整理知识库", True)

    assert decision["should_call_tools"] is True
    assert decision["selected_tools"] == ["todo"]
    assert decision["router"] == "rule"


def test_project_goal_write_skips_todo_tool_router(monkeypatch):
    from app.agents.tool_agent import decide_tool_use

    monkeypatch.setattr(settings, "tool_agent_enabled", True)
    monkeypatch.setattr(settings, "tool_router_llm_enabled", True)

    def fail_route(question: str, context: str = ""):
        raise AssertionError("project goal writes should not call LLM tool router")

    monkeypatch.setattr("app.agents.tool_agent.route_tools_with_llm", fail_route)

    decision = decide_tool_use("u1", "c1", "在 T2测试项目 下新增一个目标：完成测试报告，周期本周", True)

    assert decision["should_call_tools"] is False
    assert decision["router"] == "v3_project_guard"


def test_tool_agent_keeps_original_question_for_calendar(monkeypatch):
    from app.agents.tool_agent import run_tool_agent

    monkeypatch.setattr(settings, "tool_router_llm_enabled", True)
    monkeypatch.setattr("app.agents.tool_agent.recall_memory", lambda user_id, conversation_id, question, **kwargs: {})
    monkeypatch.setattr(
        "app.agents.tool_agent.route_tools_with_llm",
        lambda question, context: {
            "should_call_tools": True,
            "selected_tools": ["calendar"],
            "tool_inputs": {"calendar": '{"time":"tomorrow 15:00","title":"开会"}'},
            "reason": "用户要创建日程。",
            "confidence": 0.9,
            "needs_clarification": False,
            "router": "llm",
        },
    )
    received = {}

    def fake_execute(tool_name, user_id, conversation_id, question):
        received["tool_name"] = tool_name
        received["question"] = question
        return {"tool": tool_name, "ok": True, "content": "ok"}

    monkeypatch.setattr("app.agents.tool_agent.execute_tool", fake_execute)

    run_tool_agent("u1", "c1", "明天三点提醒我开会", True)

    assert received["tool_name"] == "calendar"
    assert received["question"] == "明天三点提醒我开会"


def test_tool_agent_skips_low_confidence_llm_route(monkeypatch):
    from app.agents.tool_agent import decide_tool_use

    monkeypatch.setattr(settings, "tool_router_llm_enabled", True)
    monkeypatch.setattr(settings, "tool_router_confidence_threshold", 0.65)
    monkeypatch.setattr("app.agents.tool_agent.recall_memory", lambda user_id, conversation_id, question, **kwargs: {})
    monkeypatch.setattr(
        "app.agents.tool_agent.route_tools_with_llm",
        lambda question, context: {
            "should_call_tools": True,
            "selected_tools": ["web_search"],
            "tool_inputs": {"web_search": "不明确的主题"},
            "reason": "可能需要搜索，但主题不明确。",
            "confidence": 0.42,
            "needs_clarification": False,
            "router": "llm",
        },
    )

    decision = decide_tool_use("u1", "c1", "他们那个呢", True)

    assert decision["should_call_tools"] is False
    assert decision["confidence"] == 0.42
    assert "置信度过低" in decision["reason"]


def test_tool_agent_skips_when_router_needs_clarification(monkeypatch):
    from app.agents.tool_agent import decide_tool_use

    monkeypatch.setattr(settings, "tool_router_llm_enabled", True)
    monkeypatch.setattr("app.agents.tool_agent.recall_memory", lambda user_id, conversation_id, question, **kwargs: {})
    monkeypatch.setattr(
        "app.agents.tool_agent.route_tools_with_llm",
        lambda question, context: {
            "should_call_tools": True,
            "selected_tools": ["calendar"],
            "tool_inputs": {},
            "reason": "缺少具体时间，需要先问清楚。",
            "confidence": 0.8,
            "needs_clarification": True,
            "router": "llm",
        },
    )

    decision = decide_tool_use("u1", "c1", "帮我安排一下", True)

    assert decision["should_call_tools"] is False
    assert decision["needs_clarification"] is True
    assert "缺少具体时间" in decision["reason"]


def test_supervisor_does_not_use_tool_agent_for_chat_just_because_router_is_enabled(monkeypatch):
    from app.agents.supervisor_agent import run_supervisor_agent

    monkeypatch.setattr(settings, "tool_agent_enabled", True)
    monkeypatch.setattr(settings, "tool_router_llm_enabled", True)
    monkeypatch.setattr("app.agents.supervisor_agent.classify_intent", lambda question, knowledge_base_ids: "chat")

    result = run_supervisor_agent("随便聊聊", [])

    assert result["use_tools"] is False
    assert result["use_memory"] is False


def test_prepare_context_uses_supervisor_for_ordinary_chat(monkeypatch):
    from app.graphs.multi_agent_graph import prepare_multi_agent_context

    monkeypatch.setattr('app.graphs.multi_agent_graph.load_conversation_context', lambda *_: {'memory': {}, 'memory_context': ''})
    called = {"supervisor": False}
    monkeypatch.setattr(
        "app.graphs.multi_agent_graph.run_supervisor_agent",
        lambda question, knowledge_base_ids: called.update({"supervisor": True}) or {
            "intent": "chat",
            "use_memory": True,
            "use_rag": False,
            "use_tools": False,
            "next_agents": ["memory_agent", "writer_agent"],
        },
    )
    monkeypatch.setattr(
        "app.graphs.multi_agent_graph.run_memory_agent",
        lambda user_id, conversation_id, question: {
            "memory": {},
            "memory_context": "",
            "decision": {"should_recall": False, "depth": "skip"},
            "skipped": True,
        },
    )

    result = prepare_multi_agent_context("u1", "c1", "你在干什么呢", [])

    assert called["supervisor"] is True
    assert result["intent"] == "chat"
    assert result["tool_results"] == []
    assert [item["agent"] for item in result["agent_trace"]] == [
        "supervisor_agent",
        "memory_agent",
        "gather_agent",
    ]


def test_prepare_context_uses_full_path_for_tool_signal(monkeypatch):
    from app.graphs.multi_agent_graph import prepare_multi_agent_context

    monkeypatch.setattr('app.graphs.multi_agent_graph.load_conversation_context', lambda *_: {'memory': {}, 'memory_context': ''})
    called = {"supervisor": False, "tool": False}

    monkeypatch.setattr(
        "app.graphs.multi_agent_graph.run_supervisor_agent",
        lambda question, knowledge_base_ids: called.update({"supervisor": True}) or {
            "intent": "chat",
            "use_memory": True,
            "use_rag": False,
            "use_tools": True,
            "next_agents": ["memory_agent", "tool_agent", "writer_agent"],
        },
    )
    monkeypatch.setattr(
        "app.graphs.multi_agent_graph.run_memory_agent",
        lambda user_id, conversation_id, question: {
            "memory": {},
            "memory_context": "",
            "decision": {"should_recall": False, "depth": "skip"},
            "skipped": True,
        },
    )
    monkeypatch.setattr(
        "app.graphs.multi_agent_graph.run_tool_agent",
        lambda user_id, conversation_id, question, use_tools, knowledge_base_ids=None: called.update({"tool": True}) or {
            "tool_results": [{"tool": "todo", "ok": True, "content": "ok"}],
            "tool_context": "todo ok",
            "selected_tools": ["todo"],
            "decision": {"should_call_tools": True},
            "skipped": False,
        },
    )

    result = prepare_multi_agent_context("u1", "c1", "创建待办：整理知识库", [])

    assert called == {"supervisor": True, "tool": True}
    assert result["tool_results"][0]["tool"] == "todo"
