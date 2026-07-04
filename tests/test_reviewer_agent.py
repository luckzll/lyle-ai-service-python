from app.agents.reviewer_agent import run_reviewer_agent
from app.core.config import settings


def test_reviewer_blocks_answer_that_mentions_forgotten_memory(monkeypatch):
    monkeypatch.setattr(settings, "reviewer_enabled", False)
    memory = {
        "governance": {
            "blockedTerms": [{
                "value": "帽子",
                "status": "forgotten",
                "schemaPath": "preferences.visual.dislikes",
            }],
            "lowConfidenceTerms": [],
        }
    }

    review = run_reviewer_agent("图片风格怎么做", "你不喜欢帽子，所以我会避开。", [], "", memory=memory)

    assert review["approved"] is False
    assert review["ruleBlocked"] is True
    assert "已治理记忆" in review["reason"]


def test_reviewer_blocks_uncautious_low_confidence_memory(monkeypatch):
    monkeypatch.setattr(settings, "reviewer_enabled", False)
    memory = {
        "governance": {
            "blockedTerms": [],
            "lowConfidenceTerms": [{
                "value": "西瓜",
                "status": "active",
                "schemaPath": "preferences.food.likes",
            }],
        }
    }

    review = run_reviewer_agent("我喜欢什么", "你喜欢西瓜。", [], "", memory=memory)

    assert review["approved"] is False
    assert "低置信度" in review["reason"]


def test_reviewer_allows_cautious_low_confidence_memory(monkeypatch):
    monkeypatch.setattr(settings, "reviewer_enabled", False)
    memory = {
        "governance": {
            "blockedTerms": [],
            "lowConfidenceTerms": [{
                "value": "西瓜",
                "status": "active",
                "schemaPath": "preferences.food.likes",
            }],
        }
    }

    review = run_reviewer_agent("我喜欢什么", "你之前提到过西瓜，但我不确定这是否仍然准确。", [], "", memory=memory)

    assert review["approved"] is True
    assert review["skipped"] is True


def test_reviewer_blocks_rag_answer_when_sufficiency_is_false_without_caveat(monkeypatch):
    monkeypatch.setattr(settings, "reviewer_enabled", False)

    review = run_reviewer_agent(
        "项目预算是多少？",
        "项目预算是 100 万元。",
        [{"title": "项目资料", "chunk": "资料只描述了项目目标，没有预算信息。"}],
        "",
        rag_sufficiency={"sufficient": False, "reason": "资料不足", "score": 0.2},
    )

    assert review["approved"] is False
    assert "资料不足" in review["reason"]


def test_reviewer_allows_rag_answer_that_admits_insufficient_evidence(monkeypatch):
    monkeypatch.setattr(settings, "reviewer_enabled", False)

    review = run_reviewer_agent(
        "项目预算是多少？",
        "当前资料不足，无法确认项目预算。",
        [{"title": "项目资料", "chunk": "资料只描述了项目目标，没有预算信息。"}],
        "",
        rag_sufficiency={"sufficient": False, "reason": "资料不足", "score": 0.2},
    )

    assert review["approved"] is True


def test_reviewer_blocks_document_grounding_with_low_context_overlap(monkeypatch):
    monkeypatch.setattr(settings, "reviewer_enabled", False)

    review = run_reviewer_agent(
        "文档讲了什么？",
        "根据资料，火星基地将在 2030 年开放，并提供商业航班。",
        [{"title": "海蓝星", "chunk": "海蓝星是一颗被海洋覆盖的虚构星球。"}],
        "",
        rag_sufficiency={"sufficient": True, "score": 0.9},
    )

    assert review["approved"] is False
    assert "覆盖过低" in review["reason"]


def test_reviewer_blocks_connector_failure_claimed_as_success(monkeypatch):
    monkeypatch.setattr(settings, "reviewer_enabled", False)

    review = run_reviewer_agent(
        "读取 GitHub issue",
        "已读取 GitHub，命中 3 条 issue。",
        [],
        "",
        tool_results=[{
            "connector": "github",
            "ok": False,
            "error": {"code": "AUTH_REQUIRED", "message": "GitHub token 未配置。"},
        }],
    )

    assert review["approved"] is False
    assert review["v3Rule"] == "connector_failure"


def test_reviewer_blocks_long_task_execution_before_plan_confirmation(monkeypatch):
    monkeypatch.setattr(settings, "reviewer_enabled", False)

    review = run_reviewer_agent(
        "帮我把 V3 方案整理成 PRD 并拆任务",
        "我已经完成并写入了任务。",
        [],
        "",
        candidate_plans=[{"title": "整理 V3 PRD", "requiresConfirmation": True}],
    )

    assert review["approved"] is False
    assert review["v3Rule"] == "plan_confirmation"


def test_reviewer_blocks_goal_recorded_claim_before_confirmation(monkeypatch):
    monkeypatch.setattr(settings, "reviewer_enabled", False)

    review = run_reviewer_agent(
        "在 V3测试项目 下新增一个目标：完成测试报告，周期本周。",
        "目标已经直接记录下来，挂到 V3测试项目 下。",
        [],
        "",
        project_context={"resolver": {"projectId": 9, "requiresClarification": False}},
        candidate_goals=[{"title": "完成测试报告", "projectId": 9, "status": "pending"}],
    )

    assert review["approved"] is False
    assert review["v3Rule"] == "goal_confirmation"


def test_reviewer_blocks_goal_recorded_claim_when_project_missing(monkeypatch):
    monkeypatch.setattr(settings, "reviewer_enabled", False)

    review = run_reviewer_agent(
        "在 T2测试项目 下新增一个目标：完成测试报告，周期本周。",
        "目标已经记录下来，并挂到 T2测试项目 下。",
        [],
        "",
        project_context={"resolver": {"projectId": None, "requiresClarification": False}},
    )

    assert review["approved"] is False
    assert review["v3Rule"] == "goal_project_missing"


def test_reviewer_blocks_profile_as_project_fact_source(monkeypatch):
    monkeypatch.setattr(settings, "reviewer_enabled", False)

    review = run_reviewer_agent(
        "Lyle 项目现在是什么阶段？",
        "根据用户画像，这是项目事实：项目已经进入商业化阶段。",
        [],
        "",
    )

    assert review["approved"] is False
    assert review["v3Rule"] == "project_fact_grounding"


def test_reviewer_blocks_project_link_when_resolver_needs_clarification(monkeypatch):
    monkeypatch.setattr(settings, "reviewer_enabled", False)

    review = run_reviewer_agent(
        "Lyle 和 Eden 都要继续推进",
        "已关联到项目 Lyle。",
        [],
        "",
        project_context={"resolver": {"requiresClarification": True}},
    )

    assert review["approved"] is False
    assert review["v3Rule"] == "project_clarification"
