from app.plans.plan_builder import build_plan_candidate
from app.projects.candidate_extractor import extract_project_candidates


def test_goal_candidate_extracts_clean_title_from_project_command():
    result = extract_project_candidates(
        "在 V3测试项目 下新增一个目标：完成V3测试项目的全面验证并输出测试报告，周期本周。",
        {"resolver": {"projectId": 9, "requiresClarification": False}},
    )

    assert result["candidateGoals"][0]["title"] == "完成V3测试项目的全面验证并输出测试报告"
    assert result["candidateGoals"][0]["horizon"] == "week"


def test_plan_candidate_detects_three_step_project_plan_request():
    candidate = build_plan_candidate(
        "帮我基于这个目标拆一个三步推进计划。",
        {"resolver": {"projectId": 9, "requiresClarification": False}},
    )

    assert candidate is not None
    assert candidate["projectId"] == 9
    assert candidate["requiresConfirmation"] is True
    assert len(candidate["steps"]) >= 3


def test_plan_candidate_uses_quoted_goal_as_title():
    candidate = build_plan_candidate(
        "基于 V3测试项目 的目标「完成测试报告」拆一个三步推进计划。",
        {"resolver": {"projectId": 9, "requiresClarification": False}},
    )

    assert candidate is not None
    assert candidate["title"] == "完成测试报告推进计划"


def test_plan_request_does_not_create_extra_goal_candidate():
    result = extract_project_candidates(
        "基于 V3测试项目 的目标「完成测试报告」拆一个三步推进计划。",
        {"resolver": {"projectId": 9, "requiresClarification": False}},
    )

    assert result["candidateGoals"] == []
