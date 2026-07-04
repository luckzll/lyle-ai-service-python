from app.api.chat import _v3_confirmation_guidance, _v3_preview_answer


def test_v3_confirmation_guidance_mentions_pending_goal_plan_and_operation():
    guidance = _v3_confirmation_guidance(
        [{"title": "完成测试报告"}],
        [{"title": "三步推进计划"}],
        [{
            "requiresConfirmation": True,
            "operation": {"status": "pending", "operationType": "todo.create"},
        }],
    )

    assert "待确认目标卡" in guidance
    assert "待确认计划卡" in guidance
    assert "待确认操作卡" in guidance
    assert "禁止使用" in guidance


def test_v3_preview_answer_is_deterministic_for_pending_items():
    answer = _v3_preview_answer(
        [{"title": "完成测试报告", "horizon": "week"}],
        [{"title": "三步推进计划", "steps": [{}, {}, {}]}],
        [{
            "requiresConfirmation": True,
            "operation": {"status": "pending", "summary": "创建待办「检查报告」"},
        }],
        {"resolver": {"projectName": "V3测试项目"}},
    )

    assert "已生成待确认目标（V3测试项目）：完成测试报告" in answer
    assert "已生成待确认计划（V3测试项目）：三步推进计划，共 3 步" in answer
    assert "已生成待确认操作：创建待办「检查报告」" in answer
    assert "确认前不会创建或执行" in answer
    assert "好的，我已经" not in answer
    assert "已写入" not in answer
