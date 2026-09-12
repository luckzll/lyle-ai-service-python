from datetime import datetime

from app.llm import prompts


def test_clock_question_and_memory_changes_preserve_the_knowledge_prefix(monkeypatch):
    now = [datetime(2026, 9, 12, 20, 0, 0)]

    class Clock:
        @staticmethod
        def now(zone):
            return now[0].replace(tzinfo=zone)

    monkeypatch.setattr(prompts, "datetime", Clock)
    contexts = [{"title": "版本说明", "chunk": "2.4.1 只修复断流恢复。"}]
    first = prompts.build_user_prompt("具体改了什么？", contexts, "近期对话：正在查看2.4.1")
    now[0] = datetime(2026, 9, 13, 21, 1, 2)
    second = prompts.build_user_prompt("这个版本的恢复行为呢？", contexts, "近期对话：继续讨论恢复")

    prefix = first.split("系统运行上下文：", 1)[0]
    assert prefix == second.split("系统运行上下文：", 1)[0]
    assert "来源：版本说明" in prefix and "2.4.1 只修复断流恢复。" in prefix
    assert "近期对话" not in prefix and "2026-09-12" not in prefix
    assert "2026-09-12 20:00:00" in first and "2026-09-13 21:01:02" in second
    assert first != second


def test_reordering_keeps_source_tool_memory_and_image_safety_instructions():
    prompt = prompts.build_user_prompt("生成图片", [{"title": "资料", "chunk": "可用事实"}], "已完成工具结果")
    for required in [
        "如果资料不足，请明确说明不确定", "不要输出参考来源列表", "系统会在回答末尾自动追加来源",
        "候选必须由用户确认后才生效", "active 画像是已确认记忆", "rejected 操作是已拒绝且未执行",
        "不得把不同对象统一描述为待确认", "请直接基于工具结果回答用户问题", "不要声称自己还没有调用工具",
        "不要向用户解释内部工具的输入格式", "不要输出图片 Markdown 或伪造图片链接",
    ]:
        assert required in prompt
    assert prompt.index("来源：资料") < prompt.index("当前北京时间") < prompt.index("已完成工具结果")
    assert prompt.endswith("用户问题：生成图片")


def test_without_knowledge_context_dynamic_context_and_safety_are_still_present():
    prompt = prompts.build_user_prompt("现在比分是多少？", [], "web_search成功：2:1")
    assert "系统运行上下文" in prompt
    assert "直接基于工具结果回答" in prompt
    assert "web_search成功：2:1" in prompt
    assert prompt.endswith("用户问题：现在比分是多少？")
