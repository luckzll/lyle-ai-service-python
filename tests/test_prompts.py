from app.llm.prompts import build_user_prompt


def test_build_user_prompt_instructs_use_of_tool_results():
    prompt = build_user_prompt(
        "现在比分是多少？",
        [],
        "已完成的工具调用结果如下。\n- web_search(成功)：网页搜索结果：比分 2:1",
    )

    assert "已完成的工具调用结果" in prompt
    assert "直接基于工具结果回答用户问题" in prompt
    assert "系统运行上下文" in prompt
    assert "当前北京时间" in prompt
    assert "Asia/Shanghai" in prompt
    assert "UTC+08:00" in prompt
    assert "现在比分是多少？" in prompt
