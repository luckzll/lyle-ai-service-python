from app.llm.prompts import build_user_prompt
import pytest


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


@pytest.mark.parametrize("contexts,memory", [([], ""), ([], "工具结果"), ([{"title": "资料", "chunk": "事实"}], "")])
def test_final_answer_separation_applies_without_requiring_tool_or_memory_context(contexts, memory):
    prompt = build_user_prompt("问题", contexts, memory)
    assert "最终正文不要重复执行流水、计划、自检或内部规则" in prompt
    assert "保留回答所需的证据引用、不确定性、失败说明、澄清问题和操作确认" in prompt
    assert "没有真实执行结果时不得声称操作成功" in prompt
