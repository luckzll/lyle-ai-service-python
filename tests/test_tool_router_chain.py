from app.chains import tool_router_chain
from langchain_core.runnables import RunnableLambda


def test_tool_router_prompt_accepts_json_example(monkeypatch):
    monkeypatch.setattr(tool_router_chain, "is_intent_model_configured", lambda: True)
    monkeypatch.setattr(
        tool_router_chain,
        "build_intent_task_model",
        lambda: RunnableLambda(lambda messages: '{"should_call_tools":false,"selected_tools":[],"tool_inputs":{},"reason":"普通聊天"}'),
    )

    result = tool_router_chain.route_tools_with_llm("随便聊聊", "")

    assert result["should_call_tools"] is False
    assert result["selected_tools"] == []
    assert result["confidence"] == 0.75
    assert result["needs_clarification"] is False


def test_tool_router_parses_image_generation_input(monkeypatch):
    monkeypatch.setattr(tool_router_chain, "is_intent_model_configured", lambda: True)
    monkeypatch.setattr(
        tool_router_chain,
        "build_intent_task_model",
        lambda: RunnableLambda(
            lambda messages: (
                '{"should_call_tools":true,"selected_tools":["image_generate"],'
                '"tool_inputs":{"image_generate":"延续上一张赛博朋克猫风格，生成一只赛博朋克小狗"},'
                '"confidence":0.86,"needs_clarification":false,"reason":"用户要求续作"}'
            )
        ),
    )

    result = tool_router_chain.route_tools_with_llm("再来个小狗狗", "上一轮生成了赛博朋克猫")

    assert result["should_call_tools"] is True
    assert result["selected_tools"] == ["image_generate"]
    assert result["tool_inputs"]["image_generate"] == "延续上一张赛博朋克猫风格，生成一只赛博朋克小狗"
    assert result["confidence"] == 0.86
    assert result["needs_clarification"] is False


def test_tool_router_marks_clarification_as_no_call(monkeypatch):
    monkeypatch.setattr(tool_router_chain, "is_intent_model_configured", lambda: True)
    monkeypatch.setattr(
        tool_router_chain,
        "build_intent_task_model",
        lambda: RunnableLambda(
            lambda messages: (
                '{"should_call_tools":true,"selected_tools":["calendar"],'
                '"tool_inputs":{},"confidence":0.9,"needs_clarification":true,'
                '"reason":"缺少具体时间"}'
            )
        ),
    )

    result = tool_router_chain.route_tools_with_llm("帮我安排一下", "")

    assert result["should_call_tools"] is False
    assert result["selected_tools"] == []
    assert result["confidence"] == 0.9
    assert result["needs_clarification"] is True
