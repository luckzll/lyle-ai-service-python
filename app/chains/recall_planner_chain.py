import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.llm.model_router import TASK_INTENT, build_chat_model_for_task, is_task_model_configured
from app.memory.profile_schema import PROFILE_SCHEMA


def is_recall_planner_model_configured() -> bool:
    return is_task_model_configured(TASK_INTENT)


def invoke_recall_planner_chain(question: str) -> dict[str, object]:
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "你是 Lyle 的记忆召回规划器。只返回 JSON，不要解释。\n"
                "任务：根据用户问题选择需要召回的用户画像 schemaPaths。\n"
                "如果用户要求完整记忆，schemaPaths 返回 [\"*\"]。\n"
                "如果问题涉及多个方面，可以返回多个路径。\n"
                "不要返回不存在的路径；敏感信息只有用户明确询问隐私/敏感边界时才返回。\n"
                "返回格式：{\"schemaPaths\":[\"work.current_role\"],\"reason\":\"...\",\"intent\":\"project\",\"confidence\":0.8}"
            ),
        ),
        (
            "human",
            "可用 schemaPaths：\n{schema_paths}\n\n用户问题：{question}",
        ),
    ])
    chain = prompt | build_chat_model_for_task(TASK_INTENT) | StrOutputParser()
    content = chain.invoke({
        "question": question,
        "schema_paths": "\n".join(PROFILE_SCHEMA.keys()),
    })
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("Recall planner returned non-object JSON.")
    return parsed
