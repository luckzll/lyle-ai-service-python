from langchain_core.prompts import ChatPromptTemplate

from app.llm.prompts import build_user_prompt, get_system_prompt
from app.llm.tasks.chat_model import build_chat_task_model, is_chat_model_configured


def invoke_answer_chain(question: str, contexts: list[dict[str, str]], memory_context: str = "") -> str:
    if not is_chat_model_configured():
        return "DeepSeek API Key 还没有配置。请在环境变量 DEEPSEEK_API_KEY 中填入你的 Key。"

    system_prompt, prompt_version = get_system_prompt()
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"{system_prompt}\n\nPrompt-Version: {prompt_version}"),
        ("human", "{user_prompt}"),
    ])
    chain = prompt | build_chat_task_model()
    response = chain.invoke({
        "user_prompt": build_user_prompt(question, contexts, memory_context),
    })
    return str(response.content)
