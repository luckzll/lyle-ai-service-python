from langchain_openai import ChatOpenAI

from app.llm.model_router import build_chat_model_for_task


def build_deepseek_chat_model(streaming: bool = False) -> ChatOpenAI:
    return build_chat_model_for_task("chat", streaming=streaming)


def invoke_deepseek_chain(question: str, contexts: list[dict[str, str]], memory_context: str = "") -> str:
    from app.chains.answer_chain import invoke_answer_chain

    return invoke_answer_chain(question, contexts, memory_context)
