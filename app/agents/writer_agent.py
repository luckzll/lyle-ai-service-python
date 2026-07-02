from app.chains.answer_chain import invoke_answer_chain


def run_writer_agent(
    question: str,
    contexts: list[dict[str, str]],
    memory_context: str,
    tool_context: str = "",
    review_feedback: str = "",
) -> str:
    combined_memory_context = "\n\n".join(
        item for item in [memory_context, tool_context] if item
    )
    if not review_feedback:
        return invoke_answer_chain(question, contexts, combined_memory_context)
    rewritten_question = (
        f"{question}\n\n"
        "上一次回答没有通过审核，请根据以下审核意见重新回答：\n"
        f"{review_feedback}"
    )
    return invoke_answer_chain(rewritten_question, contexts, combined_memory_context)
