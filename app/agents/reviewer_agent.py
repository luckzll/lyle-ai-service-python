from app.chains.reviewer_chain import invoke_reviewer_chain
from app.core.config import settings


def run_reviewer_agent(
    question: str,
    answer: str,
    contexts: list[dict[str, str]],
    memory_context: str,
) -> dict[str, object]:
    if not settings.reviewer_enabled:
        return {
            "approved": True,
            "score": 1.0,
            "reason": "Reviewer disabled",
            "rewrite_instruction": "",
            "skipped": True,
        }
    review = invoke_reviewer_chain(question, answer, contexts, memory_context)
    return {
        "approved": bool(review.get("approved", True)),
        "score": float(review.get("score") or 0),
        "reason": str(review.get("reason") or ""),
        "rewrite_instruction": str(review.get("rewrite_instruction") or ""),
        "skipped": False,
    }
