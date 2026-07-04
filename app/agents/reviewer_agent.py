import re

from app.chains.reviewer_chain import invoke_reviewer_chain
from app.core.config import settings


def run_reviewer_agent(
    question: str,
    answer: str,
    contexts: list[dict[str, str]],
    memory_context: str,
    memory: dict[str, object] | None = None,
    rag_sufficiency: dict[str, object] | None = None,
) -> dict[str, object]:
    rule_review = _rule_memory_governance_review(answer, memory or {})
    if not rule_review["approved"]:
        return rule_review
    rag_review = _rule_rag_grounding_review(question, answer, contexts, rag_sufficiency or {})
    if not rag_review["approved"]:
        return rag_review
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


def _rule_memory_governance_review(answer: str, memory: dict[str, object]) -> dict[str, object]:
    governance = memory.get("governance") if isinstance(memory.get("governance"), dict) else {}
    blocked_terms = governance.get("blockedTerms") if isinstance(governance, dict) else []
    for item in blocked_terms if isinstance(blocked_terms, list) else []:
        if not isinstance(item, dict):
            continue
        value = str(item.get("value") or "").strip()
        if value and value in answer:
            status = str(item.get("status") or "blocked")
            return {
                "approved": False,
                "score": 0.1,
                "reason": f"回答引用了已治理记忆：{status}",
                "rewrite_instruction": "请删除已禁用、已遗忘或敏感记忆相关内容，基于当前可用上下文重新回答。",
                "skipped": False,
                "ruleBlocked": True,
            }

    low_confidence_terms = governance.get("lowConfidenceTerms") if isinstance(governance, dict) else []
    for item in low_confidence_terms if isinstance(low_confidence_terms, list) else []:
        if not isinstance(item, dict):
            continue
        value = str(item.get("value") or "").strip()
        if value and value in answer and not _has_cautious_language(answer, value):
            return {
                "approved": False,
                "score": 0.45,
                "reason": "回答将低置信度记忆表述为确定事实",
                "rewrite_instruction": "请用“可能、你之前提到过、我不确定”等谨慎表达，不要把低置信度记忆说成确定事实。",
                "skipped": False,
                "ruleBlocked": True,
            }

    return {
        "approved": True,
        "score": 1.0,
        "reason": "记忆治理规则通过",
        "rewrite_instruction": "",
        "skipped": True,
    }


def _has_cautious_language(answer: str, value: str) -> bool:
    index = answer.find(value)
    if index < 0:
        return False
    window = answer[max(0, index - 16): index + len(value) + 16]
    return any(marker in window for marker in ["可能", "似乎", "不确定", "你之前提到过", "仅作参考", "如果我没记错"])


def _rule_rag_grounding_review(
    question: str,
    answer: str,
    contexts: list[dict[str, str]],
    sufficiency: dict[str, object],
) -> dict[str, object]:
    if not contexts and not sufficiency:
        return _approved_rule("RAG 规则跳过：无知识库上下文")

    if sufficiency and not bool(sufficiency.get("sufficient", True)):
        if not _answer_admits_insufficient_evidence(answer):
            return {
                "approved": False,
                "score": 0.35,
                "reason": "知识库资料不足，但回答没有说明不确定或资料不足",
                "rewrite_instruction": "请明确说明当前资料不足或无法确认，只回答可由已检索资料支持的部分。",
                "skipped": False,
                "ruleBlocked": True,
            }

    if contexts and _answer_claims_document_grounding(answer):
        coverage = _context_answer_coverage(answer, contexts)
        if coverage < 0.12:
            return {
                "approved": False,
                "score": 0.4,
                "reason": "回答声称依据资料，但与检索片段关键词覆盖过低",
                "rewrite_instruction": "请只基于资料来源中的内容回答；如果资料没有覆盖，请说明资料不足。",
                "skipped": False,
                "ruleBlocked": True,
                "citationCoverage": coverage,
            }

    return _approved_rule("RAG 规则通过")


def _approved_rule(reason: str) -> dict[str, object]:
    return {
        "approved": True,
        "score": 1.0,
        "reason": reason,
        "rewrite_instruction": "",
        "skipped": True,
    }


def _answer_admits_insufficient_evidence(answer: str) -> bool:
    return any(marker in answer for marker in ["资料不足", "无法确认", "不确定", "没有足够", "未提到", "无法从资料", "当前资料"])


def _answer_claims_document_grounding(answer: str) -> bool:
    return any(marker in answer for marker in ["根据资料", "根据文档", "资料显示", "文档显示", "从资料", "来源显示"])


def _context_answer_coverage(answer: str, contexts: list[dict[str, str]]) -> float:
    answer_terms = _significant_terms(answer)
    if not answer_terms:
        return 1.0
    context_text = "\n".join(f"{item.get('title', '')}\n{item.get('chunk', '')}" for item in contexts)
    context_terms = set(_significant_terms(context_text))
    if not context_terms:
        return 0.0
    matched = sum(1 for term in answer_terms if term in context_terms)
    return round(matched / len(answer_terms), 4)


def _significant_terms(text: str) -> list[str]:
    latin = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9._-]{2,}", text)
    chinese_chunks = [
        item
        for item in re.split(r"[\s，。！？、,.!?；;：:（）()\[\]【】\"'“”]+", text)
        if len(item) >= 2 and re.search(r"[\u4e00-\u9fff]", item)
    ]
    chinese_bigrams: list[str] = []
    for chunk in chinese_chunks:
        if len(chunk) <= 8:
            chinese_bigrams.append(chunk)
            continue
        chinese_bigrams.extend(chunk[index:index + 2] for index in range(0, len(chunk) - 1))
    stopwords = {"根据", "资料", "文档", "显示", "这个", "一个", "可以", "用户", "回答", "当前"}
    terms: list[str] = []
    seen: set[str] = set()
    for raw in [*latin, *chinese_bigrams]:
        term = raw.strip().lower()
        if len(term) < 2 or term in stopwords or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms[:80]
