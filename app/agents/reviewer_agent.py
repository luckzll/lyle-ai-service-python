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
    tool_results: list[dict[str, object]] | None = None,
    project_context: dict[str, object] | None = None,
    candidate_goals: list[dict[str, object]] | None = None,
    candidate_plans: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    rule_review = _rule_memory_governance_review(answer, memory or {})
    if not rule_review["approved"]:
        return rule_review
    rag_review = _rule_rag_grounding_review(question, answer, contexts, rag_sufficiency or {})
    if not rag_review["approved"]:
        return rag_review
    v3_review = _rule_v3_project_work_review(
        question=question,
        answer=answer,
        tool_results=tool_results or [],
        project_context=project_context or {},
        candidate_goals=candidate_goals or [],
        candidate_plans=candidate_plans or [],
    )
    if not v3_review["approved"]:
        return v3_review
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


def _rule_v3_project_work_review(
    *,
    question: str,
    answer: str,
    tool_results: list[dict[str, object]],
    project_context: dict[str, object],
    candidate_goals: list[dict[str, object]],
    candidate_plans: list[dict[str, object]],
) -> dict[str, object]:
    failed_connectors = [
        item for item in tool_results
        if isinstance(item, dict)
        and (item.get("connector") or str(item.get("tool") or "").startswith("connector."))
        and item.get("ok") is False
    ]
    pending_operations = [
        item for item in tool_results
        if isinstance(item, dict)
        and item.get("requiresConfirmation")
        and isinstance(item.get("operation"), dict)
        and (item.get("operation") or {}).get("status") == "pending"
    ]
    if pending_operations and _answer_self_confirms_pending_operation(answer):
        return {
            "approved": False,
            "score": 0.35,
            "reason": "已有待确认操作卡，但回答出现自问自答式确认文案",
            "rewrite_instruction": "请只说明已生成待确认操作，请用户在卡片中点击确认或拒绝；不要问完又替用户回答，也不要暗示已经执行。",
            "skipped": False,
            "ruleBlocked": True,
            "v3Rule": "pending_operation_confirmation_copy",
        }

    if failed_connectors and _answer_claims_connector_success(answer) and not _answer_admits_connector_failure(answer):
        return {
            "approved": False,
            "score": 0.25,
            "reason": "连接器读取失败，但回答声称已成功读取或使用结果",
            "rewrite_instruction": "请明确说明连接器读取失败，展示错误原因和是否可重试，不要编造外部来源内容。",
            "skipped": False,
            "ruleBlocked": True,
            "v3Rule": "connector_failure",
        }

    if candidate_plans and _answer_claims_task_execution(answer):
        return {
            "approved": False,
            "score": 0.3,
            "reason": "长任务需要先展示计划并等待确认，回答不能声称已执行写操作",
            "rewrite_instruction": "请改为展示计划、说明需要用户确认；涉及写入或外部变更时生成待确认操作。",
            "skipped": False,
            "ruleBlocked": True,
            "v3Rule": "plan_confirmation",
        }

    if _answer_misuses_profile_as_project_source(answer):
        return {
            "approved": False,
            "score": 0.35,
            "reason": "回答把用户画像或个人偏好冒充为项目事实来源",
            "rewrite_instruction": "请区分用户画像和项目事实。项目事实只能来自项目文件、项目对话、任务、决策或开放问题。",
            "skipped": False,
            "ruleBlocked": True,
            "v3Rule": "project_fact_grounding",
        }

    resolver = project_context.get("resolver") if isinstance(project_context.get("resolver"), dict) else {}
    if candidate_goals and _answer_claims_goal_recorded(answer):
        return {
            "approved": False,
            "score": 0.3,
            "reason": "目标候选需要用户确认，回答不能声称已经直接记录",
            "rewrite_instruction": "请说明已经生成待确认目标，等待用户在目标卡片中确认；不要声称已直接写入项目目标。",
            "skipped": False,
            "ruleBlocked": True,
            "v3Rule": "goal_confirmation",
        }

    if _question_requests_project_goal_write(question) and not resolver.get("projectId") and _answer_claims_goal_recorded(answer):
        return {
            "approved": False,
            "score": 0.25,
            "reason": "项目未匹配成功，但回答声称目标已经记录或挂到项目",
            "rewrite_instruction": "请说明没有匹配到明确项目，让用户先选择或创建正确项目；不要声称目标已经记录。",
            "skipped": False,
            "ruleBlocked": True,
            "v3Rule": "goal_project_missing",
        }

    if resolver.get("requiresClarification") and _answer_claims_project_linked(answer):
        return {
            "approved": False,
            "score": 0.35,
            "reason": "项目归属需要澄清，但回答声称已完成项目关联",
            "rewrite_instruction": "请先向用户澄清项目归属，不要在低置信度或多项目命中时声称已关联。",
            "skipped": False,
            "ruleBlocked": True,
            "v3Rule": "project_clarification",
        }

    return _approved_rule("V3 项目推进规则通过")


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


def _answer_claims_connector_success(answer: str) -> bool:
    return any(marker in answer for marker in ["已读取", "读取到", "搜索到", "命中", "根据 GitHub", "根据连接器", "GitHub 显示"])


def _answer_admits_connector_failure(answer: str) -> bool:
    return any(marker in answer for marker in ["读取失败", "连接器失败", "未配置", "无法读取", "授权", "token", "重试"])


def _answer_claims_task_execution(answer: str) -> bool:
    return any(marker in answer for marker in ["已执行", "已创建", "已写入", "已经完成", "我已帮你完成", "我已经创建"])


def _answer_misuses_profile_as_project_source(answer: str) -> bool:
    source_markers = ["用户画像", "个人偏好", "长期记忆", "你的偏好"]
    fact_markers = ["项目事实", "项目结论", "项目来源", "项目已经", "因此项目"]
    return any(source in answer for source in source_markers) and any(fact in answer for fact in fact_markers)


def _answer_claims_project_linked(answer: str) -> bool:
    return any(marker in answer for marker in ["已关联到项目", "已经关联项目", "已归入项目", "已链接到项目"])


def _question_requests_project_goal_write(question: str) -> bool:
    return (
        any(marker in question for marker in ["新增一个目标", "新增目标", "记录一个目标", "添加目标", "创建目标", "目标：", "目标:"])
        and any(marker in question for marker in ["项目", "project", "版本"])
    )


def _answer_claims_goal_recorded(answer: str) -> bool:
    write_markers = [
        "已经直接记录",
        "已直接记录",
        "直接记录到",
        "直接记录进",
        "目标已经记录",
        "目标已记录",
        "已经记录下来",
        "已记录下来",
        "记录到项目",
        "记录进项目",
        "已经挂到",
        "已挂到",
        "挂到项目",
        "已经写入",
        "已写入",
        "写入项目",
        "已经创建目标",
        "已创建目标",
    ]
    if any(marker in answer for marker in write_markers):
        return True
    return bool(re.search(r"(?:已|已经|直接).{0,12}(?:记录|挂到|写入|创建).{0,16}项目", answer))


def _answer_self_confirms_pending_operation(answer: str) -> bool:
    asks_confirmation = any(marker in answer for marker in ["要现在确认", "现在确认创建", "需要你点一下确认", "点击确认才"])
    self_answer = any(marker in answer for marker in ["好的，我已经", "我已经准备好了", "我已经帮你准备", "只需你点击"])
    executed_claim = any(marker in answer for marker in ["正式创建了", "已经创建到", "已创建到", "已经执行"])
    return (asks_confirmation and self_answer) or executed_claim


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
