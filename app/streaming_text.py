"""Conservative incremental output for turns without whole-answer dependencies."""
from collections.abc import Callable


class IncrementalReviewBlocked(RuntimeError):
    pass


def streaming_policy(*, reviewer_enabled: bool, contexts: list, memory: dict,
                     sufficiency: dict, tool_results: list, project_context: dict,
                     candidate_goals: list, candidate_plans: list,
                     memory_statement: bool, memory_overview: bool,
                     has_attachments: bool) -> str:
    if reviewer_enabled:
        return "full_model_review"
    if contexts or sufficiency.get("sufficient") is False:
        return "rag_grounding_review"
    governance = memory.get("governance") or {}
    if governance.get("blockedTerms") or governance.get("lowConfidenceTerms"):
        return "memory_governance_review"
    if memory_statement or memory_overview:
        return "memory_confirmation_review"
    if tool_results or _has_project_dependencies(project_context) or candidate_goals or candidate_plans or has_attachments:
        return "tool_or_project_review"
    return "incremental"


def _has_project_dependencies(project_context: dict) -> bool:
    # The API returns a resolver envelope even when no project matched. That
    # envelope alone must not turn every ordinary response into buffered output.
    resolver = project_context.get("resolver") or {}
    return bool(
        (isinstance(resolver, dict) and (resolver.get("projectId") or resolver.get("requiresClarification")))
        or project_context.get("context")
    )


class IncrementalAnswer:
    """Hold a short suffix and unfinished images; check before publishing.

    Existing whole-answer gates stay buffered. This path checks rule-only
    cumulative output and fails closed rather than rewriting visible content.
    """
    def __init__(self, sanitize: Callable[[str], str], approved: Callable[[str], bool], holdback: int = 64):
        self.raw = ""
        self.published = ""
        self.sanitize = sanitize
        self.approved = approved
        self.holdback = holdback

    def push(self, chunk: str = "", *, final: bool = False) -> str:
        self.raw += chunk
        stable = self.raw
        if not final:
            # Keep incomplete Markdown images private until their closing ')'.
            # A long alt/URL can span any number of upstream chunks.
            start = 0
            while (start := stable.find("![", start)) >= 0:
                label_end = stable.find("]", start + 2)
                close = stable.find(")", label_end + 1) if label_end >= 0 else -1
                if close < 0:
                    stable = stable[:start]
                    break
                start = close + 1
        safe = self.sanitize(stable)
        if safe and not self.approved(safe):
            raise IncrementalReviewBlocked("回答内容检查未通过，已停止输出；请调整问题后重试。")
        end = len(safe) if final else max(0, len(safe) - self.holdback)
        candidate = safe[:end]
        if len(candidate) < len(self.published):
            return ""
        if not candidate.startswith(self.published):
            raise IncrementalReviewBlocked("回答内容需要重新检查，已保留已显示部分，请重试。")
        delta = candidate[len(self.published):]
        self.published = candidate
        return delta
