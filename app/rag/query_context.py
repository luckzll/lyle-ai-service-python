"""Request-scoped, conservative query reference resolution from recent user turns.

History is data, never an instruction source. Only short, unambiguous entity
anchors are used; the full conversation is not appended to a retrieval query.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
import json
import logging
import re
from typing import Any

from app.core.runtime_config import current_rag_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryContext:
    user_id: str
    conversation_id: str | None
    original_question: str
    effective_question: str
    additional_queries: tuple[str, ...] = ()
    history_fingerprint: str = ""
    resolution_method: str = "none"
    resolution_reason: str = "not_referential"


_context: ContextVar[QueryContext | None] = ContextVar("rag_query_context", default=None)
_REFERENCE = re.compile(
    r"(?:这个|那个|这份|那份|这种|那种|刚才(?:的|那个|这个)?|上述|前面(?:的|那个|这个)?)(?:版本|项目|知识库|文档|方案|模型)?|它|"
    r"\b(?:it|this version|that version|this project|that project)\b", re.IGNORECASE,
)
_ELLIPTICAL_CHANGE = re.compile(r"^(?:那)?具体(?:都)?(?:改了|改|更新了|变化了|变化|更新)|^有哪些(?:变化|改动|更新)")
_PLURAL = re.compile(r"它们|这些|那些|这(?:两|几|些)|那(?:两|几|些)|两者|分别|\b(?:they|them|both)\b", re.IGNORECASE)
_VERSION = re.compile(r"(?<![A-Za-z0-9_.])v?\d+\.\d+(?:\.\d+){0,2}(?:[-+][A-Za-z0-9.-]+)?(?![A-Za-z0-9_.])", re.IGNORECASE)
_TYPED = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff-]{0,24}?(?:知识库|团队版|轻量版|免费版|专业版|项目|文档|方案|模型)")
_ACK = re.compile(r"^(?:嗯|好|好的|知道了|了解了|明白了|谢谢|继续|是的|ok|okay)[。！!\s]*$", re.IGNORECASE)
_TOPIC_SHIFT = re.compile(r"换个话题|说点别的|不聊这个|不再讨论|另一个问题|现在(?:改为|想问|说的是)|转而")
_INTRO = re.compile(
    r"^(?:(?:我|我们)(?:刚刚|刚|现在|正在|目前|已经|想|要|在)*|"
    r"(?:刚刚|刚|现在|正在|目前|已经)|(?:看到了|看见了|看到|看见|在看|查看|看一下|看看|看|"
    r"介绍一下|介绍|了解一下|了解|说的是|说说|使用的是|用的是|使用|升级到|关注|有一个|有个|一个|这份|这个|那个))+"
)


def current_query_context() -> QueryContext | None:
    return _context.get()


def _safe_scope(user_id: str, conversation_id: str | None) -> bool:
    # The existing short-term store joins these values with ':'. Reject ambiguous
    # keys and never fall back to the store's shared 'default' conversation.
    return bool(user_id and conversation_id and all(
        value == value.strip() and len(value) <= 256 and ":" not in value
        and not any(ord(char) < 32 for char in value)
        for value in (user_id, conversation_id)
    ))


def _versions(text: str) -> list[re.Match[str]]:
    matches = []
    for match in _VERSION.finditer(text):
        value = match.group()
        before = text[max(0, match.start() - 32):match.start()]
        after = text[match.end():match.end() + 4]
        if re.match(r"\s*(?:元|万元|美元|%|MB|GB|秒|毫秒)", after, re.IGNORECASE):
            continue
        if (value.count(".") >= 2 or value.lower().startswith("v") or "版本" in before
                or re.search(r"(?:version|[A-Za-z][A-Za-z0-9_+#-]{1,24})\s+$", before, re.IGNORECASE)):
            matches.append(match)
    return matches


def needs_query_context(question: str) -> bool:
    value = question.strip()
    if not value or len(value) > 160 or _PLURAL.search(value) or _versions(value):
        return False
    # An explicitly named project/document/plan already identifies the subject.
    if any(_clean_name(match.group()) not in {
        "知识库", "项目", "文档", "方案", "模型", "版本", "这个知识库", "那个知识库",
        "这个项目", "那个项目", "这份文档", "那份文档",
    } for match in _TYPED.finditer(value)):
        return False
    references = list(_REFERENCE.finditer(value))
    if len(references) > 1:
        return False
    return len(references) == 1 or bool(_ELLIPTICAL_CHANGE.search(value))


def _clean_name(value: str) -> str:
    value = _INTRO.sub("", value.strip())
    return value.strip(" ，,。.!！?？:：;；的")


def _entity_anchors(text: str) -> list[str]:
    """Read one recent user turn; multiple plausible subjects remain ambiguous."""
    anchors = []
    for match in _versions(text):
        before = text[max(0, match.start() - 48):match.start()]
        before = re.split(r"[，,。!?！？;；:：\n]|和|与|以及|、", before)[-1]
        before = re.sub(r"(?:版本|version)\s*$", "", before, flags=re.IGNORECASE).strip()
        name_match = re.search(r"([A-Za-z][A-Za-z0-9_+#-]{0,24}|[\u4e00-\u9fff]{1,24})$", before)
        name = _clean_name(name_match.group()) if name_match else ""
        if name in {"版本", "version", "v", "它", "这是", "是", "为"}:
            name = ""
        anchors.append(f"{name} {match.group()}".strip())
    # A comparison of two products sharing a version must also stay ambiguous.
    if anchors:
        for match in _TYPED.finditer(text):
            name = _clean_name(match.group())
            if (name not in {"知识库", "项目", "文档", "方案", "模型", "发布文档", "说明文档", "更新文档"}
                    and not any(name in anchor for anchor in anchors)):
                anchors.append(name)
        return list(dict.fromkeys(anchors))
    for match in _TYPED.finditer(text):
        name = _clean_name(match.group())
        if "知识库" in name and re.search(r"(?:删除|删掉|删了).{0,5}知识库", name):
            name = "删除的知识库"
        if name and name not in {"项目", "文档", "方案", "模型", "团队版", "轻量版", "免费版", "专业版"}:
            anchors.append(name)
    if not anchors:
        topic = re.search(r"(?:介绍一下|介绍|了解一下|了解|说说)\s*([A-Za-z0-9_\u4e00-\u9fff-]{2,24})[。！？?!\s]*$", text)
        if topic and not _REFERENCE.search(topic.group(1)):
            anchors.append(topic.group(1))
    return list(dict.fromkeys(anchors))


def _user_history(history: Sequence[Any], turns: int, max_chars: int) -> tuple[list[str], bool]:
    selected = []
    budget = max_chars
    for row in reversed(list(history)[-turns:]):
        role = row.get("role") if isinstance(row, Mapping) else getattr(row, "role", None)
        content = row.get("content") if isinstance(row, Mapping) else getattr(row, "content", None)
        if role != "user" or not isinstance(content, str) or not content.strip():
            continue
        text = content.strip()
        if len(text) > budget:
            # Truncating a comparison can erase its second entity. Do not select
            # an older subject after dropping the most recent substantive turn.
            return selected, True
        selected.append(text)
        budget -= len(text)
    return selected, False


def resolve_query_reference(question: str, history: Sequence[Any], *, turns: int = 6, max_chars: int = 1200) -> QueryContext:
    """Pure helper for replay/tests. Scope identity is attached by the caller."""
    base = QueryContext("", None, question, question)
    if not needs_query_context(question):
        return base
    recent, truncated = _user_history(history, max(1, min(20, turns)), max(1, min(8000, max_chars)))
    fingerprint = sha256(json.dumps(recent, ensure_ascii=False).encode()).hexdigest() if recent else ""
    reason = "history_truncated" if truncated and not recent else "no_antecedent"
    for text in recent:
        if _TOPIC_SHIFT.search(text):
            reason = "topic_changed"
            break
        if re.search(r"对比|比较|两者|分别|\b(?:compare|versus|both)\b", text, re.IGNORECASE):
            reason = "ambiguous_antecedent"
            break
        anchors = _entity_anchors(text)
        if len(anchors) > 1:
            reason = "ambiguous_antecedent"
            break
        if anchors:
            anchor = anchors[0]
            effective = _REFERENCE.sub(lambda _: anchor, question, count=1) if _REFERENCE.search(question) else f"{anchor} {question}"
            return QueryContext("", None, question, effective, (question,), fingerprint, "rule", "single_entity")
        if not (_ACK.fullmatch(text) or needs_query_context(text)):
            # A new unrelated substantive user turn ends antecedent lookup.
            reason = "topic_changed"
            break
    return QueryContext("", None, question, question, (), fingerprint, "none", reason)


@contextmanager
def query_context_scope(user_id: str, conversation_id: str | None, question: str, *, history: Sequence[Any] | None = None) -> Iterator[QueryContext]:
    config = current_rag_config()
    context = QueryContext(str(user_id or ""), conversation_id, question, question)
    if getattr(config, "rag_contextual_query_enabled", True) and _safe_scope(context.user_id, conversation_id) and needs_query_context(question):
        try:
            if history is None:
                from app.memory.store import _load_short_term
                history = _load_short_term(context.user_id, conversation_id)
            resolved = resolve_query_reference(question, history,
                turns=getattr(config, "rag_context_history_turns", 6),
                max_chars=getattr(config, "rag_context_history_max_chars", 1200))
            context = QueryContext(context.user_id, conversation_id, resolved.original_question, resolved.effective_question,
                                   resolved.additional_queries, resolved.history_fingerprint,
                                   resolved.resolution_method, resolved.resolution_reason)
        except PermissionError:
            raise
        except Exception as error:
            logger.warning("RAG recent history unavailable. error_type=%s", type(error).__name__)
            context = QueryContext(context.user_id, conversation_id, question, question, resolution_reason="history_unavailable")
    token = _context.set(context)
    try:
        yield context
    finally:
        _context.reset(token)
