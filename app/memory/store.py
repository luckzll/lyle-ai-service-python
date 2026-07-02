from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from threading import Lock
from time import time
from uuid import NAMESPACE_URL, uuid5

import httpx
import redis

from app.core.config import settings
from app.core.database import postgres_connection
from app.embedding.client import EMBEDDING_DIMENSION, embed_text
from app.chains.structured_memory_chain import invoke_structured_memory_chain
from app.llm.client import summarize_memory
from app.rag.reranker import rerank_items
import logging


@dataclass
class MemoryTurn:
    role: str
    content: str
    created_at: float


@dataclass
class StructuredMemory:
    key: str
    value: str
    source: str
    updated_at: float
    status: str = "active"
    confidence: float = 1.0
    expires_at: float | None = None


_lock = Lock()
_short_term: dict[str, list[MemoryTurn]] = defaultdict(list)
_summaries: dict[str, str] = {}
_structured: dict[str, dict[str, StructuredMemory]] = defaultdict(dict)
_postgres_ready = False
_qdrant_memory_ready = False
_milvus_ready = False
_vector_backend_disabled_until: dict[str, float] = {}
logger = logging.getLogger(__name__)
_QUESTION_LIKE_PATTERN = re.compile(r"(是什么|是谁|什么|哪里|哪个|多少|怎么|为什么|为何|吗|呢|啥|？|\?)")
_INVALID_STRUCTURED_VALUES = {"什么", "啥", "谁", "哪里", "哪个", "多少"}
_FOOD_PREFERENCE_TERMS = {
    "西瓜", "香蕉", "苹果", "橘子", "橙子", "葡萄", "草莓", "芒果", "榴莲", "梨", "桃子",
    "火锅", "烧烤", "咖啡", "奶茶", "甜品", "米饭", "面条", "饺子", "汉堡", "披萨",
}
_PET_PREFERENCE_TERMS = {
    "狗", "小狗", "小狗狗", "犬", "猫", "小猫", "猫咪", "宠物", "金毛", "柯基", "柴犬", "布偶猫",
}
_SPORT_PREFERENCE_TERMS = {
    "足球", "篮球", "网球", "羽毛球", "乒乓球", "世界杯", "英超", "西甲", "NBA",
}
_ENTERTAINMENT_PREFERENCE_TERMS = {
    "唱歌", "音乐", "电影", "游戏", "动漫", "二次元", "恋与深空", "阅读", "摄影",
}
_DATA_IMAGE_PATTERN = re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\r\n\s]+")


def add_turn(user_id: str, conversation_id: str | None, role: str, content: str) -> None:
    key = _conversation_key(user_id, conversation_id)
    turn = MemoryTurn(role=role, content=_sanitize_memory_content(content), created_at=time())
    if settings.redis_enabled:
        client = _redis_client()
        client.rpush(_short_term_key(key), json.dumps(asdict(turn), ensure_ascii=False))
        client.ltrim(_short_term_key(key), -_short_term_buffer_limit(), -1)
    else:
        with _lock:
            _short_term[key].append(turn)


def recall_memory(
    user_id: str,
    conversation_id: str | None,
    question: str,
    include_vector: bool = True,
) -> dict[str, object]:
    short = _load_short_term(user_id, conversation_id)[-settings.memory_recall_limit:]
    summary_result = _load_summary_with_meta(user_id, conversation_id, allow_fallback=True)
    summary = str(summary_result["summary"])
    summary_meta = {key: value for key, value in summary_result.items() if key != "summary"}
    structured = _load_structured(user_id, question)
    candidate_limit = _memory_candidate_limit()
    semantic: list[dict[str, object]] = []
    vector: list[dict[str, object]] = []
    if include_vector:
        try:
            semantic = _search_qdrant_memory(user_id, question, candidate_limit)
        except Exception as exception:
            _mark_vector_backend_failure("qdrant_memory", exception)
            logger.exception("Qdrant memory recall failed. user_id=%s", user_id)
            semantic = []
        try:
            vector = _search_milvus_memory(user_id, question, candidate_limit)
        except Exception as exception:
            _mark_vector_backend_failure("milvus_memory", exception)
            logger.exception("Milvus memory recall failed. user_id=%s", user_id)
            vector = []
    retrieved = _merge_vector_memories(question, semantic, vector, settings.memory_recall_limit)

    return {
        "summary": summary,
        "summary_meta": summary_meta,
        "short_term": [asdict(item) for item in short],
        "structured": [asdict(item) for item in structured],
        "semantic": semantic,
        "vector": vector,
        "retrieved": retrieved,
    }


def remember_interaction(user_id: str, conversation_id: str | None, question: str, answer: str) -> dict[str, object]:
    add_turn(user_id, conversation_id, "user", question)
    add_turn(user_id, conversation_id, "assistant", answer)
    compress_memory(user_id, conversation_id)
    candidate_memories = extract_structured_memory(user_id, conversation_id, question)
    text = _build_vector_memory_text(user_id, question)
    should_write, reason = _should_write_vector_memory(question, text)
    vector_memory_write: dict[str, object] = {
        "enabled": settings.memory_vector_write_policy_enabled,
        "shouldWrite": should_write,
        "reason": reason,
        "textChars": len(text),
        "qdrantEnabled": settings.qdrant_enabled,
        "milvusEnabled": settings.milvus_enabled,
        "duplicateSkipped": False,
        "duplicateScore": None,
        "duplicateTextPreview": "",
        "writtenBackends": [],
        "failedBackends": [],
    }
    if not should_write:
        logger.info(
            "Vector memory write skipped. user_id=%s conversation_id=%s reason=%s",
            user_id,
            conversation_id,
            reason,
        )
        return {"vectorMemoryWrite": vector_memory_write, "candidateMemories": candidate_memories}
    duplicate = _find_duplicate_vector_memory(user_id, text)
    if duplicate:
        vector_memory_write.update({
            "shouldWrite": False,
            "reason": "duplicate_vector_memory",
            "duplicateSkipped": True,
            "duplicateScore": duplicate.get("score"),
            "duplicateTextPreview": str(duplicate.get("text") or "")[:120],
        })
        logger.info(
            "Vector memory write deduped. user_id=%s conversation_id=%s score=%s",
            user_id,
            conversation_id,
            duplicate.get("score"),
        )
        return {"vectorMemoryWrite": vector_memory_write, "candidateMemories": candidate_memories}
    try:
        _upsert_qdrant_memory(user_id, conversation_id, text)
        if settings.qdrant_enabled:
            vector_memory_write["writtenBackends"].append("qdrant")  # type: ignore[union-attr]
    except Exception as exception:
        _mark_vector_backend_failure("qdrant_memory", exception)
        vector_memory_write["failedBackends"].append("qdrant")  # type: ignore[union-attr]
        logger.exception("Qdrant memory write failed. user_id=%s conversation_id=%s", user_id, conversation_id)
    try:
        _upsert_milvus_memory(user_id, conversation_id, text)
        if settings.milvus_enabled:
            vector_memory_write["writtenBackends"].append("milvus")  # type: ignore[union-attr]
    except Exception as exception:
        _mark_vector_backend_failure("milvus_memory", exception)
        vector_memory_write["failedBackends"].append("milvus")  # type: ignore[union-attr]
        logger.exception("Milvus memory write failed. user_id=%s conversation_id=%s", user_id, conversation_id)
    return {"vectorMemoryWrite": vector_memory_write, "candidateMemories": candidate_memories}


def compress_memory(user_id: str, conversation_id: str | None) -> None:
    turns = _load_short_term(user_id, conversation_id)
    if len(turns) <= _compression_trigger_limit():
        return

    overflow = turns[:-settings.memory_short_term_limit]
    kept = turns[-settings.memory_short_term_limit:]
    old_summary = _load_summary(user_id, conversation_id, allow_fallback=False)
    compressed = "\n".join(f"{item.role}: {item.content}" for item in overflow)
    merged = summarize_memory(old_summary, compressed)
    _save_summary(user_id, conversation_id, merged)
    _save_short_term(user_id, conversation_id, kept)
    logger.info(
        "Memory compressed. user_id=%s conversation_id=%s overflow=%s kept=%s",
        user_id,
        conversation_id,
        len(overflow),
        len(kept),
    )


def extract_structured_memory(user_id: str, conversation_id: str | None, text: str) -> list[dict[str, object]]:
    candidates = _extract_structured_candidates(
        text,
        _load_short_term(user_id, conversation_id),
    )
    if not candidates:
        return []

    candidate_memories: list[dict[str, object]] = []
    for key, value, confidence in candidates:
        candidate_key = _pending_structured_memory_key(user_id, conversation_id, text, key, value)
        item = upsert_structured_memory(
            user_id,
            candidate_key,
            value,
            source=text[:240],
            confidence=confidence,
            status="pending",
        )
        item["targetKey"] = key
        candidate_memories.append(item)
    return candidate_memories


def format_memory_context(memory: dict[str, object]) -> str:
    parts: list[str] = []
    summary = memory.get("summary")
    if summary:
        parts.append(f"长期摘要：{summary}")

    structured = memory.get("structured") or []
    if structured:
        facts = [f"{item['key']}={item['value']}" for item in structured]  # type: ignore[index]
        parts.append("结构化记忆：" + "；".join(facts))

    retrieved = memory.get("retrieved") or []
    if retrieved:
        lines = [
            f"- [{item.get('backend', 'vector')}] {item['text']}"
            for item in retrieved  # type: ignore[union-attr]
        ]
        parts.append("向量召回记忆：\n" + "\n".join(lines))

    short_term = memory.get("short_term") or []
    if short_term:
        lines = [f"{item['role']}：{item['content']}" for item in short_term]  # type: ignore[index]
        parts.append("短期记忆(Redis)：\n" + "\n".join(lines))

    return "\n\n".join(parts)


def export_memory(user_id: str, conversation_id: str | None) -> dict[str, object]:
    memory = recall_memory(user_id, conversation_id, "")
    memory["short_term"] = [
        _export_turn_preview(item)
        for item in memory.get("short_term", [])  # type: ignore[union-attr]
    ]
    memory["structured_all"] = [asdict(item) for item in _load_all_structured(user_id)]
    return memory


def _sanitize_memory_content(content: str) -> str:
    return _DATA_IMAGE_PATTERN.sub("[图片 data URL 已省略]", content)


def _export_turn_preview(item: dict[str, object]) -> dict[str, object]:
    content = _sanitize_memory_content(str(item.get("content") or ""))
    limit = max(settings.memory_export_turn_max_chars, 100)
    preview = content if len(content) <= limit else content[:limit] + "…"
    return {
        **item,
        "content": preview,
        "contentChars": len(content),
        "truncated": len(content) > limit,
    }


def _merge_vector_memories(
    query: str,
    semantic: list[dict[str, object]],
    vector: list[dict[str, object]],
    top_k: int,
) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for item in [*semantic, *vector]:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        score = _normalise_score(item.get("score"))
        if score < settings.memory_vector_score_threshold:
            continue
        backend = str(item.get("backend") or "vector")
        existing = merged.get(text)
        if existing is None:
            merged[text] = {
                **item,
                "text": text,
                "score": score,
                "backend": backend,
                "backends": [backend],
            }
            continue
        existing_backends = list(existing.get("backends") or [])
        if backend not in existing_backends:
            existing_backends.append(backend)
        if score > float(existing.get("score") or 0):
            existing.update({**item, "text": text, "score": score, "backend": ",".join(existing_backends)})
        else:
            existing["backend"] = ",".join(existing_backends)
        existing["backends"] = existing_backends

    ranked = sorted(
        merged.values(),
        key=lambda item: float(item.get("score") or 0),
        reverse=True,
    )
    return rerank_items(query, ranked, "text", top_k) if settings.rerank_enabled else ranked[:top_k]


def _build_vector_memory_text(user_id: str, question: str) -> str:
    parts = [f"用户：{question.strip()}"]
    structured = _load_structured(user_id, "")
    if structured:
        facts = "；".join(f"{item.key}={item.value}" for item in structured)
        parts.append(f"结构化事实：{facts}")
    text = "\n".join(parts)
    return text[:settings.memory_vector_text_max_chars]


def _should_write_vector_memory(question: str, text: str) -> tuple[bool, str]:
    if not settings.memory_vector_write_policy_enabled:
        return True, "policy_disabled"

    normalized = re.sub(r"\s+", "", question or "").lower()
    if not normalized:
        return False, "empty_question"

    noise_phrases = {
        "你好", "您好", "hi", "hello", "哈喽", "在吗",
        "哈哈", "哈哈哈", "呵呵", "嘿嘿",
        "嗯", "嗯嗯", "哦", "哦哦", "好", "好的", "好啊", "可以", "行", "行吧",
        "谢谢", "谢了", "多谢", "收到", "ok", "okay",
        "继续", "继续吧", "再说", "然后呢",
    }
    if normalized in noise_phrases:
        return False, "noise_phrase"

    value_signals = (
        "我喜欢", "我不喜欢", "我更喜欢", "我经常", "我是", "我的", "我叫", "职业",
        "星座", "项目", "正在做", "最近在", "计划", "目标", "偏好", "习惯",
        "记住", "记一下", "提醒", "待办", "日程",
        "搜索", "查一下", "联网", "生成", "画", "图片", "上传",
        "代码", "后端", "前端", "算法", "数据库", "接口", "部署", "测试", "bug",
        "加班", "下班", "吃饭", "干饭", "结婚", "已婚",
        "为什么", "怎么", "如何", "什么原因", "帮我", "分析",
    )
    if any(signal in question for signal in value_signals):
        return True, "value_signal"

    if len(normalized) < settings.memory_vector_write_min_chars:
        return False, "too_short"

    if "结构化事实：" in text:
        return True, "structured_context"

    return True, "default_keep"


def _find_duplicate_vector_memory(user_id: str, text: str) -> dict[str, object] | None:
    if (
        not settings.memory_vector_dedupe_enabled
        or not settings.qdrant_enabled
        or not _vector_backend_available("qdrant_memory")
    ):
        return None
    try:
        candidates = _search_qdrant_memory(user_id, text, 1)
    except Exception as exception:
        _mark_vector_backend_failure("qdrant_memory", exception)
        logger.warning("Vector memory dedupe search skipped after failure. user_id=%s error=%s", user_id, exception)
        return None
    if not candidates:
        return None
    best = candidates[0]
    score = _normalise_score(best.get("score"))
    if score >= settings.memory_vector_dedupe_score_threshold:
        return {**best, "score": score}
    return None


def _memory_candidate_limit() -> int:
    if not settings.rerank_enabled:
        return settings.memory_recall_limit
    return max(settings.memory_recall_limit, settings.rerank_candidate_limit)


def _normalise_score(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def upsert_structured_memory(
    user_id: str,
    key: str,
    value: str,
    source: str = "manual",
    confidence: float = 1.0,
    status: str = "active",
    expires_at: float | None = None,
) -> dict[str, object]:
    now = time()
    status = _normalise_memory_status(status)
    existing = _get_structured_memory(user_id, key)
    original_value = value
    if existing and existing.status == "active":
        merged_value = _merge_structured_memory_value(key, existing.value, value, source)
        if merged_value != value:
            value = merged_value
    conflict_action = _decide_structured_memory_conflict(existing, value, confidence, status)
    if existing and value != original_value and conflict_action == "replaced":
        conflict_action = "merged"
    if conflict_action == "ignored":
        _record_structured_memory_history(
            user_id,
            key,
            existing.value if existing else None,
            value,
            "ignored",
            source,
            confidence,
            status,
        )
        logger.info(
            "Structured memory conflict ignored. user_id=%s key=%s old_value=%s new_value=%s old_confidence=%s new_confidence=%s",
            user_id,
            key,
            existing.value if existing else "",
            value,
            existing.confidence if existing else 0,
            confidence,
        )
        result = asdict(existing) if existing else {}
        result["conflict_action"] = conflict_action
        return result

    if settings.postgres_enabled:
        _ensure_postgres_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO structured_memory (user_id, memory_key, memory_value, source, updated_at, status, confidence, expires_at)
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s,
                        to_timestamp(%s),
                        %s,
                        %s,
                        CASE WHEN %s::double precision IS NULL THEN NULL ELSE to_timestamp(%s::double precision) END
                    )
                    ON CONFLICT (user_id, memory_key)
                    DO UPDATE SET
                        memory_value = EXCLUDED.memory_value,
                        source = EXCLUDED.source,
                        updated_at = EXCLUDED.updated_at,
                        status = EXCLUDED.status,
                        confidence = EXCLUDED.confidence,
                        expires_at = EXCLUDED.expires_at
                    """,
                    (user_id, key, value, source, now, status, confidence, expires_at, expires_at),
                )
            conn.commit()
    else:
        with _lock:
            _structured[user_id][key] = StructuredMemory(
                key=key,
                value=value,
                source=source,
                updated_at=now,
                status=status,
                confidence=confidence,
                expires_at=expires_at,
            )
    logger.info(
        "Structured memory upserted. user_id=%s key=%s source=%s action=%s confidence=%s",
        user_id,
        key,
        source,
        conflict_action,
        confidence,
    )
    if status == "active" and not key.startswith("pending:"):
        _disable_pending_structured_memories(user_id, key)
    _record_structured_memory_history(
        user_id,
        key,
        existing.value if existing else None,
        value,
        conflict_action,
        source,
        confidence,
        status,
    )
    result = asdict(StructuredMemory(
        key=key,
        value=value,
        source=source,
        updated_at=now,
        status=status,
        confidence=confidence,
        expires_at=expires_at,
    ))
    result["conflict_action"] = conflict_action
    return result


def delete_structured_memory(user_id: str, key: str) -> None:
    set_structured_memory_status(user_id, key, "disabled")
    logger.info("Structured memory disabled. user_id=%s key=%s", user_id, key)


def set_structured_memory_status(user_id: str, key: str, status: str) -> dict[str, object]:
    status = _normalise_memory_status(status)
    now = time()
    existing = _get_structured_memory(user_id, key)
    if settings.postgres_enabled:
        _ensure_postgres_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE structured_memory
                    SET status = %s, updated_at = to_timestamp(%s)
                    WHERE user_id = %s AND memory_key = %s
                    RETURNING memory_key, memory_value, source, extract(epoch from updated_at), status, confidence,
                              extract(epoch from expires_at)
                    """,
                    (status, now, user_id, key),
                )
                row = cur.fetchone()
            conn.commit()
        if not row:
            return {}
        item = _structured_memory_from_row(row)
        _record_structured_memory_history(
            user_id,
            key,
            existing.value if existing else None,
            item.value,
            f"status:{status}",
            item.source,
            item.confidence,
            status,
        )
        return asdict(item)

    with _lock:
        item = _structured[user_id].get(key)
        if not item:
            return {}
        item.status = status
        item.updated_at = now
        _record_structured_memory_history(user_id, key, item.value, item.value, f"status:{status}", item.source, item.confidence, status)
        logger.info("Structured memory status changed. user_id=%s key=%s status=%s", user_id, key, status)
        return asdict(item)


def purge_structured_memory(user_id: str, key: str) -> None:
    existing = _get_structured_memory(user_id, key)
    if settings.postgres_enabled:
        _ensure_postgres_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM structured_memory WHERE user_id = %s AND memory_key = %s", (user_id, key))
            conn.commit()
    else:
        with _lock:
            _structured[user_id].pop(key, None)
    _record_structured_memory_history(
        user_id,
        key,
        existing.value if existing else None,
        None,
        "purged",
        existing.source if existing else "",
        existing.confidence if existing else 1.0,
        "purged",
    )
    logger.info("Structured memory purged. user_id=%s key=%s", user_id, key)


def _disable_pending_structured_memories(user_id: str, target_key: str) -> None:
    pending_suffix = f":{target_key}"
    now = time()
    if settings.postgres_enabled:
        _ensure_postgres_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE structured_memory
                    SET status = 'disabled', updated_at = to_timestamp(%s)
                    WHERE user_id = %s
                      AND status = 'pending'
                      AND memory_key LIKE 'pending:%%'
                      AND memory_key LIKE %s
                    """,
                    (now, user_id, f"%{pending_suffix}"),
                )
            conn.commit()
        return

    with _lock:
        for item in _structured.get(user_id, {}).values():
            if item.status == "pending" and item.key.startswith("pending:") and item.key.endswith(pending_suffix):
                item.status = "disabled"
                item.updated_at = now


def get_structured_memory_history(user_id: str, key: str, limit: int = 30) -> list[dict[str, object]]:
    if not settings.postgres_enabled:
        return []
    _ensure_postgres_schema()
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, memory_key, old_value, new_value, action, source, confidence, status, created_at
                FROM structured_memory_history
                WHERE user_id = %s AND memory_key = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (user_id, key, max(1, min(limit, 100))),
            )
            rows = cur.fetchall()
    return [
        {
            "id": row[0],
            "key": row[1],
            "oldValue": row[2],
            "newValue": row[3],
            "action": row[4],
            "source": row[5],
            "confidence": row[6],
            "status": row[7],
            "createdAt": row[8].isoformat(),
        }
        for row in rows
    ]


def get_structured_memory_merge_suggestions(user_id: str) -> list[dict[str, object]]:
    items = _load_all_structured(user_id)
    grouped: dict[str, list[StructuredMemory]] = defaultdict(list)
    for item in items:
        prefix = item.key.split(".", 1)[0].split("_", 1)[0].lower()
        grouped[prefix].append(item)

    suggestions: list[dict[str, object]] = []
    for group_key, group_items in grouped.items():
        active_items = [item for item in group_items if item.status == "active"]
        if len(group_items) < 2:
            continue
        values = {item.value for item in group_items}
        if len(values) <= 1 and len(active_items) <= 1:
            continue
        suggestions.append({
            "group": group_key,
            "reason": "同类 key 存在多个值，建议人工确认是否合并或禁用旧版本。",
            "items": [asdict(item) for item in sorted(group_items, key=lambda item: item.updated_at, reverse=True)],
        })
    return suggestions


def _load_short_term(user_id: str, conversation_id: str | None) -> list[MemoryTurn]:
    key = _conversation_key(user_id, conversation_id)
    if settings.redis_enabled:
        items = _redis_client().lrange(_short_term_key(key), 0, -1)
        return [MemoryTurn(**json.loads(item)) for item in items]

    with _lock:
        return list(_short_term.get(key, []))


def _save_short_term(user_id: str, conversation_id: str | None, turns: list[MemoryTurn]) -> None:
    key = _conversation_key(user_id, conversation_id)
    if settings.redis_enabled:
        client = _redis_client()
        redis_key = _short_term_key(key)
        client.delete(redis_key)
        if turns:
            client.rpush(redis_key, *[json.dumps(asdict(item), ensure_ascii=False) for item in turns])
        return

    with _lock:
        _short_term[key] = list(turns)


def _load_summary(user_id: str, conversation_id: str | None, allow_fallback: bool = True) -> str:
    return str(_load_summary_with_meta(user_id, conversation_id, allow_fallback=allow_fallback)["summary"])


def _load_summary_with_meta(user_id: str, conversation_id: str | None, allow_fallback: bool = True) -> dict[str, object]:
    key = _conversation_key(user_id, conversation_id)
    requested_conversation_id = conversation_id or "default"
    if settings.postgres_enabled:
        _ensure_postgres_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT summary, conversation_id
                    FROM memory_summary
                    WHERE user_id = %s AND conversation_id = %s
                    """,
                    (user_id, requested_conversation_id),
                )
                row = cur.fetchone()
                if row:
                    return {
                        "summary": row[0],
                        "requestedConversationId": requested_conversation_id,
                        "sourceConversationId": row[1],
                        "fallback": False,
                    }
                if not allow_fallback:
                    return {
                        "summary": "",
                        "requestedConversationId": requested_conversation_id,
                        "sourceConversationId": None,
                        "fallback": False,
                    }
                cur.execute(
                    """
                    SELECT summary, conversation_id
                    FROM memory_summary
                    WHERE user_id = %s
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    (user_id,),
                )
                fallback_row = cur.fetchone()
                return {
                    "summary": fallback_row[0] if fallback_row else "",
                    "requestedConversationId": requested_conversation_id,
                    "sourceConversationId": fallback_row[1] if fallback_row else None,
                    "fallback": bool(fallback_row),
                }

    with _lock:
        if key in _summaries:
            return {
                "summary": _summaries[key],
                "requestedConversationId": requested_conversation_id,
                "sourceConversationId": requested_conversation_id,
                "fallback": False,
            }
        if not allow_fallback:
            return {
                "summary": "",
                "requestedConversationId": requested_conversation_id,
                "sourceConversationId": None,
                "fallback": False,
            }
        user_prefix = f"{user_id}:"
        candidates = [
            (item_key.split(":", 1)[1], item)
            for item_key, item in _summaries.items()
            if item_key.startswith(user_prefix)
        ]
        source_conversation_id, summary = candidates[-1] if candidates else (None, "")
        return {
            "summary": summary,
            "requestedConversationId": requested_conversation_id,
            "sourceConversationId": source_conversation_id,
            "fallback": bool(candidates),
        }


def _save_summary(user_id: str, conversation_id: str | None, summary: str) -> None:
    key = _conversation_key(user_id, conversation_id)
    if settings.postgres_enabled:
        _ensure_postgres_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memory_summary (user_id, conversation_id, summary, updated_at)
                    VALUES (%s, %s, %s, now())
                    ON CONFLICT (user_id, conversation_id)
                    DO UPDATE SET summary = EXCLUDED.summary, updated_at = now()
                    """,
                    (user_id, conversation_id or "default", summary),
                )
            conn.commit()
        return

    with _lock:
        _summaries[key] = summary


def _load_structured(user_id: str, question: str) -> list[StructuredMemory]:
    if settings.postgres_enabled:
        _ensure_postgres_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT memory_key, memory_value, source, extract(epoch from updated_at), status, confidence,
                           extract(epoch from expires_at)
                    FROM structured_memory
                    WHERE user_id = %s
                      AND status = 'active'
                      AND (expires_at IS NULL OR expires_at > now())
                    ORDER BY updated_at DESC
                    """,
                    (user_id,),
                )
                items = [
                    _structured_memory_from_row(row)
                    for row in cur.fetchall()
                ]
                return _select_structured_memories(items, question)

    with _lock:
        structured = list(_structured.get(user_id, {}).values())

    return _select_structured_memories(structured, question)


def _load_all_structured(user_id: str) -> list[StructuredMemory]:
    if settings.postgres_enabled:
        _ensure_postgres_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT memory_key, memory_value, source, extract(epoch from updated_at), status, confidence,
                           extract(epoch from expires_at)
                    FROM structured_memory
                    WHERE user_id = %s
                    ORDER BY updated_at DESC
                    """,
                    (user_id,),
                )
                return [_structured_memory_from_row(row) for row in cur.fetchall()]

    with _lock:
        return sorted(
            list(_structured.get(user_id, {}).values()),
            key=lambda item: item.updated_at,
            reverse=True,
        )


def _get_structured_memory(user_id: str, key: str) -> StructuredMemory | None:
    if settings.postgres_enabled:
        _ensure_postgres_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT memory_key, memory_value, source, extract(epoch from updated_at), status, confidence,
                           extract(epoch from expires_at)
                    FROM structured_memory
                    WHERE user_id = %s AND memory_key = %s
                    """,
                    (user_id, key),
                )
                row = cur.fetchone()
                return _structured_memory_from_row(row) if row else None

    with _lock:
        return _structured.get(user_id, {}).get(key)


def _decide_structured_memory_conflict(
    existing: StructuredMemory | None,
    new_value: str,
    new_confidence: float,
    new_status: str,
) -> str:
    if existing is None:
        return "created"
    if existing.status != "active" and new_status == "active":
        return "reactivated"
    if existing.value == new_value:
        return "refreshed"
    if new_confidence + settings.structured_memory_conflict_confidence_margin < existing.confidence:
        return "ignored"
    return "replaced"


_MULTI_VALUE_STRUCTURED_KEYS = {
    "food_preference",
    "pet_preference",
    "sport_preference",
    "entertainment_preference",
    "technology_stack",
    "goal",
}


def _merge_structured_memory_value(key: str, existing_value: str, new_value: str, source: str) -> str:
    if key not in _MULTI_VALUE_STRUCTURED_KEYS:
        return new_value
    if _source_prefers_replacement(source):
        return new_value
    existing_items = _split_structured_multi_value(existing_value)
    new_items = _split_structured_multi_value(new_value)
    merged = list(existing_items)
    for item in new_items:
        if item and item not in merged:
            merged.append(item)
    return "、".join(merged) if merged else new_value


def _source_prefers_replacement(source: str) -> bool:
    return any(signal in source for signal in ("不喜欢", "改成", "换成", "相比于", "更喜欢", "现在喜欢", "不再"))


def _split_structured_multi_value(value: str) -> list[str]:
    return [
        item.strip()
        for item in re.split(r"[、,，;；/和与]+", value or "")
        if item.strip()
    ]


def _upsert_qdrant_memory(user_id: str, conversation_id: str | None, text: str) -> None:
    if not settings.qdrant_enabled or not _vector_backend_available("qdrant_memory"):
        return

    global _qdrant_memory_ready
    _ensure_qdrant_memory_collection()
    point_id = str(uuid5(NAMESPACE_URL, f"memory:qdrant:{user_id}:{conversation_id}:{text}"))
    body = {
        "points": [
            {
                "id": point_id,
                "vector": embed_text(text),
                "payload": {
                    "user_id": user_id,
                    "conversation_id": conversation_id or "default",
                    "text": text,
                    "created_at": time(),
                },
            }
        ]
    }
    with httpx.Client(timeout=settings.vector_backend_timeout_seconds) as client:
        client.put(
            f"{settings.qdrant_endpoint.rstrip('/')}/collections/{settings.qdrant_memory_collection}/points",
            params={"wait": "true"},
            json=body,
        ).raise_for_status()
    logger.info("Qdrant memory upserted. user_id=%s conversation_id=%s", user_id, conversation_id)


def _search_qdrant_memory(user_id: str, question: str, top_k: int) -> list[dict[str, object]]:
    if not settings.qdrant_enabled or not _vector_backend_available("qdrant_memory"):
        return []

    _ensure_qdrant_memory_collection()
    body = {
        "vector": embed_text(question),
        "limit": top_k,
        "with_payload": True,
        "filter": {
            "must": [
                {"key": "user_id", "match": {"value": user_id}},
            ]
        },
    }
    with httpx.Client(timeout=settings.vector_backend_timeout_seconds) as client:
        response = client.post(
            f"{settings.qdrant_endpoint.rstrip('/')}/collections/{settings.qdrant_memory_collection}/points/search",
            json=body,
        )
        response.raise_for_status()
        points = response.json().get("result") or []

    return [
        {
            "text": (point.get("payload") or {}).get("text", ""),
            "score": point.get("score", 0),
            "backend": "qdrant",
        }
        for point in points
    ]


def _upsert_milvus_memory(user_id: str, conversation_id: str | None, text: str) -> None:
    if not settings.milvus_enabled or not _vector_backend_available("milvus_memory"):
        return

    _ensure_milvus_collection()
    client = _milvus_client()
    client.upsert(
        collection_name=settings.milvus_collection,
        data=[{
            "id": str(uuid5(NAMESPACE_URL, f"memory:milvus:{user_id}:{conversation_id}:{text}")),
            "user_id": user_id,
            "conversation_id": conversation_id or "default",
            "text": text,
            "vector": embed_text(text),
            "created_at": time(),
        }],
    )
    logger.info("Milvus memory upserted. user_id=%s conversation_id=%s", user_id, conversation_id)


def _search_milvus_memory(user_id: str, question: str, top_k: int) -> list[dict[str, object]]:
    if not settings.milvus_enabled or not _vector_backend_available("milvus_memory"):
        return []

    _ensure_milvus_collection()
    client = _milvus_client()
    _load_milvus_collection(client)
    result = client.search(
        collection_name=settings.milvus_collection,
        data=[embed_text(question)],
        filter=f'user_id == "{user_id}"',
        limit=top_k,
        output_fields=["text", "conversation_id", "created_at"],
    )
    if not result:
        return []

    return [
        {
            "text": item["entity"].get("text", ""),
            "score": item.get("distance", 0),
            "backend": "milvus",
        }
        for item in result[0]
    ]


def _ensure_postgres_schema() -> None:
    global _postgres_ready
    if _postgres_ready:
        return

    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_summary (
                  user_id VARCHAR(128) NOT NULL,
                  conversation_id VARCHAR(128) NOT NULL,
                  summary TEXT NOT NULL,
                  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  PRIMARY KEY (user_id, conversation_id)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS structured_memory (
                  user_id VARCHAR(128) NOT NULL,
                  memory_key VARCHAR(128) NOT NULL,
                  memory_value TEXT NOT NULL,
                  source TEXT NOT NULL,
                  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  PRIMARY KEY (user_id, memory_key)
                )
                """
            )
            cur.execute("ALTER TABLE structured_memory ADD COLUMN IF NOT EXISTS status VARCHAR(32) NOT NULL DEFAULT 'active'")
            cur.execute("ALTER TABLE structured_memory ADD COLUMN IF NOT EXISTS confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0")
            cur.execute("ALTER TABLE structured_memory ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS structured_memory_history (
                  id BIGSERIAL PRIMARY KEY,
                  user_id VARCHAR(128) NOT NULL,
                  memory_key VARCHAR(128) NOT NULL,
                  old_value TEXT,
                  new_value TEXT,
                  action VARCHAR(32) NOT NULL,
                  source TEXT NOT NULL DEFAULT '',
                  confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                  status VARCHAR(32) NOT NULL DEFAULT 'active',
                  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_structured_memory_history_user_key ON structured_memory_history(user_id, memory_key, created_at DESC)"
            )
        conn.commit()
    _postgres_ready = True


def _record_structured_memory_history(
    user_id: str,
    key: str,
    old_value: str | None,
    new_value: str | None,
    action: str,
    source: str,
    confidence: float,
    status: str,
) -> None:
    if not settings.postgres_enabled:
        return
    try:
        _ensure_postgres_schema()
        with postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO structured_memory_history (
                      user_id, memory_key, old_value, new_value, action, source, confidence, status
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (user_id, key, old_value, new_value, action[:32], source, confidence, status),
                )
            conn.commit()
    except Exception:
        logger.exception("Structured memory history write failed. user_id=%s key=%s action=%s", user_id, key, action)


def _ensure_qdrant_memory_collection() -> None:
    global _qdrant_memory_ready
    if _qdrant_memory_ready:
        return

    with httpx.Client(timeout=settings.vector_backend_timeout_seconds) as client:
        response = client.get(
            f"{settings.qdrant_endpoint.rstrip('/')}/collections/{settings.qdrant_memory_collection}"
        )
        if response.status_code == 200:
            data = response.json().get("result") or {}
            existing_size = _qdrant_vector_size(data)
            if existing_size == EMBEDDING_DIMENSION:
                _qdrant_memory_ready = True
                return
            if not settings.recreate_vector_collections_on_dimension_mismatch:
                raise ValueError(
                    f"Qdrant collection {settings.qdrant_memory_collection} has vector size "
                    f"{existing_size}, expected {EMBEDDING_DIMENSION}."
                )
            client.delete(
                f"{settings.qdrant_endpoint.rstrip('/')}/collections/{settings.qdrant_memory_collection}"
            ).raise_for_status()

        if response.status_code == 404 or response.status_code == 200:
            client.put(
                f"{settings.qdrant_endpoint.rstrip('/')}/collections/{settings.qdrant_memory_collection}",
                json={"vectors": {"size": EMBEDDING_DIMENSION, "distance": "Cosine"}},
            ).raise_for_status()
        elif response.status_code != 200:
            response.raise_for_status()
    _qdrant_memory_ready = True


def _ensure_milvus_collection() -> None:
    global _milvus_ready
    if _milvus_ready:
        return

    from pymilvus import DataType

    client = _milvus_client()
    if client.has_collection(settings.milvus_collection):
        if settings.recreate_vector_collections_on_dimension_mismatch:
            try:
                description = client.describe_collection(settings.milvus_collection)
                fields = description.get("fields") or []
                vector_field = next((field for field in fields if field.get("name") == "vector"), None)
                params = vector_field.get("params") if vector_field else {}
                existing_dim = int(params.get("dim", EMBEDDING_DIMENSION))
                if existing_dim != EMBEDDING_DIMENSION:
                    client.drop_collection(settings.milvus_collection)
                else:
                    _mark_milvus_ready(client)
                    return
            except Exception:
                client.drop_collection(settings.milvus_collection)
        else:
            _mark_milvus_ready(client)
            return

    if client.has_collection(settings.milvus_collection):
        _mark_milvus_ready(client)
        return

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("user_id", DataType.VARCHAR, max_length=128)
    schema.add_field("conversation_id", DataType.VARCHAR, max_length=128)
    schema.add_field("text", DataType.VARCHAR, max_length=4096)
    schema.add_field("created_at", DataType.DOUBLE)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMENSION)
    index_params = client.prepare_index_params()
    index_params.add_index("vector", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_collection(
        collection_name=settings.milvus_collection,
        schema=schema,
        index_params=index_params,
    )
    _mark_milvus_ready(client)


def _mark_milvus_ready(client) -> None:
    global _milvus_ready
    _load_milvus_collection(client)
    _milvus_ready = True


def _load_milvus_collection(client) -> None:
    client.load_collection(collection_name=settings.milvus_collection)


def _vector_backend_available(name: str) -> bool:
    return _vector_backend_disabled_until.get(name, 0) <= time()


def _mark_vector_backend_failure(name: str, exception: Exception) -> None:
    cooldown = max(settings.vector_backend_failure_cooldown_seconds, 0)
    if cooldown <= 0:
        return
    _vector_backend_disabled_until[name] = time() + cooldown
    logger.warning(
        "Vector backend temporarily disabled. backend=%s cooldown_seconds=%s error=%s",
        name,
        cooldown,
        exception,
    )


def _redis_client() -> redis.Redis:
    return redis.Redis.from_url(settings.redis_url, decode_responses=True)


def _milvus_client():
    from pymilvus import MilvusClient

    if settings.milvus_mode == "lite":
        return MilvusClient(settings.milvus_lite_path)
    if settings.milvus_token:
        return MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token)
    return MilvusClient(uri=settings.milvus_uri)


def _extract_structured_candidates(text: str, recent_turns: list[MemoryTurn] | None = None) -> list[tuple[str, str, float]]:
    if _is_question_like_text(text):
        return []
    if _is_task_commitment_text(text):
        return []

    mode = settings.structured_memory_extractor.lower()
    if mode not in {"rule", "llm", "hybrid"}:
        mode = "hybrid"

    candidates: list[tuple[str, str, float]] = []
    if mode in {"rule", "hybrid"}:
        candidates.extend(_extract_structured_candidates_by_rule(text))

    if mode in {"llm", "hybrid"}:
        candidates.extend(_extract_structured_candidates_by_llm(text, recent_turns or []))

    deduped: dict[str, tuple[str, float]] = {}
    for key, value, confidence in candidates:
        normalised = _normalise_structured_candidate(key, value)
        if normalised:
            normalised_key, normalised_value = normalised
            existing = deduped.get(normalised_key)
            if existing is None or confidence >= existing[1]:
                deduped[normalised_key] = (normalised_value, confidence)
    return [(key, value, confidence) for key, (value, confidence) in deduped.items()]


def _extract_structured_candidates_by_rule(text: str) -> list[tuple[str, str, float]]:
    candidates: list[tuple[str, str, float]] = []
    if _is_question_like_text(text):
        return []

    intro_match = re.search(r"我是([^，。,.!\n]{1,16})喜欢([^，。,.!\n]{1,64})", text)
    if intro_match:
        candidates.append(("name", intro_match.group(1).strip(), 0.85))
        value = intro_match.group(2).strip()
        candidates.append((_preference_key_for_value(value), value, 0.85))

    career_patterns = [
        r"我的(?:职业|工作|岗位)是([^，。,.!\n]{1,64})",
        r"我是(?:一个|一名)?([^，。,.!\n]{1,32}(?:工程师|程序员|开发|架构师|产品经理|设计师|运营|学生|教师|医生|律师))",
    ]
    for pattern in career_patterns:
        for match in re.finditer(pattern, text):
            candidates.append(("career", match.group(1).strip(), 0.9))

    zodiac_patterns = [
        r"我的星座是([^，。,.!\n]{1,16})",
        r"我是([^，。,.!\n]{1,12}?)(?:星座|座)",
    ]
    for pattern in zodiac_patterns:
        for match in re.finditer(pattern, text):
            candidates.append(("zodiac", _normalise_zodiac_value(match.group(1)), 0.9))

    patterns = [
        (r"我叫([^，。,.!\n]{1,32})", "name"),
        (r"我是([^，。,.!\n]{1,32})", "identity"),
        (r"我的([^，。,.!\n]{1,16})是([^，。,.!\n]{1,64})", None),
        (r"我喜欢([^，。,.!\n]{1,64})", "preference"),
        (r"我更喜欢([^，。,.!\n]{1,64})", "preference"),
        (r"相比于[^，。,.!\n]{1,64}我[更跟]?喜欢([^，。,.!\n]{1,64})", "preference"),
        (r"我想(?:要)?去([^，。,.!\n]{1,32})(?:旅行|旅游|玩|看看|逛逛)?", "travel_interest"),
    ]
    patterns.extend([
        (r"我正在做([^，。,.!\n]{1,64})", "current_project"),
        (r"我想(?:要)?(?:做|学习|尝试|开始)([^，。,.!\n]{1,64})", "goal"),
    ])

    for pattern, fixed_key in patterns:
        for match in re.finditer(pattern, text):
            if fixed_key is None:
                candidates.append((_normalise_structured_key(match.group(1)), match.group(2).strip(), 0.8))
            elif fixed_key == "identity" and _looks_like_mixed_identity(match.group(1)):
                continue
            elif fixed_key == "preference":
                value = match.group(1).strip()
                candidates.append((_preference_key_for_value(value), value, 0.85))
            else:
                candidates.append((fixed_key, match.group(1).strip(), 0.85))
    return [
        (key, value, confidence)
        for key, value, confidence in candidates
        if _is_valid_structured_candidate(key, value)
    ]


def _extract_structured_candidates_by_llm(text: str, recent_turns: list[MemoryTurn]) -> list[tuple[str, str, float]]:
    recent = recent_turns[-settings.structured_memory_recent_turns:]
    recent_text = "\n".join(f"{item.role}: {item.content}" for item in recent)
    try:
        memories = invoke_structured_memory_chain(text, recent_text)
    except Exception:
        logger.exception("LLM structured memory extraction failed.")
        return []

    candidates: list[tuple[str, str, float]] = []
    for item in memories:
        confidence = float(item.get("confidence") or 0)
        if confidence < settings.structured_memory_confidence_threshold:
            continue
        key = str(item.get("key") or "").strip()
        value = str(item.get("value") or "").strip()
        normalised = _normalise_structured_candidate(key, value)
        if normalised:
            candidates.append((*normalised, confidence))
    return candidates


def _is_question_like_text(text: str) -> bool:
    return bool(_QUESTION_LIKE_PATTERN.search(text))


def _is_task_commitment_text(text: str) -> bool:
    return bool(_task_commitment_goal(text))


def _task_commitment_goal(text: str) -> str:
    match = re.search(
        r"((?:这周|本周|今天|明天|后天|下周[一二三四五六日天]?|周[一二三四五六日天]?)?"
        r"(?:我)?(?:要|想要|需要|打算|计划)(?:完成|做完|搞定|处理|推进|写完|整理|提交)[^，。,.!\n]{2,80})",
        text,
    )
    return match.group(1).strip() if match else ""


def _is_valid_structured_candidate(key: str, value: str) -> bool:
    cleaned_value = value.strip(" \t\r\n，。,.!！?？")
    return bool(key.strip() and cleaned_value and cleaned_value not in _INVALID_STRUCTURED_VALUES)


def _normalise_structured_candidate(key: str, value: str) -> tuple[str, str] | None:
    normalised_key = _normalise_structured_key(key)
    normalised_value = value.strip()
    if not _is_valid_structured_candidate(normalised_key, normalised_value):
        return None
    if normalised_key == "identity":
        if _looks_like_zodiac_value(normalised_value):
            return "zodiac", _normalise_zodiac_value(normalised_value)
        if _looks_like_mixed_identity(normalised_value):
            return None
    if normalised_key == "preference":
        normalised_key = _preference_key_for_value(normalised_value)
    if normalised_key == "zodiac":
        return normalised_key, _normalise_zodiac_value(normalised_value)
    return normalised_key, normalised_value


def _preference_key_for_value(value: str) -> str:
    cleaned = value.strip()
    if _contains_any(cleaned, _FOOD_PREFERENCE_TERMS):
        return "food_preference"
    if _contains_any(cleaned, _PET_PREFERENCE_TERMS):
        return "pet_preference"
    if _contains_any(cleaned, _SPORT_PREFERENCE_TERMS):
        return "sport_preference"
    if _contains_any(cleaned, _ENTERTAINMENT_PREFERENCE_TERMS):
        return "entertainment_preference"
    return "preference"


def _contains_any(value: str, terms: set[str]) -> bool:
    return any(term in value for term in terms)


def _looks_like_mixed_identity(value: str) -> bool:
    return any(marker in value for marker in ["喜欢", "不喜欢", "爱", "讨厌", "敲代码", "正在做", "星座", "座"])


def _looks_like_zodiac_value(value: str) -> bool:
    cleaned = value.strip()
    zodiac_names = ["白羊", "金牛", "双子", "巨蟹", "狮子", "处女", "天秤", "天蝎", "射手", "摩羯", "水瓶", "双鱼"]
    return any(name in cleaned for name in zodiac_names) and ("座" in cleaned or "星座" in cleaned)


def _normalise_zodiac_value(value: str) -> str:
    cleaned = value.strip()
    if cleaned.endswith("星座"):
        cleaned = cleaned[:-2]
    if not cleaned.endswith("座"):
        cleaned = f"{cleaned}座"
    return cleaned


def _normalise_structured_key(key: str) -> str:
    cleaned = key.strip()
    return {
        "职业": "career",
        "工作": "career",
        "岗位": "career",
        "名字": "name",
        "姓名": "name",
        "星座": "zodiac",
        "项目": "current_project",
        "目标": "goal",
        "偏好": "preference",
        "喜好": "preference",
        "食物偏好": "food_preference",
        "水果偏好": "food_preference",
        "宠物偏好": "pet_preference",
        "动物偏好": "pet_preference",
        "运动偏好": "sport_preference",
        "娱乐偏好": "entertainment_preference",
    }.get(cleaned, cleaned)


def _structured_memory_from_row(row: tuple) -> StructuredMemory:
    expires_at = float(row[6]) if row[6] is not None else None
    return StructuredMemory(
        key=row[0],
        value=row[1],
        source=row[2],
        updated_at=float(row[3]),
        status=row[4] or "active",
        confidence=float(row[5] or 1.0),
        expires_at=expires_at,
    )


def _normalise_memory_status(status: str) -> str:
    value = (status or "active").strip().lower()
    return value if value in {"active", "pending", "disabled", "expired"} else "active"


def _pending_structured_memory_key(
    user_id: str,
    conversation_id: str | None,
    text: str,
    key: str,
    value: str,
) -> str:
    candidate_id = uuid5(NAMESPACE_URL, f"memory:pending:{user_id}:{conversation_id}:{text}:{key}:{value}")
    return f"pending:{candidate_id}:{key}"


def _select_structured_memories(items: list[StructuredMemory], question: str) -> list[StructuredMemory]:
    active = sorted(
        [
            item for item in items
            if item.status == "active"
            and (item.expires_at is None or item.expires_at > time())
        ],
        key=lambda item: item.updated_at,
        reverse=True,
    )
    if not question.strip():
        return active[:settings.memory_recall_limit]

    relevant = [
        item for item in active
        if item.key in question or item.value in question
    ]
    return (relevant or active)[:settings.memory_recall_limit]


def _qdrant_vector_size(collection_info: dict) -> int | None:
    vectors = (collection_info.get("config") or {}).get("params", {}).get("vectors")
    if isinstance(vectors, dict) and "size" in vectors:
        return int(vectors["size"])
    if isinstance(vectors, dict):
        for value in vectors.values():
            if isinstance(value, dict) and "size" in value:
                return int(value["size"])
    return None


def _conversation_key(user_id: str, conversation_id: str | None) -> str:
    return f"{user_id}:{conversation_id or 'default'}"


def _short_term_key(key: str) -> str:
    return f"memory:short:{key}"


def _compression_trigger_limit() -> int:
    return max(settings.memory_compression_trigger_limit, settings.memory_short_term_limit)


def _short_term_buffer_limit() -> int:
    return max(_compression_trigger_limit() + 2, settings.memory_short_term_limit)
