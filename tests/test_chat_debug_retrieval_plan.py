import json

from app.security.chat_debug import build_safe_chat_debug


def debug_plan(plan):
    return build_safe_chat_debug(
        knowledge_base_ids=["kb"], contexts=[],
        agent_trace=[{"agent": "rag_agent", "retrievalPlan": plan}],
        tool_results=[], review={}, usage={}, used_memories=[], memory_exclusions={},
    )


def test_retrieval_plan_keeps_only_public_metrics_and_short_query_resolution():
    raw = {
        "mode": "optimized", "fusionMode": "legacy", "actualQueryCount": 2,
        "candidateCount": 23, "selectedCount": 20, "parallelWorkers": 2, "adaptiveUsed": False,
        "candidateCache": {"hit": 1, "miss": 1, "shared": 0, "bypass": 0,
                           "cacheKey": "PRIVATE_CACHE_KEY", "documents": ["PRIVATE_DOCUMENT"]},
        "queryResolution": {"method": "rule", "reason": "explicit_reference",
                            "originalQuestion": "具体改了什么？", "effectiveQuestion": "星桥 2.4.1 具体改了什么？",
                            "history": "PRIVATE_HISTORY", "historyFingerprint": "PRIVATE_HISTORY_KEY",
                            "user_id": "PRIVATE_USER", "conversation_id": "PRIVATE_CONVERSATION"},
        "documentManifest": "PRIVATE_MANIFEST", "modelPayload": "PRIVATE_MODEL_PAYLOAD",
    }
    payload = debug_plan(raw)
    plan = payload["agentTrace"][0]["retrievalPlan"]
    assert plan == {
        "mode": "optimized", "fusionMode": "legacy", "actualQueryCount": 2,
        "candidateCount": 23, "rerankSelectedCount": 20, "parallelWorkers": 2, "adaptiveUsed": False,
        "candidateCache": {"hit": 1, "miss": 1, "shared": 0, "bypass": 0},
        "queryResolution": {"method": "rule", "reason": "explicit_reference",
                            "originalQuestion": "具体改了什么？", "effectiveQuestion": "星桥 2.4.1 具体改了什么？"},
    }
    assert "PRIVATE_" not in json.dumps(payload)
    assert "rerankSelectedCount" not in raw  # Sanitizing must not mutate the runtime plan.


def test_plan_nested_payloads_secrets_and_long_text_cannot_escape_the_debug_budget():
    payload = debug_plan({
        "mode": {"rawPayload": "PRIVATE_MODE"}, "fusionMode": ["PRIVATE_FUSION"],
        "queryResolution": {
            "method": "r" * 500, "reason": "x" * 500,
            "originalQuestion": "Authorization: Bearer secret-provider-token " + "问" * 900,
            "effectiveQuestion": {"history": "PRIVATE_NESTED_HISTORY"},
        },
    })
    plan = payload["agentTrace"][0]["retrievalPlan"]
    resolution = plan["queryResolution"]
    assert plan["mode"] == plan["fusionMode"] == resolution["effectiveQuestion"] == ""
    assert len(resolution["method"]) == 64 and len(resolution["reason"]) == 160
    assert len(resolution["originalQuestion"]) <= 500
    serialized = json.dumps(payload)
    assert "secret-provider-token" not in serialized and "PRIVATE_" not in serialized
    assert payload["debugTruncated"] is True


def test_plan_counts_are_non_negative_finite_and_do_not_coerce_booleans():
    payload = debug_plan({
        "actualQueryCount": float("inf"), "candidateCount": -1,
        "rerankSelectedCount": "3", "selectedCount": 20, "parallelWorkers": True,
        "adaptiveUsed": "false",
        "candidateCache": {"hit": float("nan"), "miss": -4, "shared": True, "bypass": "2"},
    })
    assert payload["agentTrace"][0]["retrievalPlan"] == {
        "candidateCount": 0, "rerankSelectedCount": 3, "candidateCache": {"miss": 0, "bypass": 2},
    }
    json.dumps(payload, allow_nan=False)


def test_non_mapping_retrieval_plan_is_dropped():
    payload = debug_plan([{"history": "PRIVATE_HISTORY"}])
    assert "retrievalPlan" not in payload["agentTrace"][0]
