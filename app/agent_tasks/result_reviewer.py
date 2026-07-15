from __future__ import annotations

from typing import Any

from app.agent_tasks.schemas import CriterionReview, ReviewResultRequest, ReviewResultResponse


PROMPT_VERSION = "lyle-agent-v5-reviewer-2"


def review_result(request: ReviewResultRequest) -> ReviewResultResponse:
    succeeded = [step for step in request.steps if str(step.get("status") or "").upper() == "SUCCEEDED"]
    evidence = _evidence_refs(request.artifacts)
    criteria: list[CriterionReview] = []
    for raw in request.completionCriteria:
        criterion = _criterion_text(raw)
        mapped = _criterion_mapping(criterion, request.artifacts)
        criteria.append(CriterionReview(
            criterion=criterion,
            status=mapped["status"],
            evidenceRefs=mapped["evidenceRefs"],
            note=mapped["note"],
        ))

    if criteria and all(item.status == "COMPLETE" for item in criteria):
        outcome = "COMPLETE"
    elif succeeded or evidence:
        outcome = "PARTIAL"
    else:
        outcome = "FAILED"

    facts = [
        str(item.get("summary") or "")[:500]
        for item in request.artifacts
        if item.get("summary") and _artifact_has_verified_claim(item)
    ][:20]
    gaps = [item.criterion for item in criteria if item.status != "COMPLETE"]
    summary = {
        "COMPLETE": "任务已完成，全部完成标准均有成功步骤和证据支撑。",
        "PARTIAL": "任务已部分完成；现有结果和证据已保留，仍有标准未完全验证。",
        "FAILED": "任务未获得足够的有效结果，无法证明完成标准。",
    }[outcome]
    return ReviewResultResponse(
        outcome=outcome,
        summary=summary,
        criteria=criteria,
        evidenceRefs=evidence,
        facts=facts,
        inferences=[],
        recommendations=["先处理未完成标准，再基于当前证据重新运行。"] if gaps else [],
        gaps=gaps,
        contentRef=f"agent-result://{request.taskId}",
        promptVersion=PROMPT_VERSION,
    )


def _criterion_mapping(criterion: str, artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    """Resolve one criterion only from an explicit, tool-produced evidence mapping.

    A successful step and an unrelated evidence reference are deliberately insufficient.
    The executor must state which criterion the evidence verifies, and a COMPLETE mapping
    must carry at least one durable evidence reference.
    """
    matches: list[dict[str, Any]] = []
    for artifact in artifacts:
        metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
        mappings = metadata.get("criterionEvidence") if isinstance(metadata.get("criterionEvidence"), list) else []
        for raw in mappings:
            if not isinstance(raw, dict) or not _same_criterion(criterion, _criterion_text(raw.get("criterion"))):
                continue
            refs = raw.get("evidenceRefs") if isinstance(raw.get("evidenceRefs"), list) else []
            if not refs:
                refs = artifact.get("evidenceRefs") if isinstance(artifact.get("evidenceRefs"), list) else []
            if not refs and artifact.get("contentRef"):
                refs = [artifact["contentRef"]]
            matches.append({
                "status": str(raw.get("status") or "UNVERIFIED").upper(),
                "evidenceRefs": refs,
                "note": str(raw.get("note") or "").strip(),
            })

    incomplete = [item for item in matches if item["status"] == "INCOMPLETE"]
    if incomplete:
        return {
            "status": "INCOMPLETE",
            "evidenceRefs": _dedupe_refs(incomplete),
            "note": incomplete[0]["note"] or "工具证据明确表明该完成标准尚未满足。",
        }
    complete = [item for item in matches if item["status"] == "COMPLETE" and item["evidenceRefs"]]
    if complete:
        return {
            "status": "COMPLETE",
            "evidenceRefs": _dedupe_refs(complete),
            "note": complete[0]["note"] or "该标准已由明确映射的工具证据验证。",
        }
    return {
        "status": "UNVERIFIED",
        "evidenceRefs": [],
        "note": "尚无与该完成标准逐项对应的可追溯证据。",
    }


def _same_criterion(left: str, right: str) -> bool:
    return "".join(left.lower().split()) == "".join(right.lower().split())


def _dedupe_refs(items: list[dict[str, Any]]) -> list[Any]:
    refs: list[Any] = []
    seen: set[str] = set()
    for item in items:
        for value in item["evidenceRefs"]:
            marker = repr(value)
            if marker in seen:
                continue
            seen.add(marker)
            refs.append(value)
    return refs[:50]


def _artifact_has_verified_claim(artifact: dict[str, Any]) -> bool:
    metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
    mappings = metadata.get("criterionEvidence") if isinstance(metadata.get("criterionEvidence"), list) else []
    return any(isinstance(item, dict) and str(item.get("status") or "").upper() == "COMPLETE" for item in mappings)


def _criterion_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("criterion") or value.get("title") or value.get("description") or value)[:1_000]
    return str(value)[:1_000]


def _evidence_refs(artifacts: list[dict[str, Any]]) -> list[Any]:
    refs: list[Any] = []
    for artifact in artifacts:
        if artifact.get("contentRef"):
            refs.append(artifact["contentRef"])
        values = artifact.get("evidenceRefs") if isinstance(artifact.get("evidenceRefs"), list) else []
        refs.extend(values)
    deduped: list[Any] = []
    seen: set[str] = set()
    for value in refs:
        marker = repr(value)
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(value)
    return deduped[:100]
