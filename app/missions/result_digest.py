from __future__ import annotations

from app.missions.schemas import MissionDigestRequest, MissionDigestResponse


def build_mission_digest(request: MissionDigestRequest) -> MissionDigestResponse:
    completed = sum(
        1 for run in request.runs
        if str(run.get("status") or "") in {"COMPLETED", "PARTIALLY_COMPLETED"}
    )
    open_exceptions = [
        item for item in request.exceptions
        if str(item.get("status") or "") in {"OPEN", "ACKNOWLEDGED"}
    ]
    evidence = [
        str(ref)
        for run in request.runs
        for ref in (run.get("evidenceRefs") if isinstance(run.get("evidenceRefs"), list) else [])
    ][:100]
    summary = (
        f"本周期完成 {completed} 次持续运行；"
        f"{len(open_exceptions)} 项异常仍需处理。"
        + (" 正常结果已合并到摘要。" if completed else " 暂无可合并的完成结果。")
    )
    return MissionDigestResponse(
        summary=summary,
        completedRuns=completed,
        attentionCount=len(open_exceptions),
        evidenceRefs=evidence,
    )
