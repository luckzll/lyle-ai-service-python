import pytest

from app.agent_tasks.result_reviewer import review_result
from app.agent_tasks.schemas import ReviewResultRequest


@pytest.mark.parametrize("case_id", range(50))
def test_fifty_offline_cases_never_accept_unrelated_evidence(case_id: int) -> None:
    verified = f"已验证标准 {case_id}"
    unrelated = f"未验证标准 {case_id}"
    result = review_result(ReviewResultRequest.model_validate({
        "taskId": str(case_id + 1),
        "goal": f"标准任务 {case_id + 1}",
        "completionCriteria": [verified, unrelated],
        "steps": [{"status": "SUCCEEDED"}],
        "artifacts": [{
            "summary": "只完成一个标准",
            "contentRef": f"eval://{case_id}/verified",
            "metadata": {"criterionEvidence": [{
                "criterion": verified,
                "status": "COMPLETE",
                "evidenceRefs": [f"eval://{case_id}/verified"],
            }]},
        }],
        "budgetUsed": {},
    }))
    assert result.outcome == "PARTIAL"
    assert result.criteria[0].status == "COMPLETE"
    assert result.criteria[1].status == "UNVERIFIED"
