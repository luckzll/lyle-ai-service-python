from fastapi import APIRouter

from app.plans.plan_store import create_plan, get_plan, list_plans, transition_plan

router = APIRouter()


@router.get("/plans/{user_id}")
def plans(
    user_id: str,
    projectId: int | None = None,
    status: str | None = None,
    limit: int = 100,
) -> dict[str, object]:
    return {"items": list_plans(user_id, project_id=projectId, status=status, limit=limit)}


@router.post("/plans/{user_id}")
def plan_create(user_id: str, request: dict[str, object]) -> dict[str, object]:
    return create_plan(user_id, request)


@router.get("/plans/{user_id}/{plan_id}")
def plan_detail(user_id: str, plan_id: int) -> dict[str, object]:
    return get_plan(user_id, plan_id)


@router.post("/plans/{user_id}/{plan_id}/confirm")
def plan_confirm(user_id: str, plan_id: int) -> dict[str, object]:
    return transition_plan(user_id, plan_id, "confirm")


@router.post("/plans/{user_id}/{plan_id}/pause")
def plan_pause(user_id: str, plan_id: int) -> dict[str, object]:
    return transition_plan(user_id, plan_id, "pause")


@router.post("/plans/{user_id}/{plan_id}/resume")
def plan_resume(user_id: str, plan_id: int) -> dict[str, object]:
    return transition_plan(user_id, plan_id, "resume")


@router.post("/plans/{user_id}/{plan_id}/cancel")
def plan_cancel(user_id: str, plan_id: int) -> dict[str, object]:
    return transition_plan(user_id, plan_id, "cancel")
