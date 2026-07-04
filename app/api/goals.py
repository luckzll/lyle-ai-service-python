from fastapi import APIRouter

from app.goals.goal_store import (
    create_goal,
    create_milestone,
    get_goal,
    link_task_to_goal,
    list_goals,
    list_milestones,
    update_goal,
)

router = APIRouter()


@router.get("/goals/{user_id}")
def goals(
    user_id: str,
    projectId: int | None = None,
    status: str | None = "active",
    limit: int = 100,
) -> dict[str, object]:
    return {"items": list_goals(user_id, project_id=projectId, status=status, limit=limit)}


@router.post("/goals/{user_id}")
def goal_create(user_id: str, request: dict[str, object]) -> dict[str, object]:
    return create_goal(user_id, request)


@router.get("/goals/{user_id}/{goal_id}")
def goal_detail(user_id: str, goal_id: int) -> dict[str, object]:
    return get_goal(user_id, goal_id)


@router.patch("/goals/{user_id}/{goal_id}")
def goal_update(user_id: str, goal_id: int, request: dict[str, object]) -> dict[str, object]:
    return update_goal(user_id, goal_id, request)


@router.get("/goals/{user_id}/{goal_id}/milestones")
def milestones(user_id: str, goal_id: int) -> dict[str, object]:
    return {"items": list_milestones(user_id, goal_id)}


@router.post("/goals/{user_id}/{goal_id}/milestones")
def milestone_create(user_id: str, goal_id: int, request: dict[str, object]) -> dict[str, object]:
    return create_milestone(user_id, goal_id, request)


@router.post("/goals/{user_id}/{goal_id}/tasks/{task_id}/link")
def goal_task_link(user_id: str, goal_id: int, task_id: str, request: dict[str, object] | None = None) -> dict[str, object]:
    return link_task_to_goal(user_id, goal_id, task_id, request)
