from fastapi import APIRouter, Response

from app.projects.project_store import (
    create_project_decision,
    create_project,
    create_project_link,
    create_project_open_question,
    delete_project_link,
    get_project,
    get_project_overview,
    list_project_links,
    list_projects,
    update_project_open_question,
    update_project,
)

router = APIRouter()


@router.get("/projects/{user_id}")
def projects(user_id: str, status: str | None = "active", limit: int = 100) -> dict[str, object]:
    return {"items": list_projects(user_id, status=status, limit=limit)}


@router.post("/projects/{user_id}")
def project_create(user_id: str, request: dict[str, object]) -> dict[str, object]:
    return create_project(user_id, request)


@router.get("/projects/{user_id}/{project_id}")
def project_detail(user_id: str, project_id: int) -> dict[str, object]:
    return get_project(user_id, project_id)


@router.patch("/projects/{user_id}/{project_id}")
def project_update(user_id: str, project_id: int, request: dict[str, object]) -> dict[str, object]:
    return update_project(user_id, project_id, request)


@router.get("/projects/{user_id}/{project_id}/overview")
def project_overview(user_id: str, project_id: int) -> dict[str, object]:
    return get_project_overview(user_id, project_id)


@router.get("/projects/{user_id}/{project_id}/links")
def project_links(
    user_id: str,
    project_id: int,
    targetType: str | None = None,
    limit: int = 100,
) -> dict[str, object]:
    return {"items": list_project_links(user_id, project_id, target_type=targetType, limit=limit)}


@router.post("/projects/{user_id}/{project_id}/links")
def project_link_create(user_id: str, project_id: int, request: dict[str, object]) -> dict[str, object]:
    return create_project_link(user_id, project_id, request)


@router.delete("/projects/{user_id}/{project_id}/links/{link_id}", status_code=204)
def project_link_delete(user_id: str, project_id: int, link_id: int) -> Response:
    delete_project_link(user_id, project_id, link_id)
    return Response(status_code=204)


@router.post("/projects/{user_id}/{project_id}/decisions")
def project_decision_create(user_id: str, project_id: int, request: dict[str, object]) -> dict[str, object]:
    return create_project_decision(user_id, project_id, request)


@router.post("/projects/{user_id}/{project_id}/open-questions")
def project_open_question_create(user_id: str, project_id: int, request: dict[str, object]) -> dict[str, object]:
    return create_project_open_question(user_id, project_id, request)


@router.patch("/projects/{user_id}/open-questions/{question_id}")
def project_open_question_update(user_id: str, question_id: int, request: dict[str, object]) -> dict[str, object]:
    return update_project_open_question(user_id, question_id, request)
