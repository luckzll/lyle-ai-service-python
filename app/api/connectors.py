from fastapi import APIRouter

from app.connectors.registry import (
    connector_get,
    connector_list,
    connector_preview_create,
    connector_search,
    connector_status,
    list_connectors,
)

router = APIRouter()


@router.get("/connectors")
def connectors() -> dict[str, object]:
    return list_connectors()


@router.get("/connectors/{connector_name}/status")
def status(connector_name: str) -> dict[str, object]:
    return connector_status(connector_name)


@router.get("/connectors/{connector_name}/list")
def list_resources(connector_name: str, repo: str | None = None, state: str | None = None, limit: int = 50) -> dict[str, object]:
    return connector_list(connector_name, {"repo": repo, "state": state, "limit": limit})


@router.get("/connectors/{connector_name}/resources/{resource_id:path}")
def get_resource(connector_name: str, resource_id: str) -> dict[str, object]:
    return connector_get(connector_name, resource_id)


@router.get("/connectors/{connector_name}/search")
def search(connector_name: str, q: str = "", repo: str | None = None, limit: int = 20) -> dict[str, object]:
    return connector_search(connector_name, {"q": q, "repo": repo, "limit": limit})


@router.post("/connectors/{user_id}/{connector_name}/preview-create")
def preview_create(user_id: str, connector_name: str, request: dict[str, object]) -> dict[str, object]:
    return connector_preview_create(user_id, connector_name, request)
