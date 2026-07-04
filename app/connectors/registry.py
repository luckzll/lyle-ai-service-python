from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import HTTPException

from app.connectors.github_connector import GitHubConnector
from app.connectors.local_docs import LocalDocsConnector


_CONNECTORS = {
    "local_docs": LocalDocsConnector(),
    "github": GitHubConnector(),
}


def list_connectors() -> dict[str, Any]:
    return {"items": [asdict(connector.status()) for connector in _CONNECTORS.values()]}


def connector_status(name: str) -> dict[str, Any]:
    return asdict(_get_connector(name).status())


def connector_list(name: str, query: dict[str, Any]) -> dict[str, Any]:
    return _get_connector(name).list(query)


def connector_get(name: str, resource_id: str) -> dict[str, Any]:
    return _get_connector(name).get(resource_id)


def connector_search(name: str, query: dict[str, Any]) -> dict[str, Any]:
    return _get_connector(name).search(query)


def connector_preview_create(user_id: str, name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return _get_connector(name).preview_create(user_id, payload)


def _get_connector(name: str):
    connector = _CONNECTORS.get(name)
    if connector is None:
        raise HTTPException(status_code=404, detail="连接器不存在。")
    return connector
