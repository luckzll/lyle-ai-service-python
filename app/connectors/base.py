from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ConnectorInfo:
    name: str
    label: str
    status: str
    capabilities: list[str]
    message: str


class Connector(Protocol):
    name: str
    label: str
    capabilities: list[str]

    def status(self) -> ConnectorInfo:
        ...

    def list(self, query: dict[str, Any]) -> dict[str, Any]:
        ...

    def get(self, resource_id: str) -> dict[str, Any]:
        ...

    def search(self, query: dict[str, Any]) -> dict[str, Any]:
        ...

    def preview_create(self, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        ...


def tool_result(
    *,
    connector: str,
    ok: bool,
    content: str,
    items: list[dict[str, Any]] | None = None,
    sources: list[dict[str, Any]] | None = None,
    error_code: str | None = None,
    retryable: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "connector": connector,
        "ok": ok,
        "content": content,
        "items": items or [],
        "sources": sources or [],
    }
    if error_code:
        result["error"] = {"code": error_code, "message": content, "retryable": retryable}
    return result
