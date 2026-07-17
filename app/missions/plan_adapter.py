from __future__ import annotations

from typing import Any


def mission_run_constraints(
    constraints: dict[str, Any],
    *,
    mission_id: str,
    grant_version: int,
) -> dict[str, Any]:
    """Attach non-authoritative run metadata without changing the approved resource scope."""
    return {
        **constraints,
        "missionId": mission_id,
        "delegationGrantVersion": grant_version,
    }
