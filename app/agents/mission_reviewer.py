from __future__ import annotations

from typing import Any


def classify_mission_exception(result: dict[str, Any]) -> dict[str, Any] | None:
    status = str(result.get("status") or "").upper()
    if status in {"SUCCESS", "COMPLETED"}:
        return None
    return {
        "type": "FAILURE",
        "severity": "HIGH" if status in {"FAILED", "ERROR"} else "MEDIUM",
        "summary": str(result.get("summary") or "持续任务结果需要用户检查。")[:2000],
    }
