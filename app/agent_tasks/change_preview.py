from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from app.agent_tasks.schemas import ChangePreviewRequest, ChangePreviewResponse
from app.cli.workspace_store import get_workspace
from app.tools.operation_confirmation import create_pending_operation


def build_change_preview(request: ChangePreviewRequest) -> ChangePreviewResponse:
    if request.toolRef == "todo.propose":
        return _todo_preview(request)
    return _file_patch_preview(request)


def apply_workspace_file_patch(user_id: str, payload: dict[str, Any]) -> dict[str, object]:
    workspace_id = _positive_int(payload.get("workspaceId"), "workspaceId")
    workspace = get_workspace(user_id, workspace_id)
    root = Path(str(workspace["rootPath"])).expanduser().resolve()
    relative_path = _relative_path(payload.get("relativePath"))
    target = (root / relative_path).resolve()
    if root not in target.parents and target != root:
        return {"ok": False, "content": "文件路径超出已授权工作区。", "errorCode": "PATH_OUTSIDE_WORKSPACE"}
    current = target.read_text(encoding="utf-8") if target.exists() else ""
    current_hash = hashlib.sha256(current.encode("utf-8")).hexdigest()
    expected_hash = str(payload.get("expectedSha256") or "")
    if expected_hash and expected_hash != current_hash:
        return {"ok": False, "content": "文件已变化，Patch 未应用。", "errorCode": "RESOURCE_VERSION_CHANGED"}
    new_content = str(payload.get("newContent") or "")
    if len(new_content.encode("utf-8")) > 1_000_000:
        return {"ok": False, "content": "Patch 结果超过 1MB 安全上限。", "errorCode": "PATCH_TOO_LARGE"}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(new_content, encoding="utf-8")
    return {
        "ok": True,
        "content": f"已应用文件变更：{relative_path}",
        "file": {
            "workspaceId": workspace_id,
            "relativePath": relative_path,
            "beforeSha256": current_hash,
            "afterSha256": hashlib.sha256(new_content.encode("utf-8")).hexdigest(),
        },
    }


def _todo_preview(request: ChangePreviewRequest) -> ChangePreviewResponse:
    title = str(request.input.get("title") or request.goal).strip()[:200]
    if not title:
        raise ValueError("任务标题不能为空。")
    raw_items = request.input.get("items") if isinstance(request.input.get("items"), list) else []
    items = [
        {
            "title": str(item.get("title") or "").strip()[:200],
            "dueAt": item.get("dueAt"),
            "source": str(item.get("source") or f"Agent Task {request.taskId}")[:500],
        }
        for item in raw_items
        if isinstance(item, dict) and str(item.get("title") or "").strip()
    ][:3]
    batch = len(items) > 1
    payload = {"items": items} if batch else {"title": items[0]["title"] if items else title, "dueAt": items[0].get("dueAt") if items else request.input.get("dueAt")}
    summary = f"创建 {len(items)} 个项目任务草稿" if batch else f"创建任务草稿「{payload['title']}」"
    result = create_pending_operation(
        user_id=request.userId,
        conversation_id=request.conversationId,
        tool_name="todo",
        operation_type="todo.batch_create" if batch else "todo.create",
        summary=summary,
        payload=payload,
        source_text=f"Agent Task {request.taskId} / Step {request.stepRunId}",
        risk_level="write",
    )
    operation = result.get("operation") if isinstance(result.get("operation"), dict) else {}
    if not result.get("ok") or not operation.get("id"):
        raise ValueError(str(result.get("content") or "无法创建 Pending Operation。"))
    return ChangePreviewResponse(
        pendingOperationId=int(operation["id"]),
        resourceRef="project-task://new",
        actionType="todo.batch_create" if batch else "todo.create",
        summary=summary,
        before={},
        after={"items": items} if batch else payload,
        impact=f"确认后只创建一次{'这批任务' if batch else '任务'}；后续步骤将继续执行。",
        undoSupported=False,
        compensation={},
    )


def _file_patch_preview(request: ChangePreviewRequest) -> ChangePreviewResponse:
    workspace_id = _positive_int(request.input.get("workspaceId"), "workspaceId")
    workspace = get_workspace(request.userId, workspace_id)
    root = Path(str(workspace["rootPath"])).expanduser().resolve()
    relative_path = _relative_path(request.input.get("relativePath"))
    target = (root / relative_path).resolve()
    if root not in target.parents and target != root:
        raise ValueError("文件路径超出已授权工作区。")
    before_content = target.read_text(encoding="utf-8") if target.exists() else ""
    before_hash = hashlib.sha256(before_content.encode("utf-8")).hexdigest()
    expected_hash = str(request.input.get("expectedSha256") or before_hash)
    if expected_hash != before_hash:
        raise ValueError("文件版本与计划生成时不一致。")
    new_content = str(request.input.get("newContent") or "")
    payload = {
        "workspaceId": workspace_id,
        "relativePath": relative_path,
        "expectedSha256": expected_hash,
        "newContent": new_content,
    }
    result = create_pending_operation(
        user_id=request.userId,
        conversation_id=request.conversationId,
        tool_name="file_patch",
        operation_type="file.patch",
        summary=f"修改工作区文件 {relative_path}",
        payload=payload,
        source_text=f"Agent Task {request.taskId} / Step {request.stepRunId}",
        risk_level="write",
    )
    operation = result.get("operation") if isinstance(result.get("operation"), dict) else {}
    if not result.get("ok") or not operation.get("id"):
        raise ValueError(str(result.get("content") or "无法创建 Pending Operation。"))
    return ChangePreviewResponse(
        pendingOperationId=int(operation["id"]),
        resourceRef=f"workspace://{workspace_id}/{relative_path}",
        actionType="file.patch",
        summary=f"修改文件 {relative_path}",
        before={"sha256": before_hash, "characters": len(before_content)},
        after={"sha256": hashlib.sha256(new_content.encode("utf-8")).hexdigest(), "characters": len(new_content)},
        impact="确认时会再次校验文件哈希；文件已变化则拒绝应用。",
        undoSupported=False,
        compensation={},
    )


def _positive_int(value: object, field: str) -> int:
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 必须是正整数。") from exc
    if result <= 0:
        raise ValueError(f"{field} 必须是正整数。")
    return result


def _relative_path(value: object) -> str:
    text = str(value or "").strip()
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise ValueError("relativePath 必须是工作区内的相对路径。")
    return path.as_posix()
