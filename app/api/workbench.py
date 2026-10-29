from fastapi import APIRouter, Query

from app.automation.automation_runner import check_event_automation, run_automation
from app.automation.automation_store import (
    create_automation,
    delete_automation,
    list_automation_runs,
    list_automations,
    set_automation_status,
    update_automation,
)
from app.cli.command_executor import list_command_runs, preview_command, run_command
from app.mcp.client_manager import run_mcp_health_check, run_mcp_tool_discovery
from app.mcp.tool_executor import invoke_mcp_tool
from app.cli.workspace_store import (
    create_workspace,
    create_workspace_policy,
    delete_workspace_policy,
    list_workspace_policies,
    list_workspaces,
)
from app.mcp.tool_registry import (
    create_mcp_server,
    get_mcp_server_config,
    list_mcp_servers,
    list_mcp_tools,
    mark_health,
    sync_mcp_tools,
    update_mcp_server_status,
    update_tool_policy,
)
from app.security.execution_audit import list_execution_audit
from app.workflows.workflow_runner import (
    cancel_workflow_run,
    get_workflow_run,
    list_workflow_runs,
    resume_workflow_run,
    retry_workflow_run,
    run_workflow,
)
from app.workflows.workflow_store import create_workflow, create_workflow_from_plan, get_workflow, list_workflows, update_workflow

router = APIRouter()


@router.get("/workbench/integrations/{user_id}")
def integration_catalog(user_id: str):
    from app.tools.integrated_tools import catalog
    from app.tools.integration_presets import presets_for_user
    return {**catalog(user_id), "presets": presets_for_user(user_id)}


@router.post("/workbench/integrations/{user_id}/invoke")
def integration_invoke(user_id: str, request: dict[str, object]):
    import re
    from fastapi import HTTPException
    from app.tools.integrated_tools import ToolSelection, invoke
    selection = ToolSelection.model_validate(request.get("toolSelection") or {})
    session_id = str(request.get("sessionId") or "")
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", session_id):
        raise HTTPException(400, "本地对话标识无效。")
    # Models cannot supply approval booleans; writes always use durable operations.
    return invoke(user_id, "local:" + session_id, str(request.get("name") or ""),
                  request.get("arguments") or {}, selection)


@router.get("/workbench/mcp/{user_id}/servers")
def mcp_servers(user_id: str) -> dict[str, object]:
    return {"items": list_mcp_servers(user_id)}


@router.post("/workbench/mcp/{user_id}/servers")
def mcp_server_create(user_id: str, request: dict[str, object]) -> dict[str, object]:
    return create_mcp_server(user_id, request)


@router.patch("/workbench/mcp/{user_id}/servers/{server_id}/status")
def mcp_server_status(user_id: str, server_id: int, request: dict[str, object]) -> dict[str, object]:
    return update_mcp_server_status(user_id, server_id, str(request.get("status") or ""))


@router.post("/workbench/mcp/{user_id}/servers/{server_id}/health-check")
def mcp_health(user_id: str, server_id: int) -> dict[str, object]:
    server = get_mcp_server_config(user_id, server_id)
    result = run_mcp_health_check(server)
    return mark_health(user_id, server_id, "ok" if result["ok"] else "error", str(result["message"]))


@router.get("/workbench/mcp/{user_id}/tools")
def mcp_tools(user_id: str, serverId: int | None = None) -> dict[str, object]:
    return {"items": list_mcp_tools(user_id, serverId)}


@router.post("/workbench/mcp/{user_id}/servers/{server_id}/tools/sync")
def mcp_tools_sync(user_id: str, server_id: int, request: dict[str, object]) -> dict[str, object]:
    raw_tools = request.get("tools") if isinstance(request.get("tools"), list) else []
    tools = [item for item in raw_tools if isinstance(item, dict)]
    if not tools:
        server = get_mcp_server_config(user_id, server_id)
        if server["serverType"] == "builtin":
            return {"items": list_mcp_tools(user_id, server_id)}
        tools = run_mcp_tool_discovery(server)
        mark_health(user_id, server_id, "ok", f"工具同步成功，共 {len(tools)} 个工具。")
    return {"items": sync_mcp_tools(user_id, server_id, tools)}


@router.patch("/workbench/mcp/{user_id}/tools/{tool_id}/policy")
def mcp_tool_policy(user_id: str, tool_id: int, request: dict[str, object]) -> dict[str, object]:
    return update_tool_policy(user_id, tool_id, request)


@router.post("/workbench/mcp/{user_id}/tools/{tool_id}/invoke")
def mcp_tool_invoke(user_id: str, tool_id: int, request: dict[str, object]) -> dict[str, object]:
    return invoke_mcp_tool(user_id, tool_id, request)


@router.get("/workbench/workspaces/{user_id}")
def workspaces(user_id: str) -> dict[str, object]:
    return {"items": list_workspaces(user_id)}


@router.post("/workbench/workspaces/{user_id}")
def workspace_create(user_id: str, request: dict[str, object]) -> dict[str, object]:
    return create_workspace(user_id, request)


@router.get("/workbench/workspaces/{user_id}/{workspace_id}/policies")
def workspace_policies(user_id: str, workspace_id: int) -> dict[str, object]:
    return {"items": list_workspace_policies(user_id, workspace_id)}


@router.post("/workbench/workspaces/{user_id}/{workspace_id}/policies")
def workspace_policy_create(user_id: str, workspace_id: int, request: dict[str, object]) -> dict[str, object]:
    return create_workspace_policy(user_id, workspace_id, request)


@router.delete("/workbench/workspaces/{user_id}/{workspace_id}/policies/{policy_id}")
def workspace_policy_delete(user_id: str, workspace_id: int, policy_id: int) -> dict[str, object]:
    return delete_workspace_policy(user_id, workspace_id, policy_id)


@router.post("/workbench/workspaces/{user_id}/{workspace_id}/commands/preview")
def command_preview(user_id: str, workspace_id: int, request: dict[str, object]) -> dict[str, object]:
    return preview_command(user_id, workspace_id, str(request.get("command") or ""))


@router.post("/workbench/workspaces/{user_id}/{workspace_id}/commands/run")
def command_run(user_id: str, workspace_id: int, request: dict[str, object]) -> dict[str, object]:
    return run_command(
        user_id,
        workspace_id,
        str(request.get("command") or ""),
        confirmed=bool(request.get("confirmed", False)),
        second_confirmed=bool(request.get("secondConfirmed", False)),
        timeout_seconds=int(request.get("timeoutSeconds") or 60),
    )


@router.get("/workbench/workspaces/{user_id}/{workspace_id}/commands")
def command_runs(user_id: str, workspace_id: int, limit: int = Query(default=50, ge=1, le=200)) -> dict[str, object]:
    return {"items": list_command_runs(user_id, workspace_id, limit)}


@router.get("/workbench/workflows/{user_id}")
def workflows(user_id: str) -> dict[str, object]:
    return {"items": list_workflows(user_id)}


@router.post("/workbench/workflows/{user_id}")
def workflow_create(user_id: str, request: dict[str, object]) -> dict[str, object]:
    return create_workflow(user_id, request)


@router.post("/workbench/workflows/{user_id}/from-plan/{plan_id}")
def workflow_create_from_plan(user_id: str, plan_id: int) -> dict[str, object]:
    return create_workflow_from_plan(user_id, plan_id)


@router.get("/workbench/workflows/{user_id}/{workflow_id}")
def workflow_detail(user_id: str, workflow_id: int) -> dict[str, object]:
    return get_workflow(user_id, workflow_id)


@router.patch("/workbench/workflows/{user_id}/{workflow_id}")
def workflow_update(user_id: str, workflow_id: int, request: dict[str, object]) -> dict[str, object]:
    return update_workflow(user_id, workflow_id, request)


@router.post("/workbench/workflows/{user_id}/{workflow_id}/runs")
def workflow_run_create(user_id: str, workflow_id: int, request: dict[str, object]) -> dict[str, object]:
    return run_workflow(user_id, workflow_id, request)


@router.get("/workbench/workflows/{user_id}/{workflow_id}/runs")
def workflow_runs(user_id: str, workflow_id: int, limit: int = Query(default=50, ge=1, le=200)) -> dict[str, object]:
    return {"items": list_workflow_runs(user_id, workflow_id, limit)}


@router.get("/workbench/workflow-runs/{user_id}")
def all_workflow_runs(user_id: str, limit: int = Query(default=50, ge=1, le=200)) -> dict[str, object]:
    return {"items": list_workflow_runs(user_id, None, limit)}


@router.get("/workbench/workflow-runs/{user_id}/{run_id}")
def workflow_run_detail(user_id: str, run_id: int) -> dict[str, object]:
    return get_workflow_run(user_id, run_id)


@router.post("/workbench/workflow-runs/{user_id}/{run_id}/resume")
def workflow_run_resume(user_id: str, run_id: int, request: dict[str, object]) -> dict[str, object]:
    return resume_workflow_run(user_id, run_id, request)


@router.post("/workbench/workflow-runs/{user_id}/{run_id}/retry")
def workflow_run_retry(user_id: str, run_id: int, request: dict[str, object]) -> dict[str, object]:
    return retry_workflow_run(user_id, run_id, request)


@router.post("/workbench/workflow-runs/{user_id}/{run_id}/cancel")
def workflow_run_cancel(user_id: str, run_id: int) -> dict[str, object]:
    return cancel_workflow_run(user_id, run_id)


@router.get("/workbench/automations/{user_id}")
def automations(user_id: str) -> dict[str, object]:
    return {"items": list_automations(user_id)}


@router.post("/workbench/automations/{user_id}")
def automation_create(user_id: str, request: dict[str, object]) -> dict[str, object]:
    return create_automation(user_id, request)


@router.patch("/workbench/automations/{user_id}/{automation_id}")
def automation_update(user_id: str, automation_id: int, request: dict[str, object]) -> dict[str, object]:
    return update_automation(user_id, automation_id, request)


@router.delete("/workbench/automations/{user_id}/{automation_id}")
def automation_delete(user_id: str, automation_id: int) -> dict[str, object]:
    return delete_automation(user_id, automation_id)


@router.post("/workbench/automations/{user_id}/{automation_id}/runs")
def automation_run_create(user_id: str, automation_id: int, request: dict[str, object]) -> dict[str, object]:
    override_input = request.get("input") if isinstance(request.get("input"), dict) else None
    return run_automation(user_id, automation_id, override_input)


@router.post("/workbench/automations/{user_id}/{automation_id}/check-trigger")
def automation_trigger_check(user_id: str, automation_id: int) -> dict[str, object]:
    return check_event_automation(user_id, automation_id)


@router.get("/workbench/automations/{user_id}/{automation_id}/runs")
def automation_runs(user_id: str, automation_id: int, limit: int = Query(default=50, ge=1, le=200)) -> dict[str, object]:
    return {"items": list_automation_runs(user_id, automation_id, limit)}


@router.post("/workbench/automations/{user_id}/{automation_id}/enable")
def automation_enable(user_id: str, automation_id: int) -> dict[str, object]:
    return set_automation_status(user_id, automation_id, "enabled")


@router.post("/workbench/automations/{user_id}/{automation_id}/pause")
def automation_pause(user_id: str, automation_id: int) -> dict[str, object]:
    return set_automation_status(user_id, automation_id, "paused")


@router.get("/workbench/audit/{user_id}")
def execution_audit(user_id: str, limit: int = Query(default=100, ge=1, le=200)) -> dict[str, object]:
    return {"items": list_execution_audit(user_id, limit)}
