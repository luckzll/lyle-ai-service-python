from __future__ import annotations

from app.missions.schemas import ShadowAction, ShadowScenarioRequest, ShadowScenarioResponse


PROMPT_VERSION = "lyle-mission-v6-shadow-1"


def build_shadow_scenarios(request: ShadowScenarioRequest) -> ShadowScenarioResponse:
    workspace_id = _positive_int(request.constraints.get("workspaceId"))
    project_id = request.projectId or None
    actions: list[ShadowAction] = []
    if request.templateId in {"project-health", "release-readiness"}:
        actions.extend([
            _action("读取绑定项目状态", "project.inspect", "read", "R0", False, {"projectId": project_id}),
            _action("检查授权工作区 Git 状态", "workspace.git_status", "read", "R0", False, {"workspaceId": workspace_id}),
            _action("检查授权工作区文档", "workspace.docs_check", "read", "R0", False, {"workspaceId": workspace_id}),
        ])
    if request.templateId == "release-readiness":
        actions.append(_action("运行授权测试命令", "release.check", "read", "R0", False, {"workspaceId": workspace_id}))
    if request.templateId == "research-weekly":
        actions.append(_action("检索指定主题资料", "research.search", "search", "R0", False, {"query": request.goal}))
    actions.append(_action("合并结构化证据", "analysis.synthesize", "analyze", "R0", False, {}))

    tools = {
        str(rule.get("toolRefs", [""])[0])
        for rule in request.grantRules
        if isinstance(rule.get("toolRefs"), list) and rule.get("toolRefs")
    }
    if "todo.propose" in tools:
        actions.append(_action(
            "创建一个项目内部任务",
            "todo.propose",
            "create",
            "R1",
            True,
            {"projectId": project_id, "title": "跟进巡检异常"},
        ))
    if "file.patch.propose" in tools:
        write_target = request.constraints.get("writeTarget") if isinstance(request.constraints.get("writeTarget"), dict) else {}
        actions.append(_action(
            "应用受控工作区文件 Patch",
            "file.patch.propose",
            "patch",
            "R1",
            True,
            {"workspaceId": workspace_id, **write_target},
        ))
    actions.extend([
        _action("提交 Git 变更", "git.commit", "commit", "R2", True, {}),
        _action("永久删除资源", "resource.delete", "delete", "R3", True, {}),
    ])
    return ShadowScenarioResponse(actions=actions, promptVersion=PROMPT_VERSION)


def _action(
    title: str,
    tool_ref: str,
    operation: str,
    risk: str,
    write: bool,
    input_values: dict,
) -> ShadowAction:
    return ShadowAction(
        title=title,
        toolRef=tool_ref,
        operation=operation,
        riskLevel=risk,  # type: ignore[arg-type]
        writeIntent=write,
        input={key: value for key, value in input_values.items() if value is not None},
    )


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
