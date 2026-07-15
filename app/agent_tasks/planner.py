from __future__ import annotations

from typing import Any

from app.agent_tasks.schemas import PlanRequest, PlanResponse, PlanStep


PROMPT_VERSION = "lyle-agent-v5-planner-1"


def build_plan(request: PlanRequest) -> PlanResponse:
    allowed = {item.toolRef: item.riskLevel for item in request.toolCatalog}
    forbidden_tools = _forbidden_tools(request.constraints, allowed)
    allowed = {tool_ref: risk for tool_ref, risk in allowed.items() if tool_ref not in forbidden_tools}
    if not allowed:
        raise ValueError("禁止事项排除了全部可用工具，请调整约束后重试。")
    template = _template(request)
    steps = _steps_for_template(template, request, allowed)
    if request.autonomyLevel >= 2:
        write_step = _write_step(request, allowed, steps[-1].stepKey if steps else None)
        if write_step is not None:
            steps.append(write_step)

    inferred: list[str] = []
    if request.projectId:
        inferred.append(f"任务关联项目 #{request.projectId}，但不会因此扩大资源权限。")
    if request.autonomyLevel == 0:
        inferred.append("本次仅生成计划，不自动开始执行。")
    elif request.autonomyLevel == 1:
        inferred.append("批准后仅连续执行只读步骤。")
    else:
        inferred.append("写操作只生成变更预览，每次仍需单独确认。")

    return PlanResponse(
        understoodGoal=request.goal.strip(),
        inferredUnderstanding=inferred,
        blockingQuestions=[],
        accessScope={
            "projectIds": [request.projectId] if request.projectId else [],
            "conversationIds": [request.conversationId] if request.conversationId else [],
            "workspaceIds": [int(request.constraints["workspaceId"])] if _positive_int(request.constraints.get("workspaceId")) else [],
            "network": "AUTHORIZED_SEARCH_ONLY" if any(step.riskLevel == "EXTERNAL_READ" for step in steps) else "NONE",
            "allowedTools": [step.toolRef for step in steps],
            "forbiddenTools": sorted(forbidden_tools),
            "forbidden": _forbidden_texts(request.constraints),
            "writeRequiresConfirmation": True,
        },
        budget=request.budget,
        steps=steps,
        promptVersion=PROMPT_VERSION,
    )


def _template(request: PlanRequest) -> str:
    explicit = (request.templateId or "").lower()
    if explicit in {"project-inspection", "project_inspection", "inspection"}:
        return "inspection"
    if explicit in {"release-check", "release_check", "release"}:
        return "release"
    if explicit in {"research", "research-report"}:
        return "research"
    goal = request.goal.lower()
    if any(word in goal for word in ["发布", "release", "go/no-go", "测试失败"]):
        return "release"
    if any(word in goal for word in ["调研", "对比", "research", "检索", "资料"]):
        return "research"
    return "inspection"


def _steps_for_template(template: str, request: PlanRequest, allowed: dict[str, str]) -> list[PlanStep]:
    candidates: list[dict[str, Any]]
    if template == "research":
        candidates = [
            _step("search_sources", "SEARCH", "检索授权资料", "围绕目标检索外部来源并保留引用。", "research.search", "EXTERNAL_READ", {"query": "{{goal}}"}),
            _step("synthesize", "ANALYZE", "形成有引用的对比结论", "汇总事实、差异、风险和未验证假设。", "analysis.synthesize", "READ", {"mode": "research"}, ["search_sources"]),
        ]
    elif template == "release":
        workspace_input = _workspace_input(request)
        candidates = [
            _step("inspect_project", "READ", "读取项目现状", "读取目标项目、目标、计划和待办的实时摘要。", "project.inspect", "READ", _project_input(request)),
            _step("git_status", "TOOL", "检查 Git 状态", "读取授权工作区的 Git 状态并保留命令证据。", "workspace.git_status", "READ", workspace_input, ["inspect_project"]),
            _step("docs_check", "READ", "检查发布文档", "检查 README 与 docs 文档是否存在并保留文件证据。", "workspace.docs_check", "READ", workspace_input, ["git_status"]),
            _step("release_check", "TOOL", "运行授权测试", "在授权工作区运行白名单测试命令，记录退出码和摘要。", "release.check", "READ", {**workspace_input, "commands": _release_commands(request)}, ["docs_check"]),
            _step("synthesize", "ANALYZE", "形成 Go/No-Go 建议", "对照完成标准整理证据、阻塞项和发布建议。", "analysis.synthesize", "READ", {"mode": "release"}, ["release_check"]),
        ]
    else:
        workspace_input = _workspace_input(request)
        candidates = [
            _step("inspect_project", "READ", "读取项目状态", "检查目标项目、目标、计划、待办及其关联。", "project.inspect", "READ", _project_input(request)),
            _step("git_status", "TOOL", "检查 Git 状态", "读取授权工作区的 Git 状态，不修改仓库。", "workspace.git_status", "READ", workspace_input, ["inspect_project"]),
            _step("docs_check", "READ", "检查项目文档", "检查 README 与 docs 文档并保留证据。", "workspace.docs_check", "READ", workspace_input, ["git_status"]),
            _step("inspect_memory", "READ", "核对相关上下文", "读取授权记忆快照，核对偏好和既有事实。", "memory.snapshot", "READ", {"question": "{{goal}}"}, ["docs_check"]),
            _step("synthesize", "ANALYZE", "汇总风险与下一步", "对照完成标准整理事实、证据、缺口与建议。", "analysis.synthesize", "READ", {"mode": "inspection"}, ["inspect_memory"]),
        ]
    skipped = {str(item) for item in request.constraints.get("skipToolRefs", [])} if isinstance(request.constraints.get("skipToolRefs"), list) else set()
    completed = {str(item) for item in request.constraints.get("completedToolRefs", [])} if isinstance(request.constraints.get("completedToolRefs"), list) else set()
    filtered = [
        dict(item)
        for item in candidates
        if item["toolRef"] in allowed and item["toolRef"] not in skipped and (item["toolRef"] not in completed or item["toolRef"] == "analysis.synthesize")
    ]
    included_keys: list[str] = []
    for item in filtered:
        declared = [value for value in item.get("dependencies", []) if value in included_keys]
        if item.get("dependencies") and not declared and included_keys:
            declared = [included_keys[-1]]
        item["dependencies"] = declared
        included_keys.append(str(item["stepKey"]))
    result = [PlanStep.model_validate(item) for item in filtered]
    if not result:
        fallback = next(iter(allowed.items()))
        result = [PlanStep.model_validate(_step(
            "authorized_check", "TOOL", "执行授权检查", "使用审批范围内的工具推进目标。",
            fallback[0], fallback[1], {"question": "{{goal}}"},
        ))]
    return result


def _write_step(request: PlanRequest, allowed: dict[str, str], dependency: str | None) -> PlanStep | None:
    constraints = request.constraints
    goal = request.goal.lower()
    if "todo.propose" in allowed and any(word in goal for word in ["创建任务", "创建待办", "任务草稿", "todo"]):
        title = str(constraints.get("taskTitle") or request.goal)[:200]
        configured = constraints.get("taskDrafts") if isinstance(constraints.get("taskDrafts"), list) else []
        items = [item for item in configured if isinstance(item, dict) and str(item.get("title") or "").strip()][:3]
        if not items and any(word in goal for word in ["3 个", "3个", "三个", "任务草稿"]):
            items = [
                {"title": "整理调研结论与证据", "source": "Agent Task 调研结果"},
                {"title": "验证关键假设与风险", "source": "Agent Task 未验证项"},
                {"title": "推进建议方案并明确负责人", "source": "Agent Task 建议"},
            ]
        return PlanStep.model_validate(_step(
            "propose_tasks", "PROPOSE_CHANGE", "提出任务草稿", "生成结构化任务草稿并等待单次确认。",
            "todo.propose", "WRITE_LOW", {"title": title, "items": items}, [dependency] if dependency else [], True,
        ))
    write_target = constraints.get("writeTarget") if isinstance(constraints.get("writeTarget"), dict) else {}
    if "file.patch.propose" in allowed and write_target and any(word in goal for word in ["修改文件", "patch", "补充文档"]):
        return PlanStep.model_validate(_step(
            "propose_patch", "PROPOSE_CHANGE", "提出文件 Patch", "生成文件差异并等待单次确认。",
            "file.patch.propose", "WRITE_LOW", write_target, [dependency] if dependency else [], True,
        ))
    return None


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _workspace_input(request: PlanRequest) -> dict[str, Any]:
    workspace_id = _positive_int(request.constraints.get("workspaceId"))
    return {"workspaceId": workspace_id} if workspace_id else {}


def _project_input(request: PlanRequest) -> dict[str, Any]:
    if request.projectId is None:
        raise ValueError("项目巡检和发布检查必须绑定目标项目。")
    return {"projectId": request.projectId, "question": "{{goal}}"}


def _forbidden_texts(constraints: dict[str, Any]) -> list[str]:
    raw = constraints.get("forbidden")
    if isinstance(raw, str):
        values = raw.splitlines()
    elif isinstance(raw, list):
        values = [str(item) for item in raw]
    else:
        values = []
    return [value.strip() for value in values if value.strip()][:50]


def _forbidden_tools(constraints: dict[str, Any], allowed: dict[str, str]) -> set[str]:
    texts = [value.lower().replace(" ", "") for value in _forbidden_texts(constraints)]
    result: set[str] = set()
    for tool_ref, risk in allowed.items():
        compact_ref = tool_ref.lower().replace(" ", "")
        if any(compact_ref in text for text in texts):
            result.add(tool_ref)
        if risk == "EXTERNAL_READ" and any(_matches(text, [
            "不访问外网", "禁止访问外网", "不联网", "禁止联网", "离线执行",
            "nointernet", "nonetwork", "offlineonly",
        ]) for text in texts):
            result.add(tool_ref)
        if tool_ref == "todo.propose" and any(_matches(text, [
            "不创建任务", "禁止创建任务", "不创建待办", "禁止创建待办",
            "notaskcreation", "notodo", "donotcreatetask",
        ]) for text in texts):
            result.add(tool_ref)
        if tool_ref == "file.patch.propose" and any(_matches(text, [
            "不改文件", "不修改文件", "禁止修改文件", "不写文件", "只读文件",
            "nofilechanges", "donotmodifyfiles", "readonlyfiles",
        ]) for text in texts):
            result.add(tool_ref)
        if tool_ref == "memory.snapshot" and any(_matches(text, [
            "不读取记忆", "禁止读取记忆", "nomemory", "donotreadmemory",
        ]) for text in texts):
            result.add(tool_ref)
    return result


def _matches(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _release_commands(request: PlanRequest) -> list[str]:
    raw = request.constraints.get("releaseCommands")
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()][:5]


def _step(
    key: str,
    step_type: str,
    title: str,
    purpose: str,
    tool_ref: str,
    risk: str,
    input_template: dict[str, Any],
    dependencies: list[str] | None = None,
    write_intent: bool = False,
) -> dict[str, Any]:
    return {
        "stepKey": key,
        "stepType": step_type,
        "title": title,
        "purpose": purpose,
        "toolRef": tool_ref,
        "inputTemplate": input_template,
        "dependencies": dependencies or [],
        "riskLevel": risk,
        "expectedArtifact": {"type": "CHANGE_PREVIEW" if write_intent else "EVIDENCE"},
        "writeIntent": write_intent,
    }
