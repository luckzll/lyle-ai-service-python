from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from app.agent_tasks.schemas import ExecuteStepRequest, ToolResult
from app.agent_tasks.resume_check import compute_workspace_state, project_state_hash
from app.cli.command_executor import run_command
from app.cli.command_risk import classify_command
from app.cli.workspace_store import get_workspace
from app.projects.project_store import get_project_overview
from app.security.secret_masker import mask_secrets
from app.tools.tool_registry import execute_tool
from app.missions.delegated_actions import execute_delegated_action


TOOL_MAPPING = {
    "research.search": "web_search",
    "memory.snapshot": "memory_snapshot",
    "tools.list": "list_tools",
}

SAFE_SOURCE_HOST = re.compile(r"(?=.{1,253}\Z)[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?\Z", re.IGNORECASE)


def execute_approved_step(request: ExecuteStepRequest) -> ToolResult:
    if request.toolRef in {"todo.propose", "file.patch.propose"}:
        return execute_delegated_action(request)
    if request.toolRef == "project.inspect":
        return _project_inspect(request)
    if request.toolRef == "analysis.synthesize":
        return _synthesize(request)
    if request.toolRef == "release.check":
        return _release_check(request)
    if request.toolRef == "workspace.git_status":
        return _git_status(request)
    if request.toolRef == "workspace.docs_check":
        return _docs_check(request)
    tool_name = TOOL_MAPPING.get(request.toolRef)
    if tool_name is None:
        return ToolResult(
            status="ERROR",
            summary="控制器批准的工具没有对应执行适配器。",
            retryable=False,
            errorCode="TOOL_NOT_SUPPORTED",
        )
    question = str(request.input.get("question") or request.input.get("query") or request.goal)[:10_000]
    result = execute_tool(tool_name, request.userId, request.conversationId, question)
    ok = bool(result.get("ok"))
    content = str(result.get("userMessage") or result.get("content") or "").strip()
    error = result.get("error") if isinstance(result.get("error"), dict) else {}
    trust = "UNTRUSTED_EXTERNAL" if request.toolRef == "research.search" else "TRUSTED"
    security_flags = _security_flags(content) if trust == "UNTRUSTED_EXTERNAL" else []
    evidence = _evidence(result, request)
    summary, redacted, sensitivity = _display_summary(request.toolRef, result, ok)
    criterion_evidence = _criterion_evidence(
        request.completionCriteria,
        request.toolRef,
        summary,
        evidence,
        bool(ok),
        result,
    )
    return ToolResult(
        status="SUCCESS" if ok else "ERROR",
        summary=summary,
        dataRef=f"tool-result://{request.stepRunId}",
        evidenceRefs=evidence,
        sideEffect="NONE",
        retryable=bool(result.get("retryable") or error.get("retryable")),
        errorCode=None if ok else str(error.get("code") or "TOOL_ERROR")[:80],
        contentTrust=trust,
        redacted=redacted,
        sensitivity=sensitivity,
        securityFlags=security_flags,
        criterionEvidence=criterion_evidence,
    )


def _display_summary(tool_ref: str, result: dict[str, Any], ok: bool) -> tuple[str, bool, str]:
    if tool_ref == "memory.snapshot":
        raw = result.get("raw") if isinstance(result.get("raw"), dict) else {}
        summary_count = 1 if str(raw.get("summary") or "").strip() else 0
        structured_count = _safe_count(raw.get("structured"))
        short_term_count = _safe_count(raw.get("short_term"))
        retrieved_count = _safe_count(raw.get("retrieved"))
        status = "已读取个人记忆索引" if ok else "个人记忆索引读取失败"
        return (
            f"{status}：长期摘要 {summary_count} 条，结构化记忆 {structured_count} 条，"
            f"短期记忆 {short_term_count} 条，召回结果 {retrieved_count} 条。敏感正文已省略。",
            True,
            "PERSONAL",
        )
    if tool_ref == "research.search":
        results = result.get("results") if isinstance(result.get("results"), list) else []
        domains = _source_domains(results)
        source_text = "、".join(domains[:8]) if domains else "无可展示来源域名"
        status = "已完成授权网页检索" if ok else "授权网页检索失败"
        return (
            f"{status}：获得 {len(results)} 个可追溯来源；来源域名：{source_text}。"
            "外部网页正文与搜索片段已省略。",
            True,
            "EXTERNAL_CONTENT",
        )
    content = str(result.get("userMessage") or result.get("content") or "").strip()
    summary = mask_secrets(content)[:8_000] or ("工具执行成功。" if ok else "工具执行失败。")
    return summary, summary != content, "NONE"


def _safe_count(value: Any) -> int:
    return len(value) if isinstance(value, (list, tuple, set, dict)) else 0


def _source_domains(results: list[Any]) -> list[str]:
    domains: list[str] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        try:
            candidate = (urlsplit(str(item.get("url") or "")).hostname or "").lower().rstrip(".")
        except ValueError:
            candidate = ""
        if candidate and SAFE_SOURCE_HOST.fullmatch(candidate) and candidate not in domains:
            domains.append(candidate)
    return domains


def _project_inspect(request: ExecuteStepRequest) -> ToolResult:
    project_id = request.projectId or _positive_int(request.input.get("projectId"))
    if project_id is None:
        return _tool_error("项目巡检必须绑定目标 projectId，不能用全局工作摘要代替。", "PROJECT_REQUIRED")
    input_project_id = _positive_int(request.input.get("projectId"))
    if input_project_id is not None and input_project_id != project_id:
        return _tool_error("执行输入中的 projectId 与已批准任务范围不一致。", "PROJECT_SCOPE_MISMATCH")
    try:
        overview = get_project_overview(request.userId, project_id)
    except Exception as exc:
        return _tool_error(f"目标项目读取失败：{exc}", "PROJECT_UNAVAILABLE")
    project = overview.get("project") if isinstance(overview.get("project"), dict) else {}
    name = str(project.get("name") or f"项目 #{project_id}")
    status = str(project.get("status") or "unknown")
    stage = str(project.get("currentStage") or "未设置")
    link_counts = overview.get("linkCounts") if isinstance(overview.get("linkCounts"), dict) else {}
    decisions = overview.get("decisions") if isinstance(overview.get("decisions"), list) else []
    questions = overview.get("openQuestions") if isinstance(overview.get("openQuestions"), list) else []
    summary = (
        f"已读取目标项目 #{project_id}「{name}」：状态 {status}，阶段 {stage}；"
        f"关联资源 {sum(int(value) for value in link_counts.values() if isinstance(value, (int, float)))} 个，"
        f"有效决策 {len(decisions)} 条，开放问题 {len(questions)} 条。"
    )
    ref = f"project://{project_id}/overview"
    state = project_state_hash(overview)
    return ToolResult(
        status="SUCCESS",
        summary=summary,
        dataRef=ref,
        evidenceRefs=[ref],
        sideEffect="NONE",
        retryable=False,
        criterionEvidence=_criterion_evidence(
            request.completionCriteria, request.toolRef, summary, [ref], True,
            {"projectId": project_id},
        ),
        externalStateHashes={f"project:{project_id}": state},
    )


def _synthesize(request: ExecuteStepRequest) -> ToolResult:
    summaries: list[str] = []
    derived_sensitivities: set[str] = set()
    for item in request.artifacts:
        if not item.get("summary"):
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        tool_ref = str(metadata.get("toolRef") or "")
        value = mask_secrets(str(item.get("summary") or "").strip())
        sensitivity = str(metadata.get("sensitivity") or "NONE")
        if sensitivity in {"PERSONAL", "EXTERNAL_CONTENT"}:
            derived_sensitivities.add(sensitivity)
        if tool_ref in {"memory.snapshot", "research.search"} and metadata.get("redacted") is not True:
            value = _derived_sensitive_fallback(tool_ref)
            derived_sensitivities.add("PERSONAL" if tool_ref == "memory.snapshot" else "EXTERNAL_CONTENT")
        elif _contains_sensitive_body(value):
            body_sensitivity = "PERSONAL" if "长期摘要：" in value or "长期记忆摘要" in value else "EXTERNAL_CONTENT"
            value = "敏感上游正文已省略；仅保留结构化执行结论。"
            derived_sensitivities.add(body_sensitivity)
        summaries.append(value)
    if not summaries:
        summary = "前序步骤没有留下可复用证据，当前只能保留未验证结论。"
        status = "PARTIAL"
        evidence: list[Any] = []
    else:
        summary = "执行结果汇总：\n" + "\n".join(f"- {value[:1_500]}" for value in summaries[-10:])
        status = "SUCCESS"
        evidence = [item.get("contentRef") for item in request.artifacts if item.get("contentRef")][:50]
    return ToolResult(
        status=status,
        summary=summary[:8_000],
        dataRef=f"tool-result://{request.stepRunId}",
        evidenceRefs=evidence,
        sideEffect="NONE",
        retryable=False,
        contentTrust="TRUSTED",
        redacted=bool(derived_sensitivities),
        sensitivity="PERSONAL" if "PERSONAL" in derived_sensitivities else (
            "EXTERNAL_CONTENT" if derived_sensitivities else "NONE"
        ),
        criterionEvidence=[
            {
                "criterion": criterion,
                "status": "COMPLETE" if evidence else "INCOMPLETE",
                "evidenceRefs": evidence,
                "note": "已基于前序证据形成结论。" if evidence else "缺少前序证据，无法形成结论。",
            }
            for criterion in (_criterion_text(item) for item in request.completionCriteria)
            if _is_conclusion_criterion(criterion) and not _is_mutation_criterion(criterion)
        ],
    )


def _contains_sensitive_body(value: str) -> bool:
    return any(marker in value for marker in ("长期摘要：", "长期记忆摘要", "网页搜索结果："))


def _derived_sensitive_fallback(tool_ref: str) -> str:
    if tool_ref == "memory.snapshot":
        return "个人记忆正文已省略；仅保留结构化记忆索引摘要。"
    return "外部网页正文与搜索片段已省略；仅保留结构化来源摘要。"


def _release_check(request: ExecuteStepRequest) -> ToolResult:
    workspace_id = _workspace_id(request.input)
    if workspace_id is None:
        return _tool_error("发布检查需要已授权的 workspaceId，不能用工作摘要代替真实测试。", "WORKSPACE_REQUIRED")
    try:
        workspace = get_workspace(request.userId, workspace_id)
    except Exception as exc:
        return _tool_error(f"无法读取授权工作区：{exc}", "WORKSPACE_UNAVAILABLE")
    commands = _release_commands(request.input, Path(str(workspace["rootPath"])))
    if not commands:
        return _tool_error("工作区未配置可识别的发布测试命令。", "RELEASE_COMMANDS_REQUIRED")

    results: list[dict[str, Any]] = []
    for command in commands[:5]:
        risk = classify_command(command)
        if risk.policy != "allow" or risk.requires_confirmation or risk.risk_level != "compute":
            return _tool_error(f"发布命令必须属于已批准的测试/构建白名单：{command}", "COMMAND_NOT_ALLOWED")
        try:
            raw = run_command(
                request.userId,
                workspace_id,
                command,
                confirmed=False,
                second_confirmed=False,
                timeout_seconds=_command_timeout(request),
            )
        except Exception as exc:
            raw = {"ok": False, "command": command, "summary": str(exc), "errorCode": "COMMAND_ERROR"}
        results.append({**raw, "command": command})

    refs = [f"cli-command://{item['id']}" for item in results if item.get("id")]
    failed = [item for item in results if not item.get("ok")]
    lines = [
        f"{'通过' if item.get('ok') else '失败'}：{item['command']} — {str(item.get('summary') or item.get('errorCode') or '')[:800]}"
        for item in results
    ]
    summary = ("发布检查全部通过。" if not failed else f"发布检查有 {len(failed)} 项失败。") + "\n" + "\n".join(lines)
    criterion_evidence = _release_criterion_evidence(request.completionCriteria, refs, not failed, results)
    return ToolResult(
        status="SUCCESS" if not failed else "PARTIAL",
        summary=summary[:8_000],
        dataRef=f"release-check://{request.stepRunId}",
        evidenceRefs=refs,
        retryable=any(str(item.get("errorCode") or "").upper() == "TIMEOUT" for item in failed),
        criterionEvidence=criterion_evidence,
        externalStateHashes={f"workspace:{workspace_id}": compute_workspace_state(request.userId, workspace_id)},
    )


def _git_status(request: ExecuteStepRequest) -> ToolResult:
    workspace_id = _workspace_id(request.input)
    if workspace_id is None:
        return _tool_error("Git 巡检需要已授权的 workspaceId。", "WORKSPACE_REQUIRED")
    try:
        workspace = get_workspace(request.userId, workspace_id)
        result = run_command(request.userId, workspace_id, "git status --short", timeout_seconds=_command_timeout(request))
    except Exception as exc:
        return _tool_error(f"Git 状态检查失败：{exc}", "GIT_STATUS_FAILED")
    ref = f"cli-command://{result['id']}" if result.get("id") else f"workspace://{workspace_id}/git-status"
    ok = bool(result.get("ok"))
    raw_summary = str(result.get("summary") or "")
    summary = raw_summary or ("Git 工作区干净。" if ok else "Git 状态读取失败。")
    result["gitClean"] = ok and not raw_summary.strip()
    return ToolResult(
        status="SUCCESS" if ok else "ERROR",
        summary=summary[:8_000],
        dataRef=ref,
        evidenceRefs=[ref],
        retryable=False,
        errorCode=None if ok else str(result.get("errorCode") or "GIT_STATUS_FAILED"),
        criterionEvidence=_criterion_evidence(request.completionCriteria, request.toolRef, summary, [ref], ok, result),
        externalStateHashes={f"workspace:{workspace_id}": compute_workspace_state(request.userId, workspace_id)},
    )


def _docs_check(request: ExecuteStepRequest) -> ToolResult:
    workspace_id = _workspace_id(request.input)
    if workspace_id is None:
        return _tool_error("文档巡检需要已授权的 workspaceId。", "WORKSPACE_REQUIRED")
    try:
        workspace = get_workspace(request.userId, workspace_id)
        root = Path(str(workspace["rootPath"])).expanduser().resolve()
        candidates = sorted({
            path for pattern in ("README*", "docs/**/*.md", "*.md")
            for path in root.glob(pattern)
            if path.is_file() and root in path.resolve().parents
        })[:200]
        gaps: list[str] = []
        marker = re.compile(r"\b(?:TODO|TBD|FIXME)\b|待补充|待完善", re.IGNORECASE)
        for path in candidates[:50]:
            try:
                content = path.read_text(encoding="utf-8")[:200_000]
            except (OSError, UnicodeError):
                gaps.append(f"{path.relative_to(root).as_posix()}: 无法读取")
                continue
            for line_no, line in enumerate(content.splitlines(), 1):
                if marker.search(line):
                    gaps.append(f"{path.relative_to(root).as_posix()}:{line_no}")
                    if len(gaps) >= 50:
                        break
            if len(gaps) >= 50:
                break
    except Exception as exc:
        return _tool_error(f"文档巡检失败：{exc}", "DOCS_CHECK_FAILED")
    refs = [f"workspace://{workspace_id}/{path.relative_to(root).as_posix()}" for path in candidates[:50]]
    ok = bool(candidates) and not gaps
    if not candidates:
        summary = "没有发现 README 或 docs Markdown 文档。"
    elif gaps:
        summary = f"检查了 {len(candidates)} 个文档文件，发现 {len(gaps)} 个待补充标记或读取缺口：" + "、".join(gaps[:20])
    else:
        summary = f"检查了 {len(candidates)} 个文档文件，未发现 TODO/TBD/FIXME 或待补充标记。"
    return ToolResult(
        status="SUCCESS" if ok else "PARTIAL",
        summary=summary,
        dataRef=f"workspace://{workspace_id}/docs",
        evidenceRefs=refs,
        criterionEvidence=_criterion_evidence(request.completionCriteria, request.toolRef, summary, refs, ok, {}),
        externalStateHashes={f"workspace:{workspace_id}": compute_workspace_state(request.userId, workspace_id)},
    )


def _tool_error(summary: str, code: str) -> ToolResult:
    return ToolResult(status="ERROR", summary=summary[:8_000], retryable=False, errorCode=code)


def _workspace_id(values: dict[str, Any]) -> int | None:
    try:
        value = int(values.get("workspaceId"))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _release_commands(values: dict[str, Any], root: Path) -> list[str]:
    configured = values.get("commands")
    if isinstance(configured, list):
        commands = [str(item).strip() for item in configured if str(item).strip()]
        if commands:
            return commands
    detected: list[str] = []
    if (root / "pom.xml").is_file():
        detected.append("mvn test")
    if (root / "package.json").is_file():
        try:
            package = json.loads((root / "package.json").read_text(encoding="utf-8"))
            scripts = package.get("scripts") if isinstance(package, dict) else {}
            scripts = scripts if isinstance(scripts, dict) else {}
        except (OSError, UnicodeError, json.JSONDecodeError):
            scripts = {}
        if "test" in scripts:
            detected.append("npm test")
        elif "test:unit" in scripts:
            detected.append("npm run test:unit")
        if "build" in scripts:
            detected.append("npm run build")
    if (root / "pyproject.toml").is_file() or (root / "requirements.txt").is_file():
        detected.append("pytest -q")
    return detected


def _command_timeout(request: ExecuteStepRequest) -> int:
    millis = int(request.remainingBudget.get("wallTimeMillis") or 60_000)
    return max(1, min(120, millis // 1_000))


def _criterion_evidence(
    criteria: list[Any],
    tool_ref: str,
    summary: str,
    evidence_refs: list[Any],
    ok: bool,
    raw_result: dict[str, Any],
) -> list[dict[str, Any]]:
    explicit = raw_result.get("criterionEvidence")
    if isinstance(explicit, list):
        explicit_mappings = [item for item in explicit if isinstance(item, dict)]
        if explicit_mappings:
            return explicit_mappings
    result: list[dict[str, Any]] = []
    for raw in criteria:
        criterion = _criterion_text(raw)
        if _is_mutation_criterion(criterion):
            continue
        category = _criterion_category(criterion)
        supported = (
            (tool_ref == "project.inspect" and category == "project")
            or (tool_ref == "research.search" and category == "research")
            or (tool_ref == "workspace.git_status" and category == "git")
            or (tool_ref == "workspace.docs_check" and category == "docs")
        )
        if not supported:
            continue
        criterion_ok = ok
        if tool_ref == "research.search" and _requires_source_evidence(criterion):
            results = raw_result.get("results")
            criterion_ok = criterion_ok and isinstance(results, list) and any(
                _safe_source_url(item) for item in results
            )
        if tool_ref == "workspace.git_status" and _requires_clean_git(criterion):
            criterion_ok = criterion_ok and bool(raw_result.get("gitClean"))
        status = "COMPLETE" if criterion_ok and evidence_refs else "INCOMPLETE"
        result.append({
            "criterion": criterion,
            "status": status,
            "evidenceRefs": evidence_refs,
            "note": summary[:1_000],
        })
    return result


def _is_mutation_criterion(value: str) -> bool:
    lowered = value.lower()
    research_observation = (
        any(word in lowered for word in [
            "资料", "来源", "引用", "调研", "检索", "搜索", "research", "source",
        ])
        and any(word in lowered for word in [
            "去重", "保留", "检索", "搜索", "收集", "对比", "deduplic", "retain", "search",
        ])
        and not any(word in lowered for word in [
            "任务", "待办", "文件", "日程", "记录", "数据库", "patch", "todo", "file", "calendar", "database",
        ])
    )
    if research_observation:
        return False
    return any(word in lowered for word in [
        "创建", "新增", "写入", "修改", "删除", "应用 patch", "apply patch",
        "create ", "add ", "write ", "update ", "delete ", "任务草稿", "todo draft",
    ])


def _requires_clean_git(value: str) -> bool:
    lowered = value.lower()
    return any(word in lowered for word in [
        "工作区干净", "无未提交", "无修改", "没有修改", "git clean", "clean working tree", "no changes",
    ])


def _requires_source_evidence(value: str) -> bool:
    lowered = value.lower()
    return any(word in lowered for word in [
        "来源", "引用", "资料", "source", "citation", "reference",
    ])


def _is_conclusion_criterion(value: str) -> bool:
    lowered = value.lower()
    return any(word in lowered for word in [
        "给出", "结论", "建议", "报告", "周报", "冲突说明",
        "summary", "recommendation", "report", "conflict note",
    ])


def _release_criterion_evidence(
    criteria: list[Any], refs: list[Any], all_passed: bool, results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    note = "全部授权发布命令通过。" if all_passed else "至少一个授权发布命令失败，不能判定发布条件满足。"
    for raw in criteria:
        criterion = _criterion_text(raw)
        if _is_mutation_criterion(criterion):
            continue
        if _criterion_category(criterion) not in {"tests", "release"}:
            continue
        if _is_conclusion_criterion(criterion):
            continue
        mapped.append({
            "criterion": criterion,
            "status": "COMPLETE" if all_passed and refs else "INCOMPLETE",
            "evidenceRefs": refs,
            "note": note,
        })
    return mapped


def _criterion_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("criterion") or value.get("title") or value.get("description") or value)[:1_000]
    return str(value)[:1_000]


def _criterion_category(value: str) -> str:
    lowered = value.lower()
    if any(word in lowered for word in ["测试", "test", "构建", "build", "检查通过"]):
        return "tests"
    if any(word in lowered for word in ["发布", "go/no-go", "go no go", "阻塞", "release"]):
        return "release"
    if any(word in lowered for word in ["文档", "readme", "docs"]):
        return "docs"
    if any(word in lowered for word in ["git", "代码状态", "工作区"]):
        return "git"
    if any(word in lowered for word in [
        "引用", "来源", "资料", "调研", "对比", "检索", "搜索", "research", "search",
    ]):
        return "research"
    if any(word in lowered for word in ["项目", "任务", "待办", "目标", "进度"]):
        return "project"
    return "other"


def _evidence(result: dict[str, object], request: ExecuteStepRequest) -> list[Any]:
    refs: list[Any] = [f"tool-result://{request.stepRunId}"]
    raw = result.get("sources") or result.get("evidenceRefs")
    if isinstance(raw, list):
        refs.extend(raw[:49])
    if request.toolRef == "research.search" and isinstance(result.get("results"), list):
        refs.extend(
            value for value in (_safe_source_url(item) for item in result["results"][:49]) if value
        )
    deduplicated: list[Any] = []
    seen: set[str] = set()
    for ref in refs:
        key = json.dumps(ref, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(ref)
    return deduplicated[:50]


def _safe_source_url(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    try:
        parsed = urlsplit(str(value.get("url") or ""))
    except ValueError:
        return None
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme not in {"http", "https"} or not SAFE_SOURCE_HOST.fullmatch(host):
        return None
    try:
        port = parsed.port
    except ValueError:
        return None
    netloc = f"{host}:{port}" if port is not None else host
    return urlunsplit((parsed.scheme, netloc, "", "", ""))[:512]


def _security_flags(content: str) -> list[str]:
    compact = content.lower()
    flags: list[str] = []
    if re.search(r"ignore (all |the )?(previous|system)|忽略.{0,8}(规则|指令|系统)", compact):
        flags.append("PROMPT_INJECTION_PATTERN")
    if any(value in compact for value in ["上传本地", "upload local", "secret", "api key"]):
        flags.append("DATA_EXFILTRATION_PATTERN")
    return flags
