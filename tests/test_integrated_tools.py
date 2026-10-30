import shlex
from unittest.mock import Mock

import pytest
from pydantic import ValidationError
from langchain_core.messages import AIMessageChunk

from app.tools import integrated_tools as bridge
from app.api import local_workspace


def tool(tool_id=7, risk="read_only"):
    return {"id": tool_id, "toolName": "lookup", "description": "Find issues", "riskLevel": risk,
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}}


def test_selection_never_admits_identity_or_model_approval():
    for value in [{"userId": "other"}, {"confirmed": True}, {"workspaceId": -1}, {"mcpToolIds": list(range(1, 22))}]:
        with pytest.raises(ValidationError):
            bridge.ToolSelection.model_validate(value)


def test_empty_selection_is_off_without_touching_database(monkeypatch):
    catalog = Mock(side_effect=AssertionError("No DB needed"))
    monkeypatch.setattr(bridge.agent_mcp, "catalog", catalog)
    assert bridge.selected_mcp("user", bridge.ToolSelection()) == {}
    catalog.assert_not_called()


def test_catalog_filters_user_selection_and_read_mode(monkeypatch):
    monkeypatch.setattr(bridge.agent_mcp, "catalog", lambda user: {"mcp_7": tool(), "mcp_8": tool(8, "write_high")})
    assert [t["function"]["name"] for t in bridge.schemas("u", bridge.ToolSelection(mcpToolIds=[7, 8]), read_only=True)] == ["mcp_7"]
    assert bridge.selected_mcp("u", bridge.ToolSelection(mcpToolIds=[99])) == {}


def test_cli_requires_explicit_tenant_owned_workspace(monkeypatch):
    run = Mock()
    monkeypatch.setattr(bridge, "run_command", run)
    with pytest.raises(ValueError, match="绑定"):
        bridge.invoke("u", None, "project_git_status", {}, bridge.ToolSelection())
    monkeypatch.setattr(bridge, "get_workspace", Mock(side_effect=ValueError("not owned")))
    with pytest.raises(ValueError, match="not owned"):
        bridge.schemas("u", bridge.ToolSelection(workspaceId=9))
    run.assert_not_called()


def test_cli_arguments_are_literals_and_never_expand_to_shell_or_paths():
    query = '--pre=evil $(whoami); & dir "example"'
    command = bridge.command_for("project_search", {"query": query})
    assert shlex.split(command)[-2:] == [query, "."]
    for name, args in [("project_search", {"query": "x", "path": "../"}), ("project_git_status", {"confirmed": True}), ("shell", {"command": "rm"})]:
        with pytest.raises(ValueError):
            bridge.command_for(name, args)


def test_cli_previews_before_execution_and_revalidates_binding(monkeypatch):
    workspace = {"id": 3, "name": "example", "rootPath": "/authorized", "status": "active"}
    monkeypatch.setattr(bridge, "preview_command", lambda *args: {"permission": {"allowed": True}, "risk": {"riskLevel": "read_only"}, "workspace": workspace})
    pending, run = Mock(return_value={"requiresConfirmation": True}), Mock(return_value={"ok": True, "summary": "clean"})
    monkeypatch.setattr(bridge, "create_pending_operation", pending)
    monkeypatch.setattr(bridge, "run_command", run)
    with bridge.selection_scope(bridge.ToolSelection(workspaceId=3)):
        assert bridge.invoke("u", "c", "project_git_status", {}, bridge.current_selection())["requiresConfirmation"]
    run.assert_not_called()
    payload = pending.call_args.kwargs["payload"]
    monkeypatch.setattr(bridge, "get_workspace", lambda *args: {**workspace, "rootPath": "/changed"})
    assert not bridge.execute_confirmed("u", {"payload": payload})["ok"]
    run.assert_not_called()
    monkeypatch.setattr(bridge, "get_workspace", lambda *args: workspace)
    assert bridge.execute_confirmed("u", {"payload": payload})["ok"]
    assert run.call_args.kwargs["confirmed"]


def test_selection_scope_does_not_leak_between_users():
    assert bridge.current_selection() is None
    with bridge.selection_scope(bridge.ToolSelection(workspaceId=3)):
        assert bridge.current_selection().workspaceId == 3
        with bridge.selection_scope(bridge.ToolSelection()):
            assert bridge.current_selection().workspaceId is None
        assert bridge.current_selection().workspaceId == 3
    assert bridge.current_selection() is None


def test_recommended_github_credential_is_unique_per_account():
    from app.tools.integration_presets import presets_for_user
    first = next(item for item in presets_for_user("42") if item["id"] == "github")
    second = next(item for item in presets_for_user("43") if item["id"] == "github")
    assert first["secretRef"] == "env:LYLE_GITHUB_MCP_42"
    assert second["secretRef"] == "env:LYLE_GITHUB_MCP_43"
    assert "{{secret_ref}}" in first["config"]["headers"]["Authorization"]


def test_unavailable_database_does_not_hide_local_tools_or_recommendations(monkeypatch):
    monkeypatch.setattr(bridge.agent_mcp, "catalog", Mock(side_effect=RuntimeError("PRIVATE_DB_FAILURE")))
    result = bridge.catalog("42")
    assert result["availabilityError"]
    assert "PRIVATE_DB_FAILURE" not in str(result)
    assert result["tools"] == [] and result["cli"]


@pytest.mark.parametrize("enabled,expected", [("false", False), ("true", True)])
def test_deployment_can_withhold_unverified_browser_recipe(monkeypatch, enabled, expected):
    from app.tools.integration_presets import presets_for_user
    monkeypatch.setenv("LYLE_ENABLE_PLAYWRIGHT_PRESET", enabled)
    presets = presets_for_user("42")
    assert any(item["id"] == "playwright" for item in presets) is expected
    assert {"github", "fetch", "ripgrep", "github-cli"} <= {item["id"] for item in presets}


def test_local_project_can_call_selected_mcp_without_file_connection(monkeypatch):
    monkeypatch.setattr(bridge.agent_mcp, "catalog", lambda user: {"mcp_7": tool()})
    observed = []
    class Model:
        def bind_tools(self, tools, **kwargs):
            observed.extend(t["function"]["name"] for t in tools)
            return self
        def stream(self, *args, **kwargs):
            yield AIMessageChunk(content="", tool_calls=[{"name": "mcp_7", "args": {}, "id": "call1"}])
    monkeypatch.setattr(local_workspace, "build_chat_model_for_task", lambda *args, **kwargs: Model())
    request = local_workspace.WorkspaceStep(user_id="u", turn_id="t", messages=[{"role": "user", "content": "查 Issue"}],
        workspace_connected=False, toolSelection={"mcpToolIds": [7]})
    frames = "".join(local_workspace.frames(request))
    assert observed == ["mcp_7"]
    assert 'event: result' in frames and 'mcp_7' in frames


def test_legacy_local_histories_can_be_read_after_deselecting_external_tools():
    local_workspace.WorkspaceStep(user_id="u", turn_id="t", toolSelection={"mcpToolIds": []}, messages=[
        {"role": "user", "content": "query"},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "mcp_7", "id": "call1", "args": {}}]},
        {"role": "tool", "tool_call_id": "call1", "content": "result"},
    ])


@pytest.mark.parametrize("question,expected", [("运行测试", True), ("列出 open issues", True), ("你好", False), ("翻译：运行测试", False)])
def test_explicit_tools_reach_the_planner_without_legacy_keywords(monkeypatch, question, expected):
    from app.graphs import multi_agent_graph as graph
    monkeypatch.setattr(graph, "run_supervisor_agent", lambda *args: {"intent": "chat", "use_tools": False, "next_agents": ["writer_agent"]})
    state = {"user_id": "u", "conversation_id": "c", "question": question, "knowledge_base_ids": [], "agent_trace": []}
    with bridge.selection_scope(bridge.ToolSelection(workspaceId=3)):
        result = graph.supervisor_node(state)
    assert result["supervisor"]["use_tools"] is expected
    assert result["supervisor"]["next_agents"].count("tool_agent") == int(expected)


def test_selected_tools_report_unavailable_models_instead_of_silent_legacy_fallback(monkeypatch):
    from app.agents import tool_agent, native_tool_loop
    monkeypatch.setattr(native_tool_loop, "available", lambda: False)
    monkeypatch.setattr(tool_agent, "decide_tool_use", Mock(side_effect=AssertionError("no silent fallback")))
    with bridge.selection_scope(bridge.ToolSelection(workspaceId=3)):
        result = tool_agent.run_tool_agent("u", "c", "运行测试", True)
    assert result["decision"]["router"] == "selected_tools_unavailable"
    assert not result["tool_results"]
