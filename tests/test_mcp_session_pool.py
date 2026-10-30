import sys
from pathlib import Path

from app.mcp.client_manager import run_mcp_tool_call
from app.mcp import session_pool


def test_state_survives_calls_but_never_crosses_tenants_or_conversations():
    server = {"id": 700, "serverType": "stdio", "configRef": {"command": sys.executable,
              "args": [str(Path(__file__).parent / "fixtures" / "stateful_mcp_server.py")]}}
    def value(user, conversation, config=server):
        result = run_mcp_tool_call(config, "next_value", {}, session_scope=(user, conversation))
        assert result["ok"]
        return result["content"][0]["text"]
    try:
        assert value("user-a", "conversation-1") == "1"
        assert value("user-a", "conversation-1") == "2"
        assert value("user-a", "conversation-2") == "1"
        assert value("user-b", "conversation-1") == "1"
        assert value("user-a", "conversation-1") == "3"
        changed = {**server, "configRef": {**server["configRef"], "env": {"LYLE_TEST_CONFIG": "changed"}}}
        assert value("user-a", "conversation-1", changed) == "1"
        session_pool.close_server(700)
    finally:
        session_pool.close_all()
    assert not session_pool._actors
