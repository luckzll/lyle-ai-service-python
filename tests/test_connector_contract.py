from app.connectors.github_connector import GitHubConnector
from app.connectors.local_docs import LocalDocsConnector
from app.connectors.registry import connector_search, list_connectors
from app.core.config import settings


def test_connector_registry_exposes_stable_status_shape():
    result = list_connectors()

    assert "items" in result
    assert {"local_docs", "github"} <= {item["name"] for item in result["items"]}
    for item in result["items"]:
        assert {"name", "label", "status", "capabilities", "message"} <= set(item)


def test_local_docs_search_and_get_are_source_grounded(tmp_path, monkeypatch):
    doc = tmp_path / "V3-note.md"
    doc.write_text("Lyle V3 需要项目空间、目标计划和轻量连接器。", encoding="utf-8")
    monkeypatch.setattr(settings, "connector_local_docs_roots", str(tmp_path))
    monkeypatch.setattr(settings, "connector_local_docs_max_files", 20)

    connector = LocalDocsConnector()
    search = connector.search({"q": "连接器", "limit": 5})
    detail = connector.get(str(doc))

    assert search["ok"] is True
    assert search["items"][0]["id"] == str(doc)
    assert search["sources"][0]["ref"] == str(doc)
    assert detail["ok"] is True
    assert "项目空间" in detail["items"][0]["content"]


def test_local_docs_rejects_resource_outside_allowed_root(tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    denied = tmp_path / "denied"
    allowed.mkdir()
    denied.mkdir()
    outside_doc = denied / "secret.md"
    outside_doc.write_text("not allowed", encoding="utf-8")
    monkeypatch.setattr(settings, "connector_local_docs_roots", str(allowed))

    result = LocalDocsConnector().get(str(outside_doc))

    assert result["ok"] is False
    assert result["error"]["code"] == "NOT_FOUND"


def test_github_without_token_returns_auth_error(monkeypatch):
    monkeypatch.setattr(settings, "connector_github_token", "")

    status = GitHubConnector().status()
    result = GitHubConnector().search({"q": "bug", "repo": "owner/repo"})

    assert status.status == "needs_auth"
    assert result["ok"] is False
    assert result["error"]["code"] == "AUTH_REQUIRED"
    assert result["items"] == []


def test_connector_preview_create_uses_pending_operation(monkeypatch):
    captured = {}

    def fake_pending_operation(**kwargs):
        captured.update(kwargs)
        return {
            "ok": True,
            "content": "已生成待确认操作",
            "requiresConfirmation": True,
            "operation": {"id": 1, "status": "pending"},
        }

    monkeypatch.setattr("app.connectors.local_docs.create_pending_operation", fake_pending_operation)

    result = LocalDocsConnector().preview_create("user-1", {"title": "V3 复盘", "content": "body"})

    assert result["requiresConfirmation"] is True
    assert captured["user_id"] == "user-1"
    assert captured["operation_type"] == "connector.local_docs.create"
    assert captured["payload"]["title"] == "V3 复盘"


def test_registry_search_routes_to_connector(tmp_path, monkeypatch):
    doc = tmp_path / "route.md"
    doc.write_text("connector registry route", encoding="utf-8")
    monkeypatch.setattr(settings, "connector_local_docs_roots", str(tmp_path))

    result = connector_search("local_docs", {"q": "registry"})

    assert result["ok"] is True
    assert result["items"][0]["title"] == "route.md"
