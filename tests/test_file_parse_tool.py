from app.core.config import settings
from app.rag.store import remember_storage_object_owner
from app.tools import file_parse_tool
from app.tools.file_parse_tool import parse_file_from_question


def test_file_parse_rejects_path_outside_whitelist(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "tool_file_parse_allowed_dirs", str(tmp_path))

    result = parse_file_from_question("帮我解析 /etc/passwd 的内容")

    assert result["ok"] is False
    assert "白名单" in result["content"]


def test_file_parse_rejects_file_url_outside_whitelist(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "tool_file_parse_allowed_dirs", str(tmp_path))

    result = parse_file_from_question("帮我解析 file:///etc/hosts")

    assert result["ok"] is False
    assert "白名单" in result["content"]


def test_file_parse_allows_path_inside_whitelist(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "tool_file_parse_allowed_dirs", str(tmp_path))
    path = tmp_path / "note.txt"
    path.write_text("白名单内的文件内容。", encoding="utf-8")

    result = parse_file_from_question(f"帮我解析 {path}")

    assert result["ok"] is True
    assert "白名单内的文件内容" in result["content"]


def test_file_parse_rejects_dot_dot_escape(monkeypatch, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("外部机密内容", encoding="utf-8")
    monkeypatch.setattr(settings, "tool_file_parse_allowed_dirs", str(allowed))

    result = parse_file_from_question(f"帮我解析 {allowed}/../secret.txt")

    assert result["ok"] is False


def test_file_parse_rejects_s3_object_without_user_ownership():
    result = parse_file_from_question("帮我解析 s3://knowledge-files/200/doc.pdf", user_id="user-2")

    assert result["ok"] is False
    assert "对象不属于当前用户" in result["content"]


def test_file_parse_allows_s3_object_owned_by_user(monkeypatch):
    remember_storage_object_owner("user-1", "knowledge-files", "200/doc.txt")
    monkeypatch.setattr(
        file_parse_tool,
        "load_document_text",
        lambda storage_path: ("doc.txt", "用户自己的对象内容。"),
    )

    result = parse_file_from_question("帮我解析 s3://knowledge-files/200/doc.txt", user_id="user-1")

    assert result["ok"] is True
    assert "用户自己的对象内容" in result["content"]
