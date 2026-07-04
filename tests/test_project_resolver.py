from app.projects.project_resolver import resolve_project


def test_resolve_project_prefers_explicit_project_id(monkeypatch):
    monkeypatch.setattr(
        "app.projects.project_resolver.get_project",
        lambda user_id, project_id: {"id": project_id, "name": "T2测试项目"},
    )
    linked = {}

    def fake_link(user_id, project_id, payload):
        linked.update({"user_id": user_id, "project_id": project_id, "payload": payload})
        return {"id": 1}

    monkeypatch.setattr("app.projects.project_resolver.create_project_link", fake_link)

    result = resolve_project("u1", "c1", "新增一个目标", project_id=9)

    assert result["projectId"] == 9
    assert result["projectName"] == "T2测试项目"
    assert result["confidence"] == 1.0
    assert result["requiresClarification"] is False
    assert linked["project_id"] == 9
