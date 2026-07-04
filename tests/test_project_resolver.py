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


def test_resolve_project_uses_conversation_link_for_followup(monkeypatch):
    monkeypatch.setattr(
        "app.projects.project_resolver.list_projects",
        lambda user_id, status="active", limit=50: [{"id": 9, "name": "V3测试项目", "slug": "v3-test"}],
    )
    monkeypatch.setattr(
        "app.projects.project_resolver.list_project_links",
        lambda user_id, project_id, target_type=None, limit=20: [{
            "projectId": project_id,
            "targetType": "conversation",
            "targetId": "c1",
            "confidence": 0.92,
        }],
    )

    result = resolve_project("u1", "c1", "帮我基于这个目标拆一个三步推进计划。")

    assert result["projectId"] == 9
    assert result["projectName"] == "V3测试项目"
    assert result["reason"] == "沿用当前会话已关联项目"
