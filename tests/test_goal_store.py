from contextlib import contextmanager
from datetime import datetime, timezone

from app.core.config import settings
from app.goals import goal_store


class _Result:
    def __init__(self, one=None, many=None):
        self._one = one
        self._many = many or []

    def fetchone(self):
        return self._one

    def fetchall(self):
        return self._many


class _Conn:
    def __init__(self, results):
        self._results = list(results)
        self.committed = False

    def execute(self, *_args, **_kwargs):
        return self._results.pop(0)

    def commit(self):
        self.committed = True


def test_create_goal_returns_milestones_without_recursive_lookup(monkeypatch):
    now = datetime(2026, 7, 4, tzinfo=timezone.utc)
    goal_row = (42, 9, "推进 V3 测试项目", None, "week", "active", 0.0, None, now, now)
    connections = [
        _Conn([_Result(one=goal_row), _Result()]),
        _Conn([_Result(one=(42,))]),
        _Conn([_Result(many=[])]),
    ]

    @contextmanager
    def fake_postgres_connection():
        yield connections.pop(0)

    monkeypatch.setattr(settings, "postgres_enabled", True)
    monkeypatch.setattr(goal_store, "postgres_connection", fake_postgres_connection)
    monkeypatch.setattr(goal_store, "get_project", lambda _user_id, _project_id: {"id": _project_id})

    goal = goal_store.create_goal(
        "user-1",
        {
            "projectId": 9,
            "title": "推进 V3 测试项目",
            "horizon": "week",
        },
    )

    assert goal["id"] == 42
    assert goal["projectId"] == 9
    assert goal["milestones"] == []
    assert connections == []
