from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Lock
from time import sleep

from app.memory import profile_store


def test_profile_schema_initialization_is_serialized(monkeypatch):
    guard = Lock()
    connection_count = 0

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, _query):
            sleep(0.01)

    class Connection:
        def cursor(self):
            return Cursor()

        def commit(self):
            return None

    @contextmanager
    def fake_postgres_connection():
        nonlocal connection_count
        with guard:
            connection_count += 1
        yield Connection()

    monkeypatch.setattr(profile_store, "_postgres_ready", False)
    monkeypatch.setattr(profile_store, "postgres_connection", fake_postgres_connection)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(profile_store._ensure_postgres_schema) for _ in range(2)]
        for future in futures:
            future.result()

    assert connection_count == 1
