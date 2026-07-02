from contextlib import contextmanager
from typing import Iterator

from psycopg import Connection
from psycopg_pool import ConnectionPool

from app.core.config import settings


_postgres_pool: ConnectionPool | None = None


def get_postgres_pool() -> ConnectionPool:
    global _postgres_pool
    if _postgres_pool is None:
        _postgres_pool = ConnectionPool(
            conninfo=settings.postgres_dsn,
            min_size=settings.postgres_pool_min_size,
            max_size=settings.postgres_pool_max_size,
            open=False,
        )
        _postgres_pool.open()
    return _postgres_pool


@contextmanager
def postgres_connection() -> Iterator[Connection]:
    with get_postgres_pool().connection() as conn:
        yield conn


def close_postgres_pool() -> None:
    global _postgres_pool
    if _postgres_pool is not None:
        _postgres_pool.close()
        _postgres_pool = None
