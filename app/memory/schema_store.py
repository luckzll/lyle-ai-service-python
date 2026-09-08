"""Durable per-user build queue and atomic publication of bounded schema history."""
from contextlib import contextmanager
from functools import wraps
from copy import deepcopy
from threading import RLock
from time import time
from uuid import uuid4

from psycopg.types.json import Jsonb

from app.core.config import settings
from app.core.database import postgres_connection


_lock = RLock()
_states: dict[str, dict] = {}
_ready = False


def schema_dirty(function):
    @wraps(function)
    def wrapped(user_id, *args, **kwargs):
        result = function(user_id, *args, **kwargs)
        if settings.memory_schema_enabled:
            request_build(user_id)
        return result
    return wrapped


DDL = """
CREATE TABLE IF NOT EXISTS memory_schema_state (
  user_id TEXT PRIMARY KEY,
  state_json JSONB NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_memory_schema_queue
  ON memory_schema_state ((state_json->>'status'), updated_at);
"""


def _ensure_schema():
    global _ready
    if _ready:
        return
    with _lock:
        if not _ready:
            with postgres_connection() as conn:
                conn.execute(DDL)
                conn.commit()
            _ready = True


def _empty():
    return {"status": "missing", "revision": 0, "version": 0, "desiredHash": "", "snapshots": [], "attempts": 0}


@contextmanager
def _edit(user_id: str):
    if settings.postgres_enabled:
        _ensure_schema()
        with postgres_connection() as conn:
            with conn.transaction():
                conn.execute("INSERT INTO memory_schema_state(user_id,state_json) VALUES (%s,%s) ON CONFLICT DO NOTHING", (user_id, Jsonb(_empty())))
                state = conn.execute("SELECT state_json FROM memory_schema_state WHERE user_id=%s FOR UPDATE", (user_id,)).fetchone()[0]
                yield state
                conn.execute("UPDATE memory_schema_state SET state_json=%s,updated_at=now() WHERE user_id=%s", (Jsonb(state), user_id))
    else:
        with _lock:
            state = deepcopy(_states.get(user_id, _empty()))
            yield state
            _states[user_id] = state


def read_state(user_id: str) -> dict:
    if settings.postgres_enabled:
        _ensure_schema()
        with postgres_connection() as conn:
            row = conn.execute("SELECT state_json FROM memory_schema_state WHERE user_id=%s", (user_id,)).fetchone()
        return row[0] if row else _empty()
    with _lock:
        return deepcopy(_states.get(user_id, _empty()))


def request_build(user_id: str, fingerprint: str = "", *, force: bool = False) -> dict:
    with _edit(user_id) as state:
        changed = not fingerprint or state.get("desiredHash") != fingerprint
        if changed or force:
            if force:
                state['forceFullBuild'] = True
            state["revision"] += 1
            state["desiredHash"] = fingerprint
            state["attempts"] = 0
            state["errorCode"] = ""
            if state["status"] != "building":
                state["status"] = "pending"
            state["requestedAt"] = time()
        return deepcopy(state)


def claim_build(user_id: str) -> dict | None:
    with _edit(user_id) as state:
        if state["status"] == "building" and state.get("leaseUntil", 0) > time():
            return None
        if state["status"] not in {"pending", "building"}:
            return None
        state.update(status="building", buildId=str(uuid4()), leaseUntil=time() + settings.memory_schema_build_lease_seconds,
                     claimedRevision=state["revision"], attempts=state.get("attempts", 0) + 1)
        return deepcopy(state)


def finish_build(user_id: str, claim: dict, document: dict | None, *, error_code: str = "") -> bool:
    with _edit(user_id) as state:
        if state.get("buildId") != claim.get("buildId") or state.get("status") != "building":
            return False  # Another worker reclaimed the expired lease.
        if state["revision"] != claim["claimedRevision"]:
            state["status"] = "pending"
            return False  # Never publish inputs superseded during model execution.
        if error_code or document is None:
            state.update(status="failed", errorCode=error_code or "BUILD_FAILED", leaseUntil=0)
            return False
        version = state["version"] + 1
        snapshot = {**document, "version": version, "buildId": claim["buildId"], "publishedAt": time()}
        state.update(status="ready", version=version, errorCode="", leaseUntil=0, forceFullBuild=False,
                     snapshots=[snapshot, *state.get("snapshots", [])][:settings.memory_schema_history_limit])
        return True


def queued_users(limit: int = 5) -> list[str]:
    if settings.postgres_enabled:
        _ensure_schema()
        with postgres_connection() as conn:
            rows = conn.execute("""SELECT user_id FROM memory_schema_state
                WHERE state_json->>'status'='pending'
                   OR (state_json->>'status'='building' AND (state_json->>'leaseUntil')::double precision <= %s)
                ORDER BY updated_at LIMIT %s""", (time(), max(1, min(limit, 20)))).fetchall()
        return [row[0] for row in rows]
    with _lock:
        return [user for user, state in _states.items() if state["status"] == "pending"
                or (state["status"] == "building" and state.get("leaseUntil", 0) <= time())][:limit]


def purge_schema_history(user_id: str) -> None:
    # A privacy purge also removes all retained derived text for this user.
    # Remaining confirmed sources can rebuild; version numbers never go backwards.
    with _edit(user_id) as state:
        state.update(snapshots=[], status="pending", desiredHash="", buildId="", leaseUntil=0)
        state["revision"] += 1


def build_status(state: dict) -> dict:
    latest = next(iter(state.get("snapshots", [])), {})
    return {"status": state["status"], "version": state["version"], "etag": latest.get("etag", ""),
            "buildId": state.get("buildId", ""), "errorCode": state.get("errorCode", ""),
            "slotCount": len(latest.get("slots", {})), "engine": latest.get("engine", ""),
            "publishedAt": latest.get("publishedAt"), "attempts": state.get("attempts", 0)}
