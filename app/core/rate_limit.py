from time import time
from threading import Lock

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from app.core.config import settings


_lock = Lock()
_windows: dict[str, tuple[int, int]] = {}


async def rate_limit_middleware(request: Request, call_next) -> Response:
    if request.url.path in {"/health", "/ready"}:
        return await call_next(request)

    limit = settings.python_rate_limit_per_minute
    if limit <= 0:
        return await call_next(request)

    client = request.client.host if request.client else "unknown"
    key = f"{client}:{request.url.path}"
    window_id = int(time() // 60)

    with _lock:
        current_window, count = _windows.get(key, (window_id, 0))
        if current_window != window_id:
            current_window, count = window_id, 0
        count += 1
        _windows[key] = (current_window, count)

    if count > limit:
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests."},
        )

    return await call_next(request)
