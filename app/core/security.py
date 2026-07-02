from secrets import compare_digest

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from app.core.config import settings


PUBLIC_PATHS = {"/health"}
PUBLIC_PREFIXES = ("/generated-images/",)


async def internal_token_middleware(request: Request, call_next) -> Response:
    if request.url.path in PUBLIC_PATHS or request.url.path.startswith(PUBLIC_PREFIXES):
        return await call_next(request)

    expected = settings.internal_api_token
    if expected:
        actual = request.headers.get("X-Internal-Token", "")
        if not compare_digest(actual, expected):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid internal token."},
            )

    return await call_next(request)
