from secrets import compare_digest

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from app.core.config import settings


PUBLIC_PATHS = {"/health"}
PUBLIC_PREFIXES = ("/generated-images/",)

# 空值和历史默认占位 token 一律视为不安全配置，直接拒绝服务，防止鉴权被静默跳过。
INSECURE_INTERNAL_TOKENS = {
    "",
    "lyle-internal-dev-token-change-me",
    "change-this-internal-token",
}


def internal_token_is_insecure() -> bool:
    return settings.internal_api_token.strip() in INSECURE_INTERNAL_TOKENS


async def internal_token_middleware(request: Request, call_next) -> Response:
    if request.url.path in PUBLIC_PATHS or request.url.path.startswith(PUBLIC_PREFIXES):
        return await call_next(request)

    if internal_token_is_insecure():
        return JSONResponse(
            status_code=503,
            content={
                "detail": (
                    "INTERNAL_API_TOKEN 未配置为安全值，服务拒绝处理请求。"
                    "请在 .env 中设置强随机 token（例如 openssl rand -hex 32），"
                    "并同步配置 Java 侧 AI_SERVICE_INTERNAL_TOKEN。"
                )
            },
        )

    actual = request.headers.get("X-Internal-Token", "")
    if not compare_digest(actual, settings.internal_api_token):
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid internal token."},
        )

    return await call_next(request)
