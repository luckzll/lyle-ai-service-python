import logging
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.agent import router as agent_router
from app.api.chat import router as chat_router
from app.api.evaluation import router as evaluation_router
from app.api.ingest import router as ingest_router
from app.api.memory import router as memory_router
from app.api.metrics import router as metrics_router
from app.api.prompts import router as prompts_router
from app.api.rag_debug import router as rag_debug_router
from app.api.tools import router as tools_router
from app.core.config import settings
from app.core.database import close_postgres_pool
from app.core.logging_config import configure_logging
from app.core.metrics import metrics_middleware
from app.core.rate_limit import rate_limit_middleware
from app.core.security import internal_token_is_insecure, internal_token_middleware
from app.workers.notification_worker import run_notification_worker

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Python AI service starting. postgres_enabled=%s qdrant_enabled=%s milvus_enabled=%s allowed_tools=%s image_analyze=%s image_generate=%s image_edit=%s",
        settings.postgres_enabled,
        settings.qdrant_enabled,
        settings.milvus_enabled,
        settings.tool_allowed_names,
        settings.tool_image_analyze_enabled,
        settings.tool_image_generate_enabled,
        settings.tool_image_edit_enabled,
    )
    if internal_token_is_insecure():
        logger.error(
            "INTERNAL_API_TOKEN 未配置为安全值（为空或占位值），所有 /ai/* 请求将返回 503。"
            "请在 .env 中设置强随机 token。"
        )
    stop_event = asyncio.Event()
    notification_task: asyncio.Task[None] | None = None
    if settings.notification_worker_enabled:
        notification_task = asyncio.create_task(run_notification_worker(stop_event))
    try:
        yield
    finally:
        stop_event.set()
        if notification_task is not None:
            await notification_task
    close_postgres_pool()
    logger.info("Python AI service stopped.")


app = FastAPI(title="AI App Python Service", version="0.1.0", lifespan=lifespan)
app.middleware("http")(metrics_middleware)
app.middleware("http")(rate_limit_middleware)
app.middleware("http")(internal_token_middleware)
generated_image_dir = Path(settings.generated_image_dir).expanduser()
generated_image_dir.mkdir(parents=True, exist_ok=True)
app.mount("/generated-images", StaticFiles(directory=generated_image_dir), name="generated-images")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ai-service-python"}


app.include_router(chat_router, prefix="/ai", tags=["chat"])
app.include_router(evaluation_router, prefix="/ai", tags=["evaluation"])
app.include_router(ingest_router, prefix="/ai", tags=["ingest"])
app.include_router(agent_router, prefix="/ai", tags=["agent"])
app.include_router(memory_router, prefix="/ai", tags=["memory"])
app.include_router(metrics_router, prefix="/ai", tags=["metrics"])
app.include_router(tools_router, prefix="/ai", tags=["tools"])
app.include_router(prompts_router, prefix="/ai", tags=["prompts"])
app.include_router(rag_debug_router, prefix="/ai", tags=["rag-debug"])
