import logging
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from app.core.runtime import require_supported_python

require_supported_python()

from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles

from app.api.agent import router as agent_router
from app.api.briefings import router as briefings_router
from app.api.chat import router as chat_router
from app.api.local_workspace import router as local_workspace_router
from app.api.connectors import router as connectors_router
from app.api.evaluation import router as evaluation_router
from app.api.goals import router as goals_router
from app.api.ingest import router as ingest_router
from app.api.memory import router as memory_router
from app.api.metrics import router as metrics_router
from app.api.plans import router as plans_router
from app.api.prompts import router as prompts_router
from app.api.projects import router as projects_router
from app.api.rag_debug import router as rag_debug_router
from app.api.tools import router as tools_router
from app.api.workbench import router as workbench_router
from app.api.experience import router as experience_router
from app.api.agent_tasks import router as agent_tasks_router
from app.api.missions import router as missions_router
from app.core.config import settings
from app.core.database import close_postgres_pool
from app.core.logging_config import configure_logging
from app.core.metrics import metrics_middleware
from app.core.rate_limit import rate_limit_middleware
from app.core.security import internal_token_is_insecure, internal_token_middleware
from app.embedding.client import get_embedding_readiness, warm_up_embedding_model
from app.workers.memory_decay_worker import run_memory_decay_worker
from app.workers.notification_worker import run_notification_worker
from app.workers.automation_scheduler_worker import run_automation_scheduler_worker
from app.workers.usage_outbox_worker import run_usage_outbox_worker
from app.workers.memory_purge_worker import run_memory_purge_worker
from app.workers.memory_schema_worker import run_memory_schema_worker

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
            "请在 deploy/.env 中设置强随机 token。"
        )
    try:
        embedding_status = await asyncio.to_thread(warm_up_embedding_model)
        logger.info(
            "Embedding model ready. provider=%s model=%s",
            embedding_status["provider"],
            embedding_status["model"],
        )
    except Exception:
        # Keep liveness available for diagnostics while readiness remains 503.
        logger.exception("Embedding model warm-up failed; service will remain unready.")
    stop_event = asyncio.Event()
    notification_task: asyncio.Task[None] | None = None
    memory_decay_task: asyncio.Task[None] | None = None
    automation_task: asyncio.Task[None] | None = None
    usage_outbox_task: asyncio.Task[None] | None = None
    memory_purge_task: asyncio.Task[None] | None = None
    schema_task = asyncio.create_task(run_memory_schema_worker(stop_event)) if settings.memory_schema_enabled else None
    if settings.notification_worker_enabled:
        notification_task = asyncio.create_task(run_notification_worker(stop_event))
    if settings.memory_decay_worker_enabled:
        memory_decay_task = asyncio.create_task(run_memory_decay_worker(stop_event))
    if settings.automation_worker_enabled:
        automation_task = asyncio.create_task(run_automation_scheduler_worker(stop_event))
    if settings.ai_usage_outbox_worker_enabled and settings.postgres_enabled:
        usage_outbox_task = asyncio.create_task(run_usage_outbox_worker(stop_event))
    if settings.memory_purge_worker_enabled:
        memory_purge_task = asyncio.create_task(run_memory_purge_worker(stop_event))
    try:
        yield
    finally:
        stop_event.set()
        if schema_task is not None:
            await schema_task
        if notification_task is not None:
            await notification_task
        if memory_decay_task is not None:
            await memory_decay_task
        if automation_task is not None:
            await automation_task
        if usage_outbox_task is not None:
            await usage_outbox_task
        if memory_purge_task is not None:
            await memory_purge_task
    close_postgres_pool()
    logger.info("Python AI service stopped.")


app = FastAPI(title="AI App Python Service", version="6.6.0", lifespan=lifespan)
app.middleware("http")(metrics_middleware)
app.middleware("http")(rate_limit_middleware)
app.middleware("http")(internal_token_middleware)
generated_image_dir = Path(settings.generated_image_dir).expanduser()
generated_image_dir.mkdir(parents=True, exist_ok=True)
app.mount("/generated-images", StaticFiles(directory=generated_image_dir), name="generated-images")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ai-service-python"}


@app.get("/ready")
def ready(response: Response) -> dict[str, object]:
    embedding = get_embedding_readiness()
    is_ready = bool(embedding["ready"]) and not internal_token_is_insecure()
    if not is_ready:
        response.status_code = 503
    return {
        "status": "ready" if is_ready else "not_ready",
        "service": "ai-service-python",
        "embedding": embedding,
        "securityConfigured": not internal_token_is_insecure(),
    }


app.include_router(chat_router, prefix="/ai", tags=["chat"])
app.include_router(local_workspace_router, prefix="/ai", tags=["local-workspace"])
app.include_router(briefings_router, prefix="/ai", tags=["briefings"])
app.include_router(connectors_router, prefix="/ai", tags=["connectors"])
app.include_router(evaluation_router, prefix="/ai", tags=["evaluation"])
app.include_router(goals_router, prefix="/ai", tags=["goals"])
app.include_router(ingest_router, prefix="/ai", tags=["ingest"])
app.include_router(agent_router, prefix="/ai", tags=["agent"])
app.include_router(memory_router, prefix="/ai", tags=["memory"])
app.include_router(metrics_router, prefix="/ai", tags=["metrics"])
app.include_router(plans_router, prefix="/ai", tags=["plans"])
app.include_router(tools_router, prefix="/ai", tags=["tools"])
app.include_router(prompts_router, prefix="/ai", tags=["prompts"])
app.include_router(projects_router, prefix="/ai", tags=["projects"])
app.include_router(rag_debug_router, prefix="/ai", tags=["rag-debug"])
app.include_router(workbench_router, prefix="/ai", tags=["workbench"])
app.include_router(experience_router, prefix="/ai", tags=["experience"])
app.include_router(agent_tasks_router, prefix="/internal", tags=["agent-tasks-internal"])
app.include_router(missions_router, prefix="/internal", tags=["missions-internal"])
