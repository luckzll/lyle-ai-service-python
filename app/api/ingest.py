import logging

from fastapi import APIRouter, HTTPException

from app.rag.store import get_task
from app.schemas.ingest import IngestRequest, IngestResponse, IngestTaskResponse
from app.workers.ingest_worker import enqueue_ingest_task

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    logger.info("Ingest requested. file_id=%s knowledge_base_id=%s", request.file_id, request.knowledge_base_id)
    return enqueue_ingest_task(request)


@router.get("/ingest/tasks/{task_id}", response_model=IngestTaskResponse)
def get_ingest_task(task_id: str) -> IngestTaskResponse:
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Ingest task not found.")

    return IngestTaskResponse(
        taskId=task["task_id"],
        fileId=task["file_id"],
        knowledgeBaseId=task["knowledge_base_id"],
        status=task["status"],
        message=task["message"],
    )
