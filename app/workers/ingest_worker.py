from app.rag.store import add_document
from app.schemas.ingest import IngestRequest, IngestResponse


def enqueue_ingest_task(request: IngestRequest) -> IngestResponse:
    result = add_document(
        file_id=request.file_id,
        knowledge_base_id=request.knowledge_base_id,
        storage_path=request.storage_path,
        user_id=request.user_id,
    )
    return IngestResponse(
        taskId=str(result["task_id"]),
        status="completed",
        message="Document ingest completed.",
        candidateMemories=result.get("candidate_memories") or [],
    )
