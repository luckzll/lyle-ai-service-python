from app.rag.store import add_document
from app.rag.document_loader import StorageObjectRef
from app.schemas.ingest import IngestRequest, IngestResponse


def enqueue_ingest_task(request: IngestRequest) -> IngestResponse:
    result = add_document(
        file_id=request.file_id,
        knowledge_base_id=request.knowledge_base_id,
        storage_path=request.storage_path,
        user_id=request.user_id,
        retrieval_revision=request.retrieval_revision,
        storage_ref=StorageObjectRef(
            storage_path=request.storage_path,
            storage_provider=request.storage_provider,
            bucket=request.bucket,
            object_key=request.object_key,
            original_name=request.original_name,
            content_type=request.content_type,
            file_size=request.file_size,
            visibility=request.visibility,
        ),
    )
    return IngestResponse(
        taskId=str(result["task_id"]),
        status="completed",
        message="Document ingest completed.",
        candidateMemories=result.get("candidate_memories") or [],
    )
