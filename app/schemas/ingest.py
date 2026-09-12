from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    file_id: str = Field(alias="fileId")
    knowledge_base_id: str = Field(alias="knowledgeBaseId")
    storage_path: str = Field(alias="storagePath")
    storage_provider: str | None = Field(default=None, alias="storageProvider")
    bucket: str | None = None
    object_key: str | None = Field(default=None, alias="objectKey")
    original_name: str | None = Field(default=None, alias="originalName")
    content_type: str | None = Field(default=None, alias="contentType")
    file_size: int | None = Field(default=None, alias="fileSize")
    visibility: str | None = None
    user_id: str | None = Field(default=None, alias="userId")
    retrieval_revision: str = Field(default="legacy", alias="retrievalRevision", min_length=1, max_length=64)


class IngestResponse(BaseModel):
    task_id: str = Field(alias="taskId")
    status: str
    message: str
    candidate_memories: list[dict[str, object]] = Field(default_factory=list, alias="candidateMemories")


class IngestTaskResponse(BaseModel):
    task_id: str = Field(alias="taskId")
    file_id: str = Field(alias="fileId")
    knowledge_base_id: str = Field(alias="knowledgeBaseId")
    status: str
    message: str
