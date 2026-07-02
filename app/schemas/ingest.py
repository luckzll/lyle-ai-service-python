from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    file_id: str = Field(alias="fileId")
    knowledge_base_id: str = Field(alias="knowledgeBaseId")
    storage_path: str = Field(alias="storagePath")
    user_id: str | None = Field(default=None, alias="userId")


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
