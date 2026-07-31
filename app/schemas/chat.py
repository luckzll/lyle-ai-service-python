from pydantic import BaseModel, Field


class ChatAttachmentStorage(BaseModel):
    storagePath: str
    storageProvider: str | None = None
    bucket: str | None = None
    objectKey: str | None = None
    originalName: str | None = None
    contentType: str | None = None
    fileSize: int | None = None


class ChatAttachment(BaseModel):
    id: str
    kind: str
    name: str
    mimeType: str
    size: int
    sourceType: str | None = None
    folderGroupId: str | None = None
    folderRoot: str | None = None
    relativePath: str | None = None
    storage: ChatAttachmentStorage


class ChatRequest(BaseModel):
    # FastAPI/Pydantic 当前组合会在请求模型字段使用 alias 时产生错误告警；
    # 直接保持 HTTP camelCase 契约，并用只读属性供内部 Python 代码使用。
    userId: str
    conversationId: str | None = None
    turnId: str | None = None
    projectId: int | None = None
    message: str
    knowledgeBaseIds: list[str] = Field(default_factory=list)
    stream: bool = False
    aiConfig: dict[str, str] = Field(default_factory=dict)
    attachments: list[ChatAttachment] = Field(default_factory=list)
    traceId: str | None = None
    reservationId: str | None = None
    usageBudget: dict[str, int] = Field(default_factory=dict)

    @property
    def user_id(self) -> str:
        return self.userId

    @property
    def conversation_id(self) -> str | None:
        return self.conversationId

    @property
    def turn_id(self) -> str | None:
        return self.turnId

    @property
    def project_id(self) -> int | None:
        return self.projectId

    @property
    def knowledge_base_ids(self) -> list[str]:
        return self.knowledgeBaseIds

    @property
    def ai_config(self) -> dict[str, str]:
        return self.aiConfig


class Source(BaseModel):
    doc_id: str = Field(alias="docId")
    title: str
    chunk: str
    score: str = ""
    backend: str = ""
    hit_modes: str = Field(default="", alias="hitModes")
    chunk_index: str = Field(default="", alias="chunkIndex")
    parent_title: str = Field(default="", alias="parentTitle")
    start_char: str = Field(default="", alias="startChar")
    end_char: str = Field(default="", alias="endChar")


class Usage(BaseModel):
    prompt_tokens: int = Field(alias="promptTokens")
    completion_tokens: int = Field(alias="completionTokens")
    call_count: int = Field(default=0, alias="callCount")
    estimated_cost_micros: int = Field(default=0, alias="estimatedCostMicros")
    measurement_source: str = Field(default="PENDING", alias="measurementSource")


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    usage: Usage
    agent_trace: list[dict[str, object]] = Field(default_factory=list, alias="agentTrace")
    tool_results: list[dict[str, object]] = Field(default_factory=list, alias="toolResults")
    candidate_memories: list[dict[str, object]] = Field(default_factory=list, alias="candidateMemories")
    candidate_tasks: list[dict[str, object]] = Field(default_factory=list, alias="candidateTasks")
    candidate_goals: list[dict[str, object]] = Field(default_factory=list, alias="candidateGoals")
    candidate_decisions: list[dict[str, object]] = Field(default_factory=list, alias="candidateDecisions")
    candidate_open_questions: list[dict[str, object]] = Field(default_factory=list, alias="candidateOpenQuestions")
    candidate_plans: list[dict[str, object]] = Field(default_factory=list, alias="candidatePlans")
    execution_suggestions: list[dict[str, object]] = Field(default_factory=list, alias="executionSuggestions")
    project_context: dict[str, object] = Field(default_factory=dict, alias="projectContext")
    review: dict[str, object] = Field(default_factory=dict)
