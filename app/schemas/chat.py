from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str = Field(alias="userId")
    conversation_id: str | None = Field(default=None, alias="conversationId")
    message: str
    knowledge_base_ids: list[str] = Field(default_factory=list, alias="knowledgeBaseIds")
    stream: bool = False
    ai_config: dict[str, str] = Field(default_factory=dict, alias="aiConfig")


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
    project_context: dict[str, object] = Field(default_factory=dict, alias="projectContext")
    review: dict[str, object] = Field(default_factory=dict)
