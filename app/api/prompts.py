from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.prompts.store import activate_prompt_template, create_prompt_template, get_prompt_history

router = APIRouter()


class PromptTemplateRequest(BaseModel):
    name: str
    version: str
    systemPrompt: str
    status: str = "draft"
    trafficPercent: int = Field(default=100, ge=0, le=100)


@router.get("/prompts")
def prompt_templates(limit: int = 50) -> dict[str, object]:
    return get_prompt_history(limit)


@router.post("/prompts")
def save_prompt_template(request: PromptTemplateRequest) -> dict[str, object]:
    return create_prompt_template(request.model_dump())


@router.post("/prompts/{prompt_id}/activate")
def activate_prompt(prompt_id: int) -> dict[str, object]:
    return activate_prompt_template(prompt_id)
