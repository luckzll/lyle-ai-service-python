from fastapi import APIRouter, HTTPException

from app.missions.delegated_actions import compensate_action
from app.missions.draft_builder import build_mission_draft, validate_mission_workspace
from app.missions.result_digest import build_mission_digest
from app.missions.schemas import (
    CompensationRequest,
    CompensationResponse,
    MissionDigestRequest,
    MissionDigestResponse,
    MissionDraftRequest,
    MissionDraftResponse,
    ShadowScenarioRequest,
    ShadowScenarioResponse,
)
from app.missions.shadow_scenarios import build_shadow_scenarios


router = APIRouter()


@router.post("/mission/draft", response_model=MissionDraftResponse)
def mission_draft(request: MissionDraftRequest) -> MissionDraftResponse:
    try:
        validate_mission_workspace(request)
        return build_mission_draft(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/mission/shadow-scenarios", response_model=ShadowScenarioResponse)
def mission_shadow_scenarios(request: ShadowScenarioRequest) -> ShadowScenarioResponse:
    return build_shadow_scenarios(request)


@router.post("/mission/compensate", response_model=CompensationResponse)
def mission_compensate(request: CompensationRequest) -> CompensationResponse:
    return compensate_action(request)


@router.post("/mission/digest", response_model=MissionDigestResponse)
def mission_digest(request: MissionDigestRequest) -> MissionDigestResponse:
    return build_mission_digest(request)
