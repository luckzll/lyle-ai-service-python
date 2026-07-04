from fastapi import APIRouter

from app.briefings.briefing_builder import generate_daily_briefing, generate_review_briefing, get_today_briefing

router = APIRouter()


@router.get("/briefings/{user_id}/today")
def today(user_id: str, projectId: int | None = None) -> dict[str, object]:
    return get_today_briefing(user_id, project_id=projectId)


@router.post("/briefings/{user_id}/daily/generate")
def generate_daily(user_id: str, projectId: int | None = None) -> dict[str, object]:
    return generate_daily_briefing(user_id, project_id=projectId, persist=True)


@router.post("/briefings/{user_id}/evening/generate")
def generate_evening(user_id: str, projectId: int | None = None) -> dict[str, object]:
    return generate_review_briefing(user_id, "evening", project_id=projectId, persist=True)


@router.post("/briefings/{user_id}/weekly/generate")
def generate_weekly(user_id: str, projectId: int | None = None) -> dict[str, object]:
    return generate_review_briefing(user_id, "weekly", project_id=projectId, persist=True)
