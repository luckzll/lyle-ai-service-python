from app.core.config import settings


def qdrant_headers() -> dict[str, str]:
    """Return Qdrant authentication headers without logging the API key."""
    api_key = settings.qdrant_api_key.strip()
    return {"api-key": api_key} if api_key else {}
