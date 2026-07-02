from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from app.core.metrics import collect_metrics, collect_prometheus_metrics

router = APIRouter()


@router.get("/metrics")
def metrics() -> dict[str, object]:
    return collect_metrics()


@router.get("/metrics/prometheus", response_class=PlainTextResponse)
def prometheus_metrics() -> str:
    return collect_prometheus_metrics()
