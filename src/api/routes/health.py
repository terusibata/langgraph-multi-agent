"""Health check and metrics routes."""

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Response
import structlog

from src.api.schemas.response import HealthResponse
from src.services.metrics import get_metrics_collector
from src.config import get_settings
from src import __version__

logger = structlog.get_logger()

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def health_check():
    """
    Check the health of the service and its dependencies.

    Returns overall health status and component-level details.
    """
    settings = get_settings()
    components = {}
    overall_status: Literal["healthy", "degraded", "unhealthy"] = "healthy"

    # Check database (placeholder - would check actual connection)
    try:
        # In production, would check actual database connection
        components["database"] = {
            "status": "healthy",
            "latency_ms": 1,
        }
    except Exception as e:
        components["database"] = {
            "status": "unhealthy",
            "error": str(e),
        }
        overall_status = "unhealthy"

    # Check Redis (placeholder)
    try:
        components["redis"] = {
            "status": "healthy",
            "latency_ms": 1,
        }
    except Exception as e:
        components["redis"] = {
            "status": "unhealthy",
            "error": str(e),
        }
        if overall_status == "healthy":
            overall_status = "degraded"

    # Check Bedrock availability (placeholder)
    try:
        components["bedrock"] = {
            "status": "healthy",
            "region": settings.aws_region,
        }
    except Exception as e:
        components["bedrock"] = {
            "status": "unhealthy",
            "error": str(e),
        }
        overall_status = "unhealthy"

    return HealthResponse(
        status=overall_status,
        version=__version__,
        timestamp=datetime.utcnow(),
        components=components,
    )


@router.get(
    "/health/live",
    summary="Liveness probe",
)
async def liveness():
    """
    Kubernetes liveness probe.

    Returns 200 if the service is running.
    """
    return {"status": "alive"}


@router.get(
    "/health/ready",
    summary="Readiness probe",
)
async def readiness():
    """
    Kubernetes readiness probe.

    Returns 200 if the service is ready to accept traffic.
    """
    # In production, would check if all dependencies are ready
    return {"status": "ready"}


@router.get(
    "/metrics",
    summary="Prometheus metrics",
)
async def metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text format.
    """
    collector = get_metrics_collector()
    return Response(
        content=collector.get_metrics(),
        media_type=collector.get_content_type(),
    )
