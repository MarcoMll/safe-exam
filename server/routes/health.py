"""Health check route."""

from fastapi import APIRouter

from server.config import get_settings
from server.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(status="ok", version=settings.version)
