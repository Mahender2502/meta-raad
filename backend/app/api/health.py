"""
Health-check endpoint.

Used by docker-compose / manual checks to confirm the backend service is up.
Business logic (RA-ZAD detection, FG-MOS, G_score) lives in sibling router
modules under app/api/.
"""

from fastapi import APIRouter

from app.core.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict:
    """Basic liveness check for the backend service itself."""
    return {"status": "ok", "service": settings.app_name, "version": settings.app_version}
