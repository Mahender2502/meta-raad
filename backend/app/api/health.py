"""
Health-check endpoints.

Used by docker-compose / manual checks to confirm the backend service is up
and can reach its ChromaDB dependency. Business logic (RAG-AD retrieval,
self-consistency detection, explanation evaluation) will live in sibling
router modules under app/api/, added in a later step.
"""

from fastapi import APIRouter

from app.core.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict:
    """Basic liveness check for the backend service itself."""
    return {"status": "ok", "service": settings.app_name, "version": settings.app_version}


@router.get("/health/chromadb")
def chromadb_health_check() -> dict:
    """
    Reports whether the backend can reach ChromaDB, without importing the
    full retrieval stack (kept dependency-light until RAG-AD is implemented).
    """
    import httpx

    url = f"http://{settings.chroma_host}:{settings.chroma_port}/api/v1/heartbeat"
    try:
        response = httpx.get(url, timeout=3.0)
        response.raise_for_status()
        return {"status": "ok", "chromadb": response.json()}
    except Exception as exc:  # noqa: BLE001 - surfaced directly for a health probe
        return {"status": "unreachable", "error": str(exc)}
