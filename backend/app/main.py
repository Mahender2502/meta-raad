"""
Meta-RAAD Backend — FastAPI application entrypoint.

Current scope: health checks + the RAG retrieval endpoint. RA-ZAD detection,
FG-MOS and G_score endpoints (see spec.md) will be added as additional
routers under app/api/ in a later step.
"""

from fastapi import FastAPI

from app.api import health, rag
from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Backend service for the Meta-RAAD anomaly detection framework "
    "(RA-ZAD retrieval-augmented detection, FG-MOS model selection, "
    "G_score grounding metric).",
)

app.include_router(health.router)
app.include_router(rag.router)


@app.get("/")
def root() -> dict:
    return {"message": f"{settings.app_name} is running. See /docs for the API reference."}
