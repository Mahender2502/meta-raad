"""
RAD-LLM Backend — FastAPI application entrypoint.

Current scope: application scaffolding + health checks only. The RAG-AD
retrieval, self-consistency scoring, and explanation-quality-judge endpoints
described in RAD-LLM_Framework_Specification.md will be added as additional
routers under app/api/ in a later step.
"""

from fastapi import FastAPI

from app.api import health, rag
from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Backend service for the RAD-LLM anomaly detection pipeline "
    "(RAG-AD retrieval, self-consistency scoring, explanation-quality "
    "evaluation). Self-consistency scoring and explanation-quality "
    "evaluation are implemented in a later step.",
)

app.include_router(health.router)
app.include_router(rag.router)


@app.get("/")
def root() -> dict:
    return {"message": f"{settings.app_name} is running. See /docs for the API reference."}
