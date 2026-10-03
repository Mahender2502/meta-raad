"""
Meta-RAAD Backend — FastAPI application entrypoint.

Current scope: health check, the RA-ZAD detection demo (/detect/compare plus the
page at "/") and the dashboard at /dashboard. FG-MOS and G_score will be added as additional
routers under app/api/ later.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import dashboard, detect, health
from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Backend service for the Meta-RAAD anomaly detection framework "
    "(RA-ZAD retrieval-augmented detection, FG-MOS model selection, "
    "G_score grounding metric).",
)

app.include_router(health.router)
app.include_router(detect.router)
app.include_router(dashboard.router)

_STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC), name="static")


@app.get("/", include_in_schema=False)
def demo_page() -> FileResponse:
    """Demo page: grounded anomaly detection with an optional baseline comparison."""
    return FileResponse(_STATIC / "index.html")


@app.get("/dashboard", include_in_schema=False)
def dashboard_page() -> FileResponse:
    """Dashboard: real N24 data statistics plus sample-data result panels."""
    return FileResponse(_STATIC / "dashboard.html")

