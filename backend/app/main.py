"""
Meta-RAAD Backend — FastAPI application entrypoint.

RA-ZAD anomaly detection, training-only FG-MOS model selection, and dashboard.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import dashboard, detect, health, selection, workspace
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
app.include_router(selection.router)
app.include_router(workspace.router)

_STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC), name="static")
_FRONTEND = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if (_FRONTEND / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=_FRONTEND / "assets"), name="frontend-assets")


@app.get("/", include_in_schema=False)
def demo_page() -> FileResponse:
    """Demo page: grounded anomaly detection with an optional baseline comparison."""
    return FileResponse(_FRONTEND / "index.html" if (_FRONTEND / "index.html").exists() else _STATIC / "index.html")


@app.get("/datasets", include_in_schema=False)
@app.get("/vectors", include_in_schema=False)
@app.get("/detection", include_in_schema=False)
@app.get("/retrieval", include_in_schema=False)
@app.get("/models", include_in_schema=False)
@app.get("/evaluation", include_in_schema=False)
@app.get("/settings", include_in_schema=False)
def workspace_page() -> FileResponse:
    return demo_page()


@app.get("/dashboard", include_in_schema=False)
def dashboard_page() -> FileResponse:
    """Dashboard: real N24 data statistics plus sample-data result panels."""
    return FileResponse(_STATIC / "dashboard.html")


@app.get("/model-selection", include_in_schema=False)
def selection_page() -> FileResponse:
    return FileResponse(_STATIC / "selection.html")

