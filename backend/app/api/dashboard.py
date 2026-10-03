"""
Dashboard data endpoints. All dashboard numbers come from JSON files in settings.stats_dir
(backend/data/): real.json is built by scripts/make_dashboard_data.py; results.json holds the
evaluation results and is written by the evaluation runner (until it exists the results panels
show a "pending" state); live_stats.json is updated on every query by app/services/stats.py.

GET    /api/dashboard/real     statistics of the N24 data
GET    /api/dashboard/results  evaluation results, or {"available": false} until a run has produced them
GET    /api/dashboard/live     usage statistics of the demo queries (zeros until the first query)
DELETE /api/dashboard/live     clear the live statistics
"""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.services.stats import StatsStore

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])
store = StatsStore(Path(settings.stats_dir))


def _file(name: str) -> dict:
    path = Path(settings.stats_dir) / f"{name}.json"
    if not path.exists():
        raise HTTPException(
            status_code=404, detail=f"{path.name} not found. Run scripts/make_dashboard_data.py."
        )
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/real")
def real() -> dict:
    return _file("real")


@router.get("/results")
def results() -> dict:
    path = Path(settings.stats_dir) / "results.json"
    if not path.exists():
        return {"available": False}
    return {"available": True, **json.loads(path.read_text(encoding="utf-8"))}


@router.get("/live")
def live() -> dict:
    return store.read()


@router.delete("/live")
def reset_live() -> dict:
    return store.reset()
