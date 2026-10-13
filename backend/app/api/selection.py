"""Training-only profiling and LLM model selection API."""
from functools import lru_cache
from pathlib import Path
from typing import Literal
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from threadpoolctl import threadpool_limits

from app.core.config import settings
from app.core.datasets import DATASETS
from app.services.llm_client import get_llm_client
from app.services.model_selection import CARDS, build_selection_prompt, parse_selection
from app.services.profiling import profile_training
from app.services.research_data import load_split

router = APIRouter(prefix="/selection", tags=["model selection"])


class SelectionRequest(BaseModel):
    dataset: str = "n24_news"
    embedding: Literal["bert", "openai"] = "bert"
    condition: Literal["baseline", "dataset_only", "models_only", "full"] = "full"
    candidates: list[str] = Field(default_factory=lambda:list(CARDS), min_length=1,max_length=8)
    seed: int = Field(42,ge=0)
    max_samples: int = Field(1000,ge=20,le=2000)


@lru_cache(maxsize=4)
def _profile(dataset, embedding, seed, max_samples, directory, fingerprint):
    spec = DATASETS[dataset]
    train = load_split(Path(directory)/spec.data_subdir,spec.file_prefix,"train",embedding)
    with threadpool_limits(limits=2):
        return profile_training(train.records,train.vectors,seed=seed,max_samples=max_samples)


def get_profile(dataset, embedding="bert", seed=42, max_samples=1000):
    if dataset not in DATASETS:
        raise HTTPException(404,"Unknown dataset")
    spec = DATASETS[dataset]
    directory = Path(settings.data_dir)/spec.data_subdir
    try:
        fingerprint = tuple((p.name,p.stat().st_mtime_ns,p.stat().st_size) for p in sorted(directory.glob(f"{spec.file_prefix}_train*")))
        return _profile(dataset,embedding,seed,max_samples,str(settings.data_dir),fingerprint)
    except FileNotFoundError as exc:
        raise HTTPException(503,"Training data or requested embeddings are missing. Run scripts/download_data.py.") from exc
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc


@router.get("/cards")
def cards():
    return {"candidates": CARDS, "note":"Capability matchups are hypotheses; compare recommendations against measured held-out performance."}


@router.get("/profile/{dataset}")
def profile(dataset: str, embedding: Literal["bert","openai"]="bert"):
    return get_profile(dataset,embedding)


@router.post("/recommend")
def recommend(request: SelectionRequest):
    if len(set(request.candidates))!=len(request.candidates) or any(c not in CARDS for c in request.candidates):
        raise HTTPException(422,"Candidates must be distinct supported model identifiers")
    data = get_profile(request.dataset,request.embedding,request.seed,request.max_samples)
    prompt, features = build_selection_prompt(data,request.dataset,request.candidates,request.condition,request.seed)
    started = time.perf_counter()
    try:
        response = get_llm_client().generate(prompt,max_tokens=max(2048,settings.llm_max_tokens))
    except Exception as exc:
        raise HTTPException(502,f"Model selection LLM failed: {exc}") from exc
    try:
        result = parse_selection(response.text,request.candidates,features)
    except (ValueError,TypeError,AttributeError) as exc:
        raise HTTPException(502,f"Invalid model-selection response: {exc}") from exc
    return {**result,"dataset":request.dataset,"embedding":request.embedding,"condition":request.condition,
            "model":response.model,"features":features,"prompt":prompt,"usage":response.raw.get("usage"),
            "seconds":time.perf_counter()-started}
