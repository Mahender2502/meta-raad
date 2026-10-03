"""Pydantic request/response models for the API layer."""

from typing import Literal

from pydantic import BaseModel, Field


class CompareRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000, description="Text to check for anomalies.")
    setting: Literal["normal_only", "normal_anomaly"] = "normal_only"
    k: int = Field(3, ge=1, le=10, description="Retrieved examples used by the RA-ZAD run.")
    dataset: str = Field("n24_news", description="Key of a dataset in app/core/datasets.py.")
    run_baseline: bool = Field(True, description="Run the plain AD-LLM zero-shot prompt (k = 0).")
    run_rag: bool = Field(True, description="Run the retrieval-grounded prompt (RA-ZAD).")


class DetectionOut(BaseModel):
    score: float | None
    verdict: str | None = Field(None, description="'anomaly' if score >= 0.5 else 'normal'.")
    reason: str | None
    model: str | None
    error: str | None
    prompt: str
    citations: list[int] = Field(
        default_factory=list,
        description="1-based numbers of the retrieved examples the model cited (RA-ZAD only).",
    )


class ExemplarOut(BaseModel):
    number: int = Field(..., description="1-based position; matches the numbers in citations.")
    doc_id: str | None
    chunk_id: str | None
    text: str
    category: str
    is_anomaly: bool
    distance: float | None


class CompareResponse(BaseModel):
    dataset: str
    setting: str
    k: int
    baseline: DetectionOut | None = Field(None, description="Plain AD-LLM zero-shot (k = 0), if run.")
    rag: DetectionOut | None = Field(None, description="RA-ZAD: same prompt + retrieved examples, if run.")
    retrieved: list[ExemplarOut] = Field(default_factory=list, description="Empty unless run_rag.")
