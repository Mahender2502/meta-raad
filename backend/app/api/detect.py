"""
Detection endpoints.

POST /detect/compare — run the same text through the LLM twice: once with only the
                       category names (plain AD-LLM zero-shot, k = 0) and once with the
                       k nearest normal train articles added to the prompt (RA-ZAD).
                       Built for the demo's side-by-side view.

Retrieval is always normal-only: the memory is the train split, which has no anomalies,
so an anomaly shows up as being far from every retrieved article.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException

from app.api import dashboard
from app.api.schemas import CompareRequest, CompareResponse, DetectionOut, ExemplarOut
from app.core.datasets import DATASETS
from app.services.bert_embedder import get_bert_embedder
from app.services.detector import DetectionResult, Exemplar, build_detection_prompt, detect
from app.services.llm_client import get_llm_client
from app.services.memory import Neighbour, get_memory

router = APIRouter(prefix="/detect", tags=["detect"])
logger = logging.getLogger(__name__)

# Characters of each retrieved article sent back to the page (the prompt itself is
# shortened further by the detector).
_DISPLAY_CHARS = 1500


def _to_out(result: DetectionResult) -> DetectionOut:
    verdict = None
    if result.score is not None:
        verdict = "anomaly" if result.score >= 0.5 else "normal"
    return DetectionOut(
        score=result.score,
        verdict=verdict,
        reason=result.reason,
        model=result.model,
        error=result.error,
        prompt=result.prompt,
        citations=result.citations,
    )


def retrieve_neighbours(dataset: str, text: str, k: int) -> list[Neighbour]:
    """Embed the text with BERT and return its k nearest normal train articles."""
    memory = get_memory(dataset)
    return memory.search(get_bert_embedder().embed(text), k)


@router.post("/compare", response_model=CompareResponse)
def compare(request: CompareRequest) -> CompareResponse:
    spec = DATASETS.get(request.dataset)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Unknown dataset '{request.dataset}'")
    if not (request.run_baseline or request.run_rag):
        raise HTTPException(status_code=422, detail="Nothing to run: set run_baseline or run_rag")

    neighbours: list[Neighbour] = []
    if request.run_rag:  # retrieval is only needed for the grounded prompt
        try:
            neighbours = retrieve_neighbours(spec.key, request.text, request.k)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Retrieval data not found ({exc.filename}). Run the data download.",
            ) from exc
        except Exception as exc:  # noqa: BLE001 - surfaced as a clear API error
            raise HTTPException(status_code=502, detail=f"Retrieval failed: {exc}") from exc

    exemplars = [
        Exemplar(text=n.text, category=n.category, is_anomaly=False, doc_id=n.doc_id)
        for n in neighbours
    ]

    def prompt_for(ex: list[Exemplar] | None) -> str:
        return build_detection_prompt(
            request.text,
            spec.normal_categories,
            spec.anomaly_category,
            request.setting,
            exemplars=ex,
        )

    try:
        llm = get_llm_client()
    except Exception as exc:  # noqa: BLE001 - e.g. missing API key
        raise HTTPException(status_code=500, detail=f"LLM not configured: {exc}") from exc

    def timed(**kwargs) -> tuple[DetectionResult, float]:
        started = time.perf_counter()
        result = detect(llm, **kwargs)
        return result, time.perf_counter() - started

    with ThreadPoolExecutor(max_workers=2) as pool:
        baseline_future = pool.submit(timed, prompt=prompt_for(None)) if request.run_baseline else None
        rag_future = (
            pool.submit(timed, prompt=prompt_for(exemplars), n_exemplars=len(exemplars))
            if request.run_rag
            else None
        )
        baseline, baseline_s = baseline_future.result() if baseline_future else (None, 0.0)
        rag, rag_s = rag_future.result() if rag_future else (None, 0.0)

    def summary(result: DetectionResult | None, seconds: float) -> dict | None:
        if result is None:
            return None
        return {
            "score": result.score,
            "error": result.error is not None,
            "seconds": seconds,
            "prompt_chars": len(result.prompt),
            "citations": len(result.citations),
        }

    try:  # statistics must never break a request
        dashboard.store.record(
            text=request.text,
            setting=request.setting,
            k=request.k,
            baseline=summary(baseline, baseline_s),
            rag=summary(rag, rag_s),
            neighbours=[(n.category, n.distance) for n in neighbours],
        )
    except Exception:  # noqa: BLE001
        logger.exception("could not update dashboard statistics")

    return CompareResponse(
        dataset=spec.key,
        setting=request.setting,
        k=request.k,
        baseline=_to_out(baseline) if baseline else None,
        rag=_to_out(rag) if rag else None,
        retrieved=[
            ExemplarOut(
                number=i,
                doc_id=n.doc_id,
                chunk_id=None,
                text=n.text[:_DISPLAY_CHARS],
                category=n.category,
                is_anomaly=False,
                distance=n.distance,
            )
            for i, n in enumerate(neighbours, start=1)
        ],
    )
