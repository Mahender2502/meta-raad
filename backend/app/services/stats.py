"""
Live usage statistics for the dashboard, kept in a JSON file and updated on every query.

File layout (data/live_stats.json):
  - counters (queries, baseline/rag runs, errors), per-setting and per-k counts,
  - verdict counts for the baseline and the retrieval run,
  - histograms of LLM scores and of the nearest-train-article distance,
  - how often each category is retrieved, latency and prompt-size sums, citation counts,
  - agreement between the baseline and the retrieval run when both were run for a text,
  - the 20 most recent queries.

The live file starts empty (all zeros) and only ever contains numbers from real queries.
Typed text is stored only as a short preview. Accuracy metrics (AUROC, AUPRC) are NOT computed
here: demo queries have no true labels; those come from the evaluation runs.
"""

import copy
import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

SCORE_BINS = 20            # LLM score 0..1
DIST_BINS, DIST_MAX = 30, 0.30   # cosine distance 0..0.30 (same range as the real N24 histogram)
RECENT_MAX = 20
PREVIEW_CHARS = 100


def empty_stats() -> dict:
    return {
        "updated_at": None,
        "totals": {"queries": 0, "baseline_runs": 0, "rag_runs": 0, "errors": 0},
        "settings": {"normal_only": 0, "normal_anomaly": 0},
        "k_used": {},
        "verdicts": {"baseline": {"anomaly": 0, "normal": 0}, "rag": {"anomaly": 0, "normal": 0}},
        "score_hist": {
            "bin_edges": [round(i / SCORE_BINS, 2) for i in range(SCORE_BINS + 1)],
            "baseline": [0] * SCORE_BINS,
            "rag": [0] * SCORE_BINS,
        },
        "distance_hist": {
            "bin_edges": [round(i * DIST_MAX / DIST_BINS, 3) for i in range(DIST_BINS + 1)],
            "nearest": [0] * DIST_BINS,
        },
        "retrieved_categories": {},
        "latency": {"baseline": {"n": 0, "seconds": 0.0}, "rag": {"n": 0, "seconds": 0.0}},
        "prompt_chars": {"baseline": {"n": 0, "chars": 0}, "rag": {"n": 0, "chars": 0}},
        "citations": {"rag_runs": 0, "runs_with_citations": 0, "total_cited": 0},
        "agreement": {"pairs": 0, "same_verdict": 0, "rag_flips_to_anomaly": 0, "rag_flips_to_normal": 0},
        "recent": [],
    }


def _verdict(score: float | None) -> str | None:
    return None if score is None else ("anomaly" if score >= 0.5 else "normal")


def _bin(value: float, n: int, hi: float) -> int:
    return max(0, min(n - 1, int(value / hi * n)))


class StatsStore:
    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)
        self.live_path = self.dir / "live_stats.json"
        self._lock = threading.Lock()

    def _load_locked(self) -> dict:
        if not self.live_path.exists():
            self._write_locked(empty_stats())
        with open(self.live_path, encoding="utf-8") as f:
            return json.load(f)

    def _write_locked(self, stats: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.live_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=1)
        os.replace(tmp, self.live_path)  # atomic: readers never see a half-written file

    def read(self) -> dict:
        with self._lock:
            return self._load_locked()

    def reset(self) -> dict:
        """Clear all live statistics."""
        with self._lock:
            stats = empty_stats()
            self._write_locked(stats)
            return stats

    def record(
        self,
        *,
        text: str,
        setting: str,
        k: int,
        baseline: dict | None,
        rag: dict | None,
        neighbours: list[tuple[str, float]],
    ) -> None:
        """
        Add one request. `baseline` / `rag` (when run) are dicts with keys: score (float|None),
        error (bool), seconds (float), prompt_chars (int), citations (int, rag only).
        `neighbours` are (category, cosine distance) of the retrieved articles, nearest first.
        """
        with self._lock:
            stats = self._load_locked()
            now = datetime.now(timezone.utc)
            text_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

            if rag is not None:  # a new query (the retrieval run is the main result)
                stats["totals"]["queries"] += 1
                stats["settings"][setting] = stats["settings"].get(setting, 0) + 1
                stats["k_used"][str(k)] = stats["k_used"].get(str(k), 0) + 1
                if neighbours:
                    stats["distance_hist"]["nearest"][_bin(neighbours[0][1], DIST_BINS, DIST_MAX)] += 1
                for category, _ in neighbours:
                    stats["retrieved_categories"][category] = stats["retrieved_categories"].get(category, 0) + 1
                stats["recent"].insert(0, {
                    "ts": now.isoformat(timespec="seconds"), "hash": text_hash,
                    "text": text[:PREVIEW_CHARS], "setting": setting, "k": k,
                    "baseline_score": None, "rag_score": rag["score"],
                    "citations": rag.get("citations", 0),
                    "nearest_distance": round(neighbours[0][1], 4) if neighbours else None,
                    "top_category": neighbours[0][0] if neighbours else None,
                })
                del stats["recent"][RECENT_MAX:]

            for mode, run in (("baseline", baseline), ("rag", rag)):
                if run is None:
                    continue
                stats["totals"][f"{mode}_runs"] += 1
                if run["error"] or run["score"] is None:
                    stats["totals"]["errors"] += 1
                else:
                    stats["verdicts"][mode][_verdict(run["score"])] += 1
                    stats["score_hist"][mode][_bin(run["score"], SCORE_BINS, 1.0)] += 1
                stats["latency"][mode]["n"] += 1
                stats["latency"][mode]["seconds"] = round(stats["latency"][mode]["seconds"] + run["seconds"], 3)
                stats["prompt_chars"][mode]["n"] += 1
                stats["prompt_chars"][mode]["chars"] += run["prompt_chars"]

            if rag is not None:
                stats["citations"]["rag_runs"] += 1
                n_cited = rag.get("citations", 0)
                stats["citations"]["runs_with_citations"] += 1 if n_cited else 0
                stats["citations"]["total_cited"] += n_cited

            if baseline is not None and rag is None and baseline["score"] is not None:
                # Baseline run on demand: pair it with the latest matching retrieval query.
                for entry in stats["recent"]:
                    if entry["hash"] == text_hash and entry["setting"] == setting and entry["baseline_score"] is None:
                        entry["baseline_score"] = baseline["score"]
                        if entry["rag_score"] is not None:
                            b, r = _verdict(baseline["score"]), _verdict(entry["rag_score"])
                            stats["agreement"]["pairs"] += 1
                            if b == r:
                                stats["agreement"]["same_verdict"] += 1
                            elif r == "anomaly":
                                stats["agreement"]["rag_flips_to_anomaly"] += 1
                            else:
                                stats["agreement"]["rag_flips_to_normal"] += 1
                        break
            elif baseline is not None and rag is not None and baseline["score"] is not None and rag["score"] is not None:
                stats["recent"][0]["baseline_score"] = baseline["score"]
                b, r = _verdict(baseline["score"]), _verdict(rag["score"])
                stats["agreement"]["pairs"] += 1
                if b == r:
                    stats["agreement"]["same_verdict"] += 1
                elif r == "anomaly":
                    stats["agreement"]["rag_flips_to_anomaly"] += 1
                else:
                    stats["agreement"]["rag_flips_to_normal"] += 1

            stats["updated_at"] = now.isoformat(timespec="seconds")
            self._write_locked(stats)
