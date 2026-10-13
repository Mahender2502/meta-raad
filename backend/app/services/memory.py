"""
Retrieval memory: the normal-only train split of a dataset.

Holds the benchmark's precomputed train vectors (one per article, row-aligned with the
train jsonl) plus the article text and category. Search is exact cosine kNN in NumPy;
no vector database. A retrieved article is identified by its row: doc_id = trn_{row:05d}.
"""

import json
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.core.config import settings
from app.core.datasets import DATASETS
from app.services.research_data import normalize, text_hash


@dataclass
class Neighbour:
    row: int
    doc_id: str
    category: str
    text: str
    distance: float  # cosine distance, 0 = identical direction


class TrainMemory:
    def __init__(self, vectors_path: Path, jsonl_path: Path) -> None:
        vectors = np.load(vectors_path).astype(np.float32)
        self._vectors = normalize(vectors)
        self._texts: list[str] = []
        self._categories: list[str] = []
        self._text_hashes: list[str] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record["label"] != 0:
                    raise ValueError("train split must be normal-only (label 0)")
                self._texts.append(record["text"])
                self._text_hashes.append(text_hash(record["text"]))
                self._categories.append(record["original_label"])
        if len(self._texts) != len(self._vectors):
            raise ValueError(
                f"row mismatch: {len(self._vectors)} vectors vs {len(self._texts)} articles"
            )

    def __len__(self) -> int:
        return len(self._texts)

    def search(self, query_vector: np.ndarray, k: int, exclude_text: str | None = None) -> list[Neighbour]:
        """The k nearest normal train articles, nearest first."""
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer")
        q = np.asarray(query_vector, dtype=np.float32)
        if q.shape != (self._vectors.shape[1],):
            raise ValueError("Query dimension does not match memory")
        q = normalize(q[None, :])[0]
        sims = self._vectors @ q
        if exclude_text is not None:
            query_hash = text_hash(exclude_text)
            sims[np.array([h == query_hash for h in self._text_hashes])] = -np.inf
        eligible = np.flatnonzero(np.isfinite(sims))
        k = min(k, len(eligible))
        if not k:
            return []
        top = eligible[np.argpartition(-sims[eligible], k - 1)[:k]]
        top = top[np.argsort(-sims[top])]
        return [
            Neighbour(
                row=int(r),
                doc_id=f"trn_{int(r):05d}",
                category=self._categories[r],
                text=self._texts[r],
                distance=float(np.clip(1.0 - sims[r], 0., 2.)),
            )
            for r in top
        ]


_lock = threading.Lock()


@lru_cache(maxsize=None)
def _load(dataset: str) -> TrainMemory:
    spec = DATASETS[dataset]
    base = Path(settings.data_dir) / spec.data_subdir
    return TrainMemory(
        base / f"{spec.file_prefix}_train_data_bert_base_uncased_feature.npy",
        base / f"{spec.file_prefix}_train_data.jsonl",
    )


def get_memory(dataset: str) -> TrainMemory:
    """Lazily load a dataset's memory once per process (loading parses ~200 MB of text)."""
    with _lock:
        return _load(dataset)
