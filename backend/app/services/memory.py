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
        self._vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        self._texts: list[str] = []
        self._categories: list[str] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record["label"] != 0:
                    raise ValueError("train split must be normal-only (label 0)")
                self._texts.append(record["text"])
                self._categories.append(record["original_label"])
        if len(self._texts) != len(self._vectors):
            raise ValueError(
                f"row mismatch: {len(self._vectors)} vectors vs {len(self._texts)} articles"
            )

    def __len__(self) -> int:
        return len(self._texts)

    def search(self, query_vector: np.ndarray, k: int) -> list[Neighbour]:
        """The k nearest normal train articles, nearest first."""
        q = query_vector.astype(np.float32)
        q = q / np.linalg.norm(q)
        sims = self._vectors @ q
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [
            Neighbour(
                row=int(r),
                doc_id=f"trn_{int(r):05d}",
                category=self._categories[r],
                text=self._texts[r],
                distance=float(1.0 - sims[r]),
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
