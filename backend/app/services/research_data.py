"""Validated, row-aligned NLP-ADBench data. Test labels never enter profiles."""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EMBEDDINGS = {"bert": "bert_base_uncased", "openai": "gpt_text-embedding-3-large"}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(vectors):
    x = np.asarray(vectors, dtype=np.float32)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty finite 2D embedding matrix")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("Zero-length embeddings are not valid cosine vectors")
    return x / norms


@dataclass
class Split:
    records: list
    vectors: np.ndarray
    text_path: Path
    vector_path: Path

    @property
    def labels(self):
        return np.array([r["label"] for r in self.records], dtype=int)


def load_split(directory, prefix, split, embedding="bert"):
    if split not in {"train", "test"} or embedding not in EMBEDDINGS:
        raise ValueError("Unknown split or embedding")
    directory = Path(directory)
    text_path = directory / f"{prefix}_{split}_data.jsonl"
    vector_path = directory / f"{prefix}_{split}_data_{EMBEDDINGS[embedding]}_feature.npy"
    records = []
    with text_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            if r.get("label") not in (0, 1) or not isinstance(r.get("text"), str) or not r["text"].strip():
                raise ValueError(f"Invalid record at {text_path}:{i+1}")
            if "original_label" not in r:
                raise ValueError("original_label category is required")
            if split == "train" and r["label"] != 0:
                raise ValueError("Benchmark training memory must contain normal records only")
            records.append(r)
    vectors = np.load(vector_path, mmap_mode="r", allow_pickle=False)
    if vectors.ndim != 2 or len(records) != len(vectors) or len(records) < 2:
        raise ValueError("Text/vector row mismatch or too few records")
    for start in range(0, len(vectors), 1024):
        normalize(vectors[start:start+1024])
    return Split(records, vectors, text_path, vector_path)


def text_hash(text):
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()


def eligible_test_rows(train, test):
    """Exclude exact normalized text overlap and duplicate test texts before sampling."""
    seen = {text_hash(r["text"]) for r in train.records}
    keep, overlap, duplicates = [], 0, 0
    test_seen = set()
    for i, r in enumerate(test.records):
        h = text_hash(r["text"])
        if h in seen:
            overlap += 1
        elif h in test_seen:
            duplicates += 1
        else:
            keep.append(i)
        test_seen.add(h)
    return np.array(keep), {"train_test_overlap_removed": overlap, "test_duplicates_removed": duplicates}
