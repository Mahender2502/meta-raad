"""Training-only property profiles with explicit geometric definitions."""
import re
from collections import Counter

import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from app.services.research_data import normalize


def profile_training(records, vectors, *, seed=42, max_samples=1000):
    if len(records) != len(vectors) or len(records) < 3 or max_samples < 3:
        raise ValueError("At least three aligned training records required")
    if any(r["label"] != 0 for r in records):
        raise ValueError("Profiles accept normal training records only")
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(records), min(max_samples, len(records)), replace=False))
    sample = [records[i] for i in rows]
    x = normalize(vectors[rows])
    categories = np.array([str(r["original_label"]) for r in sample])
    counts = Counter(categories)
    tokens = [re.findall(r"\b\w+\b", r["text"].lower()) for r in sample]
    lengths = np.array([len(t) for t in tokens])
    vocab = Counter(t for ts in tokens for t in ts)
    centers, compactness = [], []
    for category in sorted(counts):
        group = x[categories == category]
        center = group.mean(axis=0)
        norm = np.linalg.norm(center)
        if norm <= 1e-10:
            continue
        center /= norm
        centers.append(center)
        compactness.extend(np.clip(1 - group @ center, 0, 2).tolist())
    centers = np.array(centers)
    separation = None
    if len(centers) > 1:
        distances = np.clip(1-centers @ centers.T, 0, 2)
        separation = float(distances[np.triu_indices(len(centers), 1)].mean())
    k = min(6, len(x))
    distances, _ = NearestNeighbors(n_neighbors=k, metric="cosine").fit(x).kneighbors(x)
    local = np.maximum(distances[:, 1:].mean(axis=1), 0)
    eigen = np.linalg.svd(x - x.mean(axis=0), compute_uv=False) ** 2
    total = eigen.sum()
    probs = np.array(list(counts.values())) / len(sample)
    f = {
        "n_train": len(records), "embedding_dimension": int(x.shape[1]),
        "profile_sample_size": len(sample), "normal_category_count": len(counts),
        "mean_tokens": float(lengths.mean()), "std_tokens": float(lengths.std()),
        "min_tokens": int(lengths.min()), "max_tokens": int(lengths.max()),
        "vocabulary_size": len(vocab), "type_token_ratio": len(vocab)/max(1,sum(vocab.values())),
        "rare_token_fraction": sum(n for n in vocab.values() if n == 1)/max(1,sum(vocab.values())),
        "length_skewness": float(skew(lengths)) if lengths.std() else 0.,
        "length_excess_kurtosis": float(kurtosis(lengths)) if lengths.std() else 0.,
        "category_entropy": float(-(probs*np.log(probs)).sum()),
        "intra_class_compactness": float(np.mean(compactness)) if compactness else None,
        "inter_class_separability": separation,
        "category_silhouette": float(silhouette_score(x, categories, metric="cosine")) if 1 < len(counts) < len(x) else None,
        "effective_dimension": float(total**2 / (eigen@eigen)) if total > 1e-12 else 0.,
        "pca_dimensions_95": int(np.searchsorted(np.cumsum(eigen)/total, .95)+1) if total > 1e-12 else 0,
        "mean_knn_cosine_distance": float(local.mean()),
        "density_variation": float(local.std()/local.mean()) if local.mean() > 1e-10 else 0.,
        "normal_anomaly_separability": None,
        "test_anomaly_ratio": None,
    }
    return {"schema_version": 1, "split": "train", "seed": seed, "sample_rows": rows.tolist(),
            "normal_category_sizes_sample": dict(counts), "features": f,
            "definitions": {
                "intra_class_compactness": "Mean cosine distance to own normal-category centroid; smaller is tighter.",
                "inter_class_separability": "Mean pairwise cosine distance between normal-category centroids; not normal-versus-anomaly separation.",
                "effective_dimension": "Participation ratio (sum squared singular values)^2 / sum fourth powers, centered unit vectors.",
                "density_variation": "Coefficient of variation of mean five-neighbor cosine distance, excluding self.",
                "unavailable": "Test prevalence and normal-anomaly separation are withheld to prevent test-label leakage.",
            }}
