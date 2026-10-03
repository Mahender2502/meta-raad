#!/usr/bin/env python3
"""
Build the JSON files behind the dashboard page (backend/data/).

real.json   REAL statistics of the N24 data in data/n24/ (category counts, test composition,
            nearest-train-article distances of normal vs Food test articles, the distance-only
            baseline). Computed from the NLP-ADBench BERT vectors; contains counts only, no text.
results.example.json
            The SHAPE of results.json (what the evaluation runner will write), with arbitrary
            placeholder numbers. It is an example for developers only: the dashboard never reads it,
            and it must not be copied to results.json or shown as results.

Usage:  python scripts/make_dashboard_data.py [--data-dir data/n24] [--out-dir backend/data]
Needs numpy and scikit-learn.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

NORMAL_CATEGORIES = [
    "Television", "Your Money", "Automobiles", "Science", "Economy", "Dance", "Travel", "Technology",
    "Sports", "Movies", "Music", "Real Estate", "Books", "Education", "Art & Design", "Theater",
    "Media", "Style", "Global Business", "Well", "Health", "Fashion & Style", "Opinion",
]


def read_meta(path: Path) -> tuple[np.ndarray, np.ndarray]:
    cats, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cats.append(r["original_label"])
            labels.append(r["label"])
    return np.array(cats), np.array(labels)


def unit(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def nearest(test: np.ndarray, train: np.ndarray, block: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Cosine distance to, and row of, the nearest train article for every test article."""
    dist, rows = [], []
    for s in range(0, len(test), block):
        sim = test[s : s + block] @ train.T
        idx = sim.argmax(1)
        rows.append(idx)
        dist.append(1.0 - sim[np.arange(len(idx)), idx])
    return np.concatenate(dist), np.concatenate(rows)


def real(data_dir: Path) -> dict:
    d = "N24News"
    tr_cat, tr_lab = read_meta(data_dir / f"{d}_train_data.jsonl")
    te_cat, te_lab = read_meta(data_dir / f"{d}_test_data.jsonl")
    tr = unit(np.load(data_dir / f"{d}_train_data_bert_base_uncased_feature.npy"))
    te = unit(np.load(data_dir / f"{d}_test_data_bert_base_uncased_feature.npy"))
    assert tr_lab.max() == 0 and len(tr) == len(tr_cat) and len(te) == len(te_cat)

    dist, rows = nearest(te, tr)
    anomaly = te_lab == 1
    edges = np.linspace(0.0, 0.30, 31)

    def hist(x):
        h, _ = np.histogram(np.clip(x, 0, 0.2999), bins=edges)
        return (100 * h / h.sum()).round(3).tolist()

    c_tr, c_te = Counter(tr_cat), Counter(te_cat)
    food_nn = Counter(tr_cat[rows[anomaly]]).most_common(8)
    return {
        "source": "NLP-ADBench N24News, precomputed bert-base-uncased vectors",
        "train_articles": int(len(tr)),
        "test_articles": int(len(te)),
        "test_normal": int((~anomaly).sum()),
        "test_anomaly": int(anomaly.sum()),
        "anomaly_category": "Food",
        "categories": [
            {"name": c, "train": c_tr[c], "test": c_te[c]} for c in sorted(NORMAL_CATEGORIES, key=lambda c: -c_tr[c])
        ],
        "distance_hist": {
            "bin_edges": [round(float(e), 3) for e in edges],
            "normal_pct": hist(dist[~anomaly]),
            "anomaly_pct": hist(dist[anomaly]),
            "mean_normal": round(float(dist[~anomaly].mean()), 4),
            "mean_anomaly": round(float(dist[anomaly].mean()), 4),
        },
        "food_nearest_categories": [{"category": c, "count": n} for c, n in food_nn],
        "distance_only_baseline": {
            "definition": "cosine distance to the nearest train article; test Food = 1",
            "auroc": round(float(roc_auc_score(te_lab, dist)), 4),
            "auprc": round(float(average_precision_score(te_lab, dist)), 4),
            "auprc_random": round(float(te_lab.mean()), 4),
        },
    }


def mock(real_stats: dict) -> dict:
    rng = np.random.default_rng(42)
    ks = [0, 1, 3, 5]
    # Arbitrary placeholder numbers (not results, and not taken from any paper).
    llm = {"normal_only": [0.74, 0.77, 0.81, 0.80], "normal_anomaly": [0.90, 0.91, 0.92, 0.92]}
    fused = {"normal_only": [0.76, 0.80, 0.84, 0.83], "normal_anomaly": [0.91, 0.92, 0.93, 0.93]}
    dist_only = {"normal_only": 0.66, "normal_anomaly": 0.66}
    table = []
    for setting in ("normal_only", "normal_anomaly"):
        for i, k in enumerate(ks):
            for score, vals in (("llm", llm), ("fused", fused)):
                table.append({"setting": setting, "k": k, "score": score,
                              "auroc": vals[setting][i], "auprc": round(vals[setting][i] * 0.55, 2)})
        table.append({"setting": setting, "k": None, "score": "distance_only",
                      "auroc": dist_only[setting], "auprc": 0.17})

    def beta_hist(a, b, n=20):
        h, _ = np.histogram(rng.beta(a, b, 4000), bins=np.linspace(0, 1, n + 1))
        return (100 * h / h.sum()).round(2).tolist()

    score_hist = {
        "bin_edges": [round(x, 2) for x in np.linspace(0, 1, 21)],
        "k0": {"normal": beta_hist(1.0, 7), "anomaly": beta_hist(3, 2.5)},
        "k3": {"normal": beta_hist(1.0, 9), "anomaly": beta_hist(4.5, 2)},
    }
    cats = [c["name"] for c in real_stats["categories"]]
    hurt = {"Travel", "Style", "Fashion & Style", "Well"}  # where retrieval is shown to hurt
    fa = []
    for c in cats:
        base = float(rng.uniform(0.03, 0.14))
        fa.append({"category": c, "k0": round(base, 3),
                   "k3": round(base * (1.7 if c in hurt else float(rng.uniform(0.45, 0.8))), 3)})
    return {
        "note": "EXAMPLE FORMAT with placeholder numbers; not results. Not read by the dashboard.",
        "ks": ks,
        "table": table,
        "score_hist": score_hist,
        "false_alarm_by_category": fa,
        "g_score": {"without_retrieval": 0.31, "with_retrieval": 0.62},
        "cost": {"k0": {"tokens": 1100, "seconds": 1.8}, "k3": {"tokens": 2400, "seconds": 2.6}},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data/n24"))
    p.add_argument("--out-dir", type=Path, default=Path("backend/data"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    r = real(args.data_dir)
    (args.out_dir / "real.json").write_text(json.dumps(r, indent=1))
    (args.out_dir / "results.example.json").write_text(json.dumps(mock(r), indent=1))
    b = r["distance_only_baseline"]
    print(f"real.json: {r['train_articles']} train, {r['test_articles']} test ({r['test_anomaly']} Food); "
          f"distance-only AUROC {b['auroc']}, AUPRC {b['auprc']}")
    print(f"results.example.json: format example only (not read by the dashboard). Written to {args.out_dir}/")


if __name__ == "__main__":
    main()
