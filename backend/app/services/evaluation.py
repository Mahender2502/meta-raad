"""Evaluation uses held-out labels only after scores/recommendations are fixed."""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score, ndcg_score


def metrics(labels, scores, *, seed=42, bootstrap=200):
    y = np.asarray(labels)
    s = np.array([np.nan if v is None else v for v in scores], dtype=float)
    if y.ndim != 1 or s.shape != y.shape or not set(y).issubset({0, 1}):
        raise ValueError("Aligned binary labels and scores required")
    ok = np.isfinite(s)
    out = {"n_total": len(y), "n_valid": int(ok.sum()), "n_errors": int((~ok).sum()),
           "coverage": float(ok.mean()) if len(y) else 0., "auroc": None, "auprc": None,
           "ci95": None, "auprc_definition": "average_precision_score (non-interpolated AP)"}
    y, s = y[ok], s[ok]
    if len(np.unique(y)) != 2:
        return out
    out.update(auroc=float(roc_auc_score(y, s)), auprc=float(average_precision_score(y, s)), anomaly_prevalence=float(y.mean()))
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == c) for c in [0, 1]]
    samples = []
    for _ in range(bootstrap):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        samples.append([roc_auc_score(y[idx], s[idx]), average_precision_score(y[idx], s[idx])])
    if samples:
        ci = np.quantile(samples, [.025, .975], axis=0)
        out["ci95"] = {"auroc": ci[:, 0].tolist(), "auprc": ci[:, 1].tolist(), "method": "stratified percentile bootstrap", "resamples": bootstrap}
    return out


def selection_metrics(ranking, candidate_aurocs):
    if set(ranking) != set(candidate_aurocs) or len(ranking) != len(candidate_aurocs):
        raise ValueError("Ranking and measured candidate set must match")
    if not all(v is not None and np.isfinite(v) for v in candidate_aurocs.values()):
        raise ValueError("All candidates need valid held-out AUROCs")
    scores = np.array([candidate_aurocs[c] for c in ranking])
    selected = scores[0]
    best = float(scores.max())
    rank = 1 + int((scores > selected).sum())
    rho = spearmanr(-np.arange(len(scores)), scores).statistic if len(scores)>1 and scores.std()>0 else None
    return {"selected_auroc": float(selected), "best_auroc": best, "auroc_regret": best-float(selected),
            "selected_rank": rank, "top1": rank <= 1, "top3": rank <= 3, "top5": rank <= 5,
            "spearman": float(rho) if rho is not None and np.isfinite(rho) else None,
            "ndcg_at_3": float(ndcg_score([scores], [np.arange(len(scores),0,-1)], k=min(3,len(scores)))) if len(scores)>1 else 1.}
