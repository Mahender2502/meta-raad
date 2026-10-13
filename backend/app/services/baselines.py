"""Unsupervised candidate adapters; always higher score = more anomalous."""
import importlib
import random
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors


def score_baseline(name, train, test, seed=42, epochs=30):
    random.seed(seed)
    np.random.seed(seed)
    if name == "knn":
        model = NearestNeighbors(n_neighbors=min(5,len(train)), metric="cosine").fit(train)
        scores = np.concatenate([model.kneighbors(test[i:i+128])[0].mean(axis=1) for i in range(0,len(test),128)])
        return scores, {"metric": "cosine", "neighbors": min(5,len(train))}
    if name == "iforest":
        model = IsolationForest(n_estimators=100, random_state=seed, n_jobs=1).fit(train)
        return -model.score_samples(test), model.get_params()
    if name == "lof":
        model = LocalOutlierFactor(n_neighbors=min(20,len(train)-1), novelty=True, n_jobs=1).fit(train)
        return -model.score_samples(test), model.get_params()
    mapping = {"ecod": ("ecod", "ECOD"), "autoencoder": ("auto_encoder", "AutoEncoder"),
               "vae": ("vae", "VAE"), "deepsvdd": ("deep_svdd", "DeepSVDD"),
               "lunar": ("lunar", "LUNAR"), "so_gaal": ("so_gaal_new", "SO_GAAL")}
    if name not in mapping:
        raise ValueError(f"Unknown detector: {name}")
    module, cls = mapping[name]
    constructor = getattr(importlib.import_module(f"pyod.models.{module}"), cls)
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    params = {}
    if name in {"autoencoder", "vae", "so_gaal"}:
        params = {"epoch_num": epochs, "device": "cpu", "verbose": 0, "random_state": seed}
    elif name == "deepsvdd":
        params = {"n_features": train.shape[1], "epochs": epochs, "random_state": seed, "verbose": 0}
    elif name == "lunar":
        params = {"n_epochs": epochs, "verbose": 0}
    model = constructor(**params).fit(train)
    return model.decision_function(test), params
