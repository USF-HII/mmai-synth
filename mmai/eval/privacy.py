# =============================================
# FILE: mmai/eval/privacy.py 
# =============================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

@dataclass
class NNPrivacyConfig:
    k: int = 5
    metric: str = "euclidean"


def nearest_neighbor_stats(real: pd.DataFrame, fake: pd.DataFrame, cfg: NNPrivacyConfig = NNPrivacyConfig()) -> Dict[str, Any]:
    # Align columns and numeric-only for distance
    cols = sorted(set(real.columns).intersection(fake.columns))
    r = real[cols].select_dtypes(include=[np.number]).fillna(0.0).to_numpy()
    f = fake[cols].select_dtypes(include=[np.number]).fillna(0.0).to_numpy()
    if r.size == 0 or f.size == 0:
        return {"error": "no numeric overlap"}
    nn = NearestNeighbors(n_neighbors=min(cfg.k, len(r)), metric=cfg.metric)
    nn.fit(r)
    dists, idxs = nn.kneighbors(f)
    # Use 1-NN distribution for disclosure heuristics
    first = dists[:, 0]
    return {
        "mean_1nn": float(first.mean()),
        "min_1nn": float(first.min()),
        "p_below_1pct": float((first < np.percentile(first, 1)).mean()),
        "histogram": np.histogram(first, bins=20)[0].tolist(),
    }


def attribute_disclosure_risk(real: pd.DataFrame, fake: pd.DataFrame, quasi_cols, target_col) -> Dict[str, Any]:
    # Re-identification proxy: given quasi-identifiers in fake, can we guess target from real via 1-NN?
    cols = list(quasi_cols) + [target_col]
    real2 = real[cols].dropna()
    fake2 = fake[cols].dropna()
    if real2.empty or fake2.empty:
        return {"error": "insufficient rows after dropna"}
    Xr = real2[quasi_cols].select_dtypes(include=[np.number]).to_numpy()
    yr = real2[target_col].to_numpy()
    Xf = fake2[quasi_cols].select_dtypes(include=[np.number]).to_numpy()
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(Xr)
    dist, idx = nn.kneighbors(Xf)
    pred = yr[idx[:, 0]]
    # Compare to fake target (if available) to estimate memorization risk proxy
    if target_col in fake2:
        yfake = fake2[target_col].to_numpy()
        acc = float((pred == yfake).mean())
    else:
        acc = None
    return {"1nn_match_rate": acc, "median_dist": float(np.median(dist))}