# =============================================
# FILE: mmai/eval/utility.py 
# =============================================
from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split


def train_on_synth_test_on_real(real: pd.DataFrame, fake: pd.DataFrame, target: str) -> Dict[str, Any]:
    cols = sorted(set(real.columns).intersection(fake.columns))
    cols = [c for c in cols if c != target]
    # numeric-only baseline
    Xr = real[cols].select_dtypes(include=[np.number]).fillna(0).to_numpy()
    Xf = fake[cols].select_dtypes(include=[np.number]).fillna(0).to_numpy()
    yr = real[target].to_numpy() if target in real else None
    yf = fake[target].to_numpy() if target in fake else None
    if yr is None or yf is None:
        return {"error": "target must exist in both real and fake"}
    # Train on synth
    Xtr, ytr = Xf, yf
    # Test on real
    Xte, yte = Xr, yr
    clf = LogisticRegression(max_iter=200)
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, prob)
    acc = accuracy_score(yte, clf.predict(Xte))
    return {"AUC": float(auc), "Accuracy": float(acc), "n_train": int(len(ytr)), "n_test": int(len(yte))}
