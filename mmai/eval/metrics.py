from typing import Dict, Any, List
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, pearsonr

def tabular_report(real: pd.DataFrame, fake: pd.DataFrame) -> Dict[str, Any]:
    cols = sorted(set(real.columns).intersection(fake.columns))
    out = {"n_cols": len(cols), "per_column": {}, "pairwise_corr_delta_mean": None}
    # KS/chi-square style: use KS for numeric; for non-numeric compare value proportions
    for c in cols:
        if pd.api.types.is_numeric_dtype(real[c]) and pd.api.types.is_numeric_dtype(fake[c]):
            r = real[c].dropna().to_numpy()
            f = fake[c].dropna().to_numpy()
            stat = ks_2samp(r, f).statistic
            out["per_column"][c] = {"ks_stat": float(stat)}
        else:
            rvc = (real[c].value_counts(normalize=True)).to_dict()
            fvc = (fake[c].value_counts(normalize=True)).to_dict()
            keys = set(rvc) | set(fvc)
            l1 = sum(abs(rvc.get(k, 0) - fvc.get(k, 0)) for k in keys)
            out["per_column"][c] = {"l1_cat_dist": float(l1)}
    # Correlation structure
    common_num = [c for c in cols if pd.api.types.is_numeric_dtype(real[c]) and pd.api.types.is_numeric_dtype(fake[c])]
    if len(common_num) >= 2:
        r_corr = real[common_num].corr(numeric_only=True).to_numpy()
        f_corr = fake[common_num].corr(numeric_only=True).to_numpy()
        mask = np.triu(np.ones_like(r_corr, dtype=bool), k=1)
        delta = np.abs(r_corr - f_corr)[mask]
        out["pairwise_corr_delta_mean"] = float(delta.mean())
    return out
