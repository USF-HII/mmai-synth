"""
tabular.py

Wrapper for SDV single-table GaussianCopula synthesis.

Author: Kenneth Young, PhD
"""

from __future__ import annotations

from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
from pathlib import Path
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer


def _as_str_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    return out


def _coerce_numeric_float64(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x = x.replace([np.inf, -np.inf], np.nan)

    for c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    x = x.astype(np.float64)
    return x


def _split_constant_columns(x: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    nunique = x.nunique(dropna=True)
    const_cols = nunique[nunique <= 1].index.tolist()
    var_cols = [c for c in x.columns if c not in const_cols]

    const_values: Dict[str, float] = {}
    for c in const_cols:
        s = x[c]
        if s.notna().any():
            const_values[c] = float(s.dropna().iloc[0])
        else:
            const_values[c] = 0.0

    return x[var_cols].copy(), const_values


def _fill_missing_deterministic(x: pd.DataFrame) -> pd.DataFrame:
    if x.empty:
        return x
    med = x.median(axis=0, skipna=True).fillna(0.0)
    return x.fillna(med)


def _build_sdv_synth(meta: Metadata, seed: int) -> GaussianCopulaSynthesizer:
    try:
        return GaussianCopulaSynthesizer(metadata=meta, enforce_rounding=True, random_state=seed)
    except TypeError:
        return GaussianCopulaSynthesizer(metadata=meta, enforce_rounding=True)


def _postprocess_genotypes(syn: pd.DataFrame, clamp_genotypes: bool) -> pd.DataFrame:
    if not clamp_genotypes or syn.empty:
        return syn
    syn2 = syn.copy()
    syn2 = syn2.round()
    syn2 = syn2.clip(lower=0, upper=2)
    return syn2


def synthesize_gaussiancopula(
    df: pd.DataFrame,
    out_dir: str,
    seed: int = 42,
    n_rows: Optional[int] = None,
    clamp_genotypes: bool = True,
) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    target_n = int(n_rows) if n_rows is not None else len(df)
    if target_n <= 0:
        raise ValueError("n_rows must be > 0 (or df must have at least 1 row)")

    orig_cols: List[str] = [str(c) for c in df.columns]

    # 1) Normalize columns and coerce to numeric float64
    x = _as_str_columns(df)
    x = _coerce_numeric_float64(x)

    # 2) Split out constant columns (kept exactly constant)
    x_var, const_values = _split_constant_columns(x)

    # Edge case: if everything is constant, just bootstrap constants
    if x_var.shape[1] == 0:
        out = pd.DataFrame({c: [v] * target_n for c, v in const_values.items()})
        out = out.reindex(columns=orig_cols, fill_value=0.0)
        out = _postprocess_genotypes(out, clamp_genotypes=clamp_genotypes)
        return out

    # 3) Fill missing deterministically
    x_fit = _fill_missing_deterministic(x_var)

    # 4) Build metadata and fit synthesizer
    
    # Build metadata (new API)
    meta = Metadata.detect_from_dataframe(data=x_fit)

    # Save metadata ONCE (write to a FILE, not a directory)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir_p / "sdv_metadata.json"
    if not meta_path.exists():
        # Prefer the simplest call for compatibility across SDV versions
        try:
            meta.save_to_json(str(meta_path))
        except TypeError:
            # Some SDV versions support mode=...
            meta.save_to_json(str(meta_path), mode="overwrite")

    synth = _build_sdv_synth(meta, seed=seed)
    synth.fit(x_fit)

    syn = synth.sample(target_n)

    # 5) Reattach constant columns (fast, avoids fragmentation)
    if const_values:
        const_df = pd.DataFrame(
            {c: np.full(target_n, v, dtype=np.float64) for c, v in const_values.items()}
        )
        syn = pd.concat([syn, const_df], axis=1)

    # 6) Restore original column order (and ensure all are present)
    syn = syn.reindex(columns=orig_cols, fill_value=0.0)

    # 7) Optional genotype clamp
    syn = _postprocess_genotypes(syn, clamp_genotypes=clamp_genotypes)

    return syn
