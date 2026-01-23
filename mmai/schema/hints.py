# =============================================
# FILE: mmai/schema/hints.py (NEW)
# =============================================
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import pandas as pd

def load_hints(path: Optional[Path]) -> Dict:
    if not path:
        return {}
    with Path(path).open("r") as f:
        return yaml.safe_load(f) or {}


def apply_hints(df: pd.DataFrame, table_name: str, hints: Dict):
    tbl = (hints.get("tables", {}) or {}).get(table_name, {})
    cat = set(tbl.get("categorical", []) or [])
    ids = set(tbl.get("id_columns", []) or [])
    passthrough = df[list(ids.intersection(df.columns))].copy() if ids else pd.DataFrame(index=df.index)
    model_df = df.drop(columns=list(ids.intersection(df.columns)), errors="ignore").copy()
    for c in cat:
        if c in model_df.columns:
            model_df[c] = model_df[c].astype("category")
    discrete_columns = [c for c in model_df.columns if str(model_df[c].dtype) == "category"]
    return model_df, passthrough, discrete_columns