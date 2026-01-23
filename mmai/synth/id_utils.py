"""
id_utils.py

Utilities for constructing and managing global participant identifier
mappings across synthetic data modalities.

This module is responsible for producing a stable mapping between original
participant identifiers and synthetic identifiers, enabling cross-modal
linkage without exposing original identities.

Author: Kenneth Young, PhD
"""

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


def detect_id_column(df: pd.DataFrame, requested: Optional[str]) -> Optional[str]:
    """
    Identify a participant identifier column using conservative heuristics.
    """
    if requested and requested in df.columns:
        return requested

    lowered = {c.lower(): c for c in df.columns}
    if "mask_id" in lowered:
        return lowered["mask_id"]

    for c in df.columns:
        if "maskid" in c.lower():
            return c

    return None


def load_or_build_id_map(
    outdir: Path,
    id_col_hint: Optional[str],
    id_col_name: Optional[str],
    csv_paths: List[Path],
    force_new: bool = False,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Load or construct a global original_id -> synthetic_id mapping.

    ID detection uses real column names found in CSVs, but the canonical
    identifier name stored in id_col_name is ALWAYS id_col_hint.
    """
    idmap_path = outdir / "synthetic_id_map.csv"

    # --------------------------------------------------
    # Load existing map
    # --------------------------------------------------
    if idmap_path.exists() and not force_new:
        df = pd.read_csv(idmap_path)
        return df[["original_id", "synthetic_id"]].copy(), id_col_hint

    # --------------------------------------------------
    # Infer IDs from CSVs
    # --------------------------------------------------
    all_ids: list[str] = []

    for p in csv_paths:
        try:
            df_head = pd.read_csv(p, nrows=0)
            detected_col = detect_id_column(df_head, id_col_hint)
            if not detected_col:
                continue

            ids = pd.read_csv(p, usecols=[detected_col])[detected_col]
            ids = ids.dropna().astype(str).unique().tolist()
            all_ids.extend(ids)

        except Exception:
            continue

    all_ids = sorted(set(all_ids))
    if not all_ids:
        return pd.DataFrame(columns=["original_id", "synthetic_id"]), None

    # --------------------------------------------------
    # Generate synthetic IDs
    # --------------------------------------------------
    start_id = 100000
    synthetic_ids = [str(start_id + i) for i in range(len(all_ids))]

    id_map = pd.DataFrame({
        "original_id": all_ids,
        "synthetic_id": synthetic_ids,
        "id_col_name": id_col_name or "",
    })

    outdir.mkdir(parents=True, exist_ok=True)
    id_map.to_csv(idmap_path, index=False)

    return id_map[["original_id", "synthetic_id"]].copy(), id_col_name





def regenerate_synthetic_ids(
    idmap_path: Path,
    id_col_hint: str,
    start_id: int = 100000,
) -> Tuple[pd.DataFrame, str]:
    """
    Regenerate synthetic IDs in an existing synthetic_id_map.csv.

    This function:
    - Requires an existing file with original_id populated
    - Overwrites or fills the synthetic_id column deterministically
    - Sets id_col_name to id_col_hint
    - Preserves row order

    Returns:
        (id_map_df, id_col_name)
    """
    if not idmap_path.exists():
        raise FileNotFoundError(f"ID map not found: {idmap_path}")

    df = pd.read_csv(idmap_path, dtype=str)

    if "original_id" not in df.columns:
        raise ValueError("synthetic_id_map.csv must contain 'original_id'")

    n = len(df)
    if n == 0:
        raise ValueError("synthetic_id_map.csv is empty")

    # Generate deterministic numeric synthetic IDs
    synthetic_ids = [str(start_id + i) for i in range(n)]

    df["synthetic_id"] = synthetic_ids
    df["id_col_name"] = id_col_hint

    df.to_csv(idmap_path, index=False)

    return df[["original_id", "synthetic_id"]].copy(), id_col_hint

