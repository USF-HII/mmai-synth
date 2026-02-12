"""
id_utils.py

Utilities for constructing and managing global participant identifier
mappings across synthetic data modalities.

This module produces a stable mapping between original identifiers and
synthetic identifiers, enabling cross-modal linkage without exposing
original identities.

Supports multiple entity types:
- participant: numeric IDs in a reserved range (default 100000-199999)
- family: masked IDs like FAM000001

Author: Kenneth Young, PhD
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd


# ----------------------------
# Column detection
# ----------------------------
def detect_id_column(df: pd.DataFrame, requested: Optional[str]) -> Optional[str]:
    """
    Identify a participant identifier column using conservative heuristics.

    Behavior:
    - If requested exists exactly, use it.
    - Else do case-insensitive match for requested.
    - Else fall back to mask_id / maskid patterns.
    """
    if requested and requested in df.columns:
        return requested

    if requested:
        lowered = {c.lower(): c for c in df.columns}
        req = requested.lower()
        if req in lowered:
            return lowered[req]

    lowered = {c.lower(): c for c in df.columns}
    if "mask_id" in lowered:
        return lowered["mask_id"]
    if "maskid" in lowered:
        return lowered["maskid"]

    for c in df.columns:
        cl = c.lower()
        if "maskid" in cl or "mask_id" in cl:
            return c

    return None


# ----------------------------
# Internal normalization
# ----------------------------
def _standardize_id_map(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure required columns exist and are string dtype.
    """
    df = df.copy()

    if "original_id" not in df.columns:
        # allow legacy "original"
        if "original" in df.columns:
            df = df.rename(columns={"original": "original_id"})
        else:
            df["original_id"] = ""

    if "synthetic_id" not in df.columns:
        df["synthetic_id"] = ""

    if "entity_type" not in df.columns:
        df["entity_type"] = "participant"

    if "id_col_name" not in df.columns:
        df["id_col_name"] = ""

    for c in ["original_id", "synthetic_id", "entity_type", "id_col_name"]:
        df[c] = df[c].astype(str)

    return df


def _next_available_numeric(
    used: set[int],
    start_id: int,
    end_id: int,
) -> int:
    x = start_id
    while x in used:
        x += 1
    if x > end_id:
        raise ValueError(f"Ran out of numeric synthetic IDs in range {start_id}-{end_id}.")
    return x


# ----------------------------
# Public API
# ----------------------------
def load_or_build_id_map(
    outdir: Path,
    id_col_hint: Optional[str],
    id_col_name: Optional[str],
    csv_paths: List[Path],
    force_new: bool = False,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Load or construct a global original_id -> synthetic_id mapping.

    Parameters
    ----------
    id_col_hint : str
        Column name hint used ONLY for detecting ID column in source CSVs.
        Example data uses 'maskid'.
    id_col_name : str
        Canonical column name that will appear in synthetic outputs (and
        written into synthetic_id_map.csv as id_col_name).
        Example: 'MAI_T1D_maskid'

    Returns
    -------
    (id_map_df, id_col_name_used)
    """
    outdir = Path(outdir)
    idmap_path = outdir / "synthetic_id_map.csv"

    # --------------------------------------------------
    # Load existing map
    # --------------------------------------------------
    if idmap_path.exists() and not force_new:
        df = pd.read_csv(idmap_path, dtype=str)
        df = _standardize_id_map(df)

        # If caller provided a canonical name, enforce it
        if id_col_name:
            df["id_col_name"] = str(id_col_name)

        outdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(idmap_path, index=False)
        return df, (id_col_name or (df["id_col_name"].iloc[0] if len(df) else None))

    # --------------------------------------------------
    # Infer participant IDs from CSVs
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
        df_empty = pd.DataFrame(columns=["original_id", "synthetic_id", "entity_type", "id_col_name"])
        return df_empty, None

    # --------------------------------------------------
    # Generate participant synthetic IDs (numeric)
    # --------------------------------------------------
    start_id = 100000
    synthetic_ids = [str(start_id + i) for i in range(len(all_ids))]

    id_map = pd.DataFrame(
        {
            "original_id": all_ids,
            "synthetic_id": synthetic_ids,
            "entity_type": "participant",
            "id_col_name": (id_col_name or "MAI_T1D_maskid"),
        }
    )

    outdir.mkdir(parents=True, exist_ok=True)
    id_map.to_csv(idmap_path, index=False)
    return id_map, (id_col_name or "MAI_T1D_maskid")


def extend_id_map_inplace(
    id_map_df: pd.DataFrame,
    original_ids: Iterable[str],
    *,
    entity_type: str = "participant",
    start_id: int = 100000,
    end_id: int = 199999,
    family_prefix: str = "FAM",
) -> pd.DataFrame:
    """
    Extend an existing ID map with missing IDs.

    - participant: assigns next available numeric in [start_id, end_id]
    - family: assigns IDs like FAM000001

    Returns updated dataframe (copy).
    """
    df = _standardize_id_map(id_map_df)

    original_ids = [str(x) for x in original_ids if str(x) and str(x) != "nan"]
    if not original_ids:
        return df

    existing = set(df["original_id"].astype(str).tolist())
    missing = [x for x in original_ids if x not in existing]
    if not missing:
        return df

    if entity_type == "participant":
        used = set(
            int(x)
            for x in df.loc[df["entity_type"] == "participant", "synthetic_id"]
            .replace("", pd.NA)
            .dropna()
            .astype(str)
            .tolist()
            if str(x).isdigit()
        )
        new_rows = []
        for oid in missing:
            nxt = _next_available_numeric(used, start_id, end_id)
            used.add(nxt)
            new_rows.append(
                {
                    "original_id": oid,
                    "synthetic_id": str(nxt),
                    "entity_type": "participant",
                    "id_col_name": df["id_col_name"].iloc[0] if len(df) else "MAI_T1D_maskid",
                }
            )
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        return df

    if entity_type == "family":
        # Determine next family sequence
        fam_syn = (
            df.loc[df["entity_type"] == "family", "synthetic_id"]
            .astype(str)
            .tolist()
        )
        seq_used = []
        for s in fam_syn:
            m = None
            if s.startswith(family_prefix):
                tail = s.replace(family_prefix, "")
                if tail.isdigit():
                    seq_used.append(int(tail))
        next_seq = (max(seq_used) + 1) if seq_used else 1

        new_rows = []
        for oid in missing:
            syn = f"{family_prefix}{next_seq:06d}"
            next_seq += 1
            new_rows.append(
                {
                    "original_id": oid,
                    "synthetic_id": syn,
                    "entity_type": "family",
                    "id_col_name": df["id_col_name"].iloc[0] if len(df) else "MAI_T1D_maskid",
                }
            )
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        return df

    raise ValueError(f"Unsupported entity_type={entity_type!r}")


def make_lookup(df: pd.DataFrame, entity_type: str = "participant") -> Dict[str, str]:
    """
    Build a dict original_id -> synthetic_id for a given entity type.
    """
    df = _standardize_id_map(df)
    sub = df[df["entity_type"] == entity_type]
    return dict(zip(sub["original_id"].astype(str), sub["synthetic_id"].astype(str)))


def write_id_map(
    outdir_or_path: Union[str, Path, pd.DataFrame],
    id_map_df_or_path: Union[pd.DataFrame, str, Path, None] = None,
    *,
    id_col_hint: Optional[str] = None,
    id_col_name: Optional[str] = None,
) -> Path:
    """
    Write synthetic_id_map.csv.

    Supports BOTH call styles to stop the "DataFrame passed as path" failures:
    1) write_id_map(outdir, id_map_df, id_col_hint=..., id_col_name=...)
    2) write_id_map(id_map_df, out_path, id_col_hint=..., id_col_name=...)

    Returns path written.
    """
    # Detect which signature was used
    if isinstance(outdir_or_path, pd.DataFrame):
        df = outdir_or_path
        if id_map_df_or_path is None:
            raise ValueError("If first argument is a DataFrame, second must be a path.")
        target = Path(id_map_df_or_path)
    else:
        target = Path(outdir_or_path)
        if id_map_df_or_path is None or not isinstance(id_map_df_or_path, pd.DataFrame):
            raise ValueError("write_id_map(outdir, id_map_df, ...) requires a DataFrame as second argument.")
        df = id_map_df_or_path

        # If outdir is a directory, write synthetic_id_map.csv inside it
        if target.suffix.lower() != ".csv":
            target = target / "synthetic_id_map.csv"

    df = _standardize_id_map(df)

    # Canonicalize id_col_name
    canonical = id_col_name or (df["id_col_name"].iloc[0] if len(df) else None) or id_col_hint or "MAI_T1D_maskid"
    df["id_col_name"] = str(canonical)

    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    return target
