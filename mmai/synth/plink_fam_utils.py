"""
plink_fam_utils.py

Utilities for rewriting PLINK .fam files to:
- apply synthetic IID/FID mappings
- add deterministic placeholder parent IDs
- assign sex from demographics when available, otherwise deterministic fallback
- set phenotype from demographics if available, otherwise -9

Author: Kenneth Young, PhD
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _normalize_sex_to_plink(values: pd.Series) -> pd.Series:
    """
    Normalize common sex codings to PLINK (1=male, 2=female, 0=unknown).
    """
    s = values.astype(str).str.strip().str.lower()

    mapping = {
        "1": "1",
        "m": "1",
        "male": "1",
        "man": "1",
        "2": "2",
        "f": "2",
        "female": "2",
        "woman": "2",
        "0": "0",
        "unk": "0",
        "unknown": "0",
        "na": "0",
        "nan": "0",
        "none": "0",
        "": "0",
    }
    return s.map(mapping).fillna("0")


def rewrite_fam(
    fam_in: Path,
    fam_out: Path,
    id_map_path: Path,
    demographics_path: Optional[Path] = None,
    demographics_id_col: Optional[str] = None,
    demographics_sex_col: Optional[str] = None,
    demographics_pheno_col: Optional[str] = None,
    pheno_default: str = "-9",
    fill_missing_sex_randomly: bool = True,
    sex_random_seed: int = 13,
    parents_mode: str = "placeholders",
) -> None:
    """
    Rewrite a PLINK .fam file using the synthetic mapping.

    parents_mode:
      - "placeholders": PID/MID are deterministic placeholders derived from synthetic_family_id
      - "zeros": PID/MID are 0
    """
    fam = pd.read_csv(fam_in, sep=r"\s+", header=None, dtype=str)
    if fam.shape[1] < 6:
        raise ValueError(f"Expected 6 columns in .fam, found {fam.shape[1]}")

    fam.columns = ["FID", "IID", "PID", "MID", "SEX", "PHENO"]

    idmap = pd.read_csv(id_map_path, dtype=str)

    # Normalize mapping columns (supports either naming convention)
    if {"original_iid", "synthetic_iid"}.issubset(idmap.columns):
        idmap = idmap.rename(columns={"original_iid": "original_id", "synthetic_iid": "synthetic_id"})
    if not {"original_id", "synthetic_id"}.issubset(idmap.columns):
        raise ValueError("ID map must contain original_id and synthetic_id")

    # Ensure family columns exist
    if not {"synthetic_family_id", "synthetic_father_id", "synthetic_mother_id", "original_family_id"}.issubset(idmap.columns):
        raise ValueError(
            "ID map is missing family columns. Run ensure_family_ids_in_map() "
            "or rebuild the map with the updated id_utils.py."
        )

    # Apply IID mapping (original IID -> synthetic IID)
    map_iid = dict(zip(idmap["original_id"].astype(str), idmap["synthetic_id"].astype(str)))
    map_fid = dict(zip(idmap["original_id"].astype(str), idmap["synthetic_family_id"].astype(str)))
    map_pid = dict(zip(idmap["original_id"].astype(str), idmap["synthetic_father_id"].astype(str)))
    map_mid = dict(zip(idmap["original_id"].astype(str), idmap["synthetic_mother_id"].astype(str)))

    original_iid = fam["IID"].astype(str)

    fam["IID"] = original_iid.map(map_iid).fillna(fam["IID"])
    fam["FID"] = original_iid.map(map_fid).fillna(fam["FID"])

    # Parents
    if parents_mode == "placeholders":
        fam["PID"] = original_iid.map(map_pid).fillna("0")
        fam["MID"] = original_iid.map(map_mid).fillna("0")
    elif parents_mode == "zeros":
        fam["PID"] = "0"
        fam["MID"] = "0"
    else:
        raise ValueError("parents_mode must be 'placeholders' or 'zeros'")

    # Phenotype default
    fam["PHENO"] = pheno_default

    # Sex + phenotype from demographics, if provided
    if demographics_path is not None and demographics_path.exists():
        demo = pd.read_csv(demographics_path, dtype=str)

        # If user did not pass cols, try common guesses
        if demographics_id_col is None:
            for cand in ("synthetic_id", "IID", "participant_id", "mask_id", "id"):
                if cand in demo.columns:
                    demographics_id_col = cand
                    break

        if demographics_sex_col is None:
            for cand in ("sex", "Sex", "gender", "Gender"):
                if cand in demo.columns:
                    demographics_sex_col = cand
                    break

        if demographics_pheno_col is None:
            for cand in ("phenotype", "Phenotype", "pheno", "Pheno", "case_control"):
                if cand in demo.columns:
                    demographics_pheno_col = cand
                    break

        if demographics_id_col is not None:
            demo = demo.copy()
            demo[demographics_id_col] = demo[demographics_id_col].astype(str)

            # SEX merge
            if demographics_sex_col is not None and demographics_sex_col in demo.columns:
                tmp = demo[[demographics_id_col, demographics_sex_col]].copy()
                tmp[demographics_sex_col] = _normalize_sex_to_plink(tmp[demographics_sex_col])
                fam = fam.merge(tmp, how="left", left_on="IID", right_on=demographics_id_col)

                fam["SEX"] = fam[demographics_sex_col].fillna("0").astype(str)
                fam = fam.drop(columns=[c for c in (demographics_id_col, demographics_sex_col) if c in fam.columns])

            # PHENO merge
            if demographics_pheno_col is not None and demographics_pheno_col in demo.columns:
                tmp = demo[[demographics_id_col, demographics_pheno_col]].copy()
                fam = fam.merge(tmp, how="left", left_on="IID", right_on=demographics_id_col)
                fam["PHENO"] = fam[demographics_pheno_col].fillna(fam["PHENO"]).astype(str)
                fam = fam.drop(columns=[c for c in (demographics_id_col, demographics_pheno_col) if c in fam.columns])

    # Fill missing sex deterministically if requested
    if fill_missing_sex_randomly:
        missing = fam["SEX"].isin(["0", "", "nan", "None", "none"])
        if missing.any():
            rng = np.random.default_rng(sex_random_seed)
            fam.loc[missing, "SEX"] = rng.integers(1, 3, size=int(missing.sum())).astype(str)

    # Write out
    fam_out.parent.mkdir(parents=True, exist_ok=True)
    fam.to_csv(fam_out, sep="\t", header=False, index=False)
