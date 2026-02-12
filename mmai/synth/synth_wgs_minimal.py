"""
synth_wgs_minimal.py

Generate synthetic genotype data from masked PLINK input.

Key features:
- BIM is preserved from the input BIM (or its truncated subset)
- FAM is written from a real fam_out (NO dummy iid1 / sex=0 fallback)
- IIDs follow your reserved numeric range (default 100000-199999)
- If PLINK IIDs/PIDs/MIDs are not in the global ID map, the map is extended deterministically
- Family IDs (FID) are masked as FAM000001... and can be added to the mapping file
- Sex is assigned from demographics when available; otherwise sampled from observed distribution (deterministic)

Author: Kenneth Young, PhD (USF-HII)
        Dena Tewey, MPH (USF-HII)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from mmai.synth.plink_utils import (
    read_plink_prefix,
    write_plink,
    write_ped,  
    normalize_and_impute_genotypes,
)
from mmai.synth.tabular import synthesize_gaussiancopula

# This module uses extend_id_map_inplace.
from mmai.synth.id_utils import extend_id_map_inplace


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def _log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


# -----------------------------------------------------------------------------
# Demographics helpers
# -----------------------------------------------------------------------------
def _load_demographics(demographics_path: Optional[str]) -> Optional[pd.DataFrame]:
    if not demographics_path:
        return None
    p = Path(demographics_path)
    if not p.exists():
        _log(f"Demographics path does not exist: {p}")
        return None

    df = pd.read_csv(p, dtype=str)
    _log(f"Loaded demographics: {p} rows={len(df)} cols={list(df.columns)}")
    return df


def _build_sex_lookup(
    demo_df: Optional[pd.DataFrame],
    id_col_name: str,
    sex_col_candidates: Optional[List[str]] = None,
) -> Tuple[Dict[str, int], Dict[int, float]]:
    """
    Returns:
      sex_by_id: synthetic_id -> sex(1/2)
      sex_distribution: {1: p_male, 2: p_female} based on observed demographics
    """
    if sex_col_candidates is None:
        sex_col_candidates = ["sex", "Sex", "gender", "Gender"]

    sex_by_id: Dict[str, int] = {}
    dist: Dict[int, float] = {1: 0.5, 2: 0.5}

    if demo_df is None or demo_df.empty:
        return sex_by_id, dist

    if id_col_name not in demo_df.columns:
        _log(f"Demographics missing id column '{id_col_name}'. Sex will be sampled statistically.")
        return sex_by_id, dist

    sex_col = None
    for c in sex_col_candidates:
        if c in demo_df.columns:
            sex_col = c
            break

    if not sex_col:
        _log("Demographics has no Sex column. Sex will be sampled statistically.")
        return sex_by_id, dist

    tmp = demo_df[[id_col_name, sex_col]].copy()
    tmp[id_col_name] = tmp[id_col_name].astype(str)

    def _norm(v: str) -> Optional[int]:
        if v is None:
            return None
        s = str(v).strip().lower()
        if s in ("1", "m", "male"):
            return 1
        if s in ("2", "f", "female"):
            return 2
        return None

    tmp["_sex"] = tmp[sex_col].apply(_norm)
    tmp = tmp.dropna(subset=["_sex"])
    if tmp.empty:
        return sex_by_id, dist

    sex_by_id = dict(zip(tmp[id_col_name].tolist(), tmp["_sex"].astype(int).tolist()))

    counts = tmp["_sex"].value_counts().to_dict()
    total = sum(counts.values())
    if total > 0:
        p1 = counts.get(1, 0) / total
        p2 = counts.get(2, 0) / total
        if p1 == 0 and p2 > 0:
            dist = {1: 0.01, 2: 0.99}
        elif p2 == 0 and p1 > 0:
            dist = {1: 0.99, 2: 0.01}
        else:
            dist = {1: float(p1), 2: float(p2)}

    return sex_by_id, dist


def _deterministic_choice(iid: str, seed: int, probs: Dict[int, float]) -> int:
    """
    Deterministically sample sex using iid as entropy.
    """
    h = abs(hash((iid, seed))) % (2**32 - 1)
    rng = np.random.RandomState(h)
    r = rng.rand()
    p1 = probs.get(1, 0.5)
    return 1 if r < p1 else 2


# -----------------------------------------------------------------------------
# Family ID masking
# -----------------------------------------------------------------------------
def _build_family_map(fids: pd.Series, prefix: str = "FAM") -> Dict[str, str]:
    """
    Deterministic masked family IDs for any non-zero FIDs.
    """
    fids = fids.astype(str)
    uniq = sorted({x for x in fids.unique().tolist() if x and x != "0"})
    return {fid: f"{prefix}{i+1:06d}" for i, fid in enumerate(uniq)}


def _opt_add_family_rows_to_id_map(
    id_map: pd.DataFrame,
    family_map: Dict[str, str],
    id_col_name: str,
) -> pd.DataFrame:
    """
    Add family entities to synthetic_id_map in a collision-safe way.

    Canonical family rows:
      original_id   = "FAMILY::<orig_fid>"
      synthetic_id  = "FAM000001"...
      entity_type   = "family"
      id_col_name   = canonical output ID col (MAI_T1D_maskid)

    Also cleans up legacy family rows from older runs where:
      original_id = "<orig_fid>"   (no FAMILY:: prefix)
    """
    if id_map is None or id_map.empty or not family_map:
        return id_map

    df = id_map.copy()

    # Ensure required columns exist
    for col, default in [
        ("original_id", ""),
        ("synthetic_id", ""),
        ("id_col_name", id_col_name),
        ("entity_type", "participant"),
    ]:
        if col not in df.columns:
            df[col] = default

    # Force canonical id_col_name everywhere
    df["id_col_name"] = id_col_name

    # Normalize to strings
    df["original_id"] = df["original_id"].astype(str)
    df["synthetic_id"] = df["synthetic_id"].astype(str)
    df["entity_type"] = df["entity_type"].astype(str)

    # --- CLEANUP legacy family rows (no prefix) ---
    # If a row is entity_type==family and original_id is a raw fid that exists in family_map,
    # convert it to namespaced key and keep synthetic_id consistent.
    legacy_fids = set(str(k) for k in family_map.keys())

    is_legacy_family = (df["entity_type"] == "family") & (df["original_id"].isin(legacy_fids))
    if is_legacy_family.any():
        df.loc[is_legacy_family, "original_id"] = df.loc[is_legacy_family, "original_id"].apply(lambda x: f"FAMILY::{x}")
    # Remove legacy family rows from old scheme (no FAMILY:: prefix)
    # if "entity_type" in df.columns:
    #     df = df[~((df["entity_type"].astype(str) == "family") &
    #               (~df["original_id"].astype(str).str.startswith("FAMILY::")))].copy()

    # --- ADD missing canonical rows ---
    existing_keys = set(df["original_id"].tolist())
    new_rows = []
    for orig_fid, syn_fid in family_map.items():
        key = f"FAMILY::{str(orig_fid)}"
        if key in existing_keys:
            # If it exists, optionally enforce synthetic_id match (keep first, fix mismatches)
            # Fix mismatches deterministically:
            mask = (df["original_id"] == key) & (df["entity_type"] == "family")
            if mask.any():
                df.loc[mask, "synthetic_id"] = str(syn_fid)
                df.loc[mask, "id_col_name"] = id_col_name
            continue

        new_rows.append(
            {
                "original_id": key,
                "synthetic_id": str(syn_fid),
                "id_col_name": id_col_name,
                "entity_type": "family",
            }
        )

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    # --- DEDUPE ---
    # Keep one row per (original_id, entity_type). If duplicates exist, keep the last one (most recent run).
    df = df.drop_duplicates(subset=["original_id", "entity_type"], keep="last").reset_index(drop=True)

    return df




# -----------------------------------------------------------------------------
# FAM construction (NO dummy fallback)
# -----------------------------------------------------------------------------
def _build_synthetic_fam(
    fam_df: pd.DataFrame,
    iid_map: Dict[str, str],
    sex_by_syn_id: Dict[str, int],
    sex_dist: Dict[int, float],
    seed: int,
    mask_family_ids: bool = True,
    preserve_relationships: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Create a synthetic FAM based on the source FAM.
    """
    src = fam_df.copy()
    src.columns = ["fid", "iid", "pid", "mid", "sex", "pheno"]

    for c in ["fid", "iid", "pid", "mid", "sex", "pheno"]:
        src[c] = src[c].astype(str)

    family_map: Dict[str, str] = {}
    if mask_family_ids:
        family_map = _build_family_map(src["fid"])

    out = src.copy()

    # IID mapping (must exist)
    out["iid"] = out["iid"].map(iid_map)
    if out["iid"].isna().any():
        missing = src.loc[out["iid"].isna(), "iid"].astype(str).unique().tolist()
        raise RuntimeError(f"Missing IID mappings for: {missing[:10]} ...")

    # FID masking
    if mask_family_ids:
        out["fid"] = out["fid"].map(family_map).fillna("0")
    else:
        out["fid"] = out["fid"].map(iid_map).fillna(out["fid"])

    # Relationship mapping
    if preserve_relationships:
        out["pid"] = out["pid"].map(iid_map).fillna("0")
        out["mid"] = out["mid"].map(iid_map).fillna("0")
    else:
        out["pid"] = "0"
        out["mid"] = "0"

    # Sex assignment (prefer demographics keyed by synthetic IID)
    syn_iids = out["iid"].astype(str).tolist()
    sex_vals: List[str] = []
    for syn_id in syn_iids:
        if syn_id in sex_by_syn_id:
            sex_vals.append(str(int(sex_by_syn_id[syn_id])))
        else:
            sex_vals.append(str(_deterministic_choice(syn_id, seed=seed, probs=sex_dist)))
    out["sex"] = sex_vals

    # Phenotype: preserve if numeric, else -9
    ph = pd.to_numeric(out["pheno"], errors="coerce")
    out["pheno"] = ph.fillna(-9).astype(int).astype(str)

    return out[["fid", "iid", "pid", "mid", "sex", "pheno"]], family_map


# -----------------------------------------------------------------------------
# Minimal VCF writer (GT-only, deterministic)
# -----------------------------------------------------------------------------
def write_vcf_simple(geno_df: pd.DataFrame, bim_df: pd.DataFrame, out_path: Path) -> None:
    """
    Minimal VCF writer for inspection/debugging.

    - Writes GT only
    - REF=a2 ALT=a1 (consistent with your PED mapping logic in plink_utils.write_ped)
    """
    bim_snps = bim_df["snp"].astype(str).tolist()
    geno_snps = [str(c) for c in geno_df.columns.tolist()]
    if geno_snps != bim_snps:
        raise ValueError("BIM SNP order does not match genotype columns (VCF write).")

    samples = [str(i) for i in geno_df.index.tolist()]

    with open(out_path, "w", newline="\n") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")
        for s in samples:
            f.write(f"\t{s}")
        f.write("\n")

        for j, bim in enumerate(bim_df.itertuples(index=False)):
            chrom = str(bim.chrom)
            pos = str(bim.pos)
            vid = str(bim.snp)

            ref = str(bim.a2)
            alt = str(bim.a1)
            if ref in ("", "0", "nan", "None"):
                ref = "N"
            if alt in ("", "0", "nan", "None"):
                alt = "N"

            f.write(f"{chrom}\t{pos}\t{vid}\t{ref}\t{alt}\t.\tPASS\t.\tGT")

            col = geno_df.iloc[:, j]
            for g in col.tolist():
                if g is None or pd.isna(g):
                    gt = "./."
                else:
                    gi = int(g)
                    if gi == 0:
                        gt = "0/0"
                    elif gi == 1:
                        gt = "0/1"
                    elif gi == 2:
                        gt = "1/1"
                    else:
                        gt = "./."
                f.write(f"\t{gt}")
            f.write("\n")


# -----------------------------------------------------------------------------
# QC
# -----------------------------------------------------------------------------
def _run_qc_report(geno_df: pd.DataFrame, out_path: Path) -> None:
    _log("Generating QC report")
    report = []
    n_rows = geno_df.shape[0]

    for snp in geno_df.columns:
        g = pd.to_numeric(geno_df[snp], errors="coerce")
        n_obs = int(g.notna().sum())
        if n_obs == 0:
            continue
        counts = g.value_counts(dropna=True).to_dict()
        af = (counts.get(1, 0) + 2 * counts.get(2, 0)) / (2 * n_obs)
        maf = min(af, 1 - af)
        missing = 1.0 - (n_obs / n_rows)
        report.append(
            {
                "snp": str(snp),
                "maf": round(float(maf), 6),
                "missing_rate": round(float(missing), 6),
                "n_0": int(counts.get(0, 0)),
                "n_1": int(counts.get(1, 0)),
                "n_2": int(counts.get(2, 0)),
                "n_missing": int(n_rows - n_obs),
            }
        )

    pd.DataFrame(report).to_csv(out_path, index=False)
    _log(f"Wrote QC report: {out_path}")


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------
def generate_synthetic_wgs(
    plink_prefix: str,
    out_dir: str,
    global_id_map_df: Optional[pd.DataFrame] = None,
    id_col_hint: str = "maskid",
    id_col_name: str = "MAI_T1D_maskid",
    map_ids: bool = True,
    id_start: int = 100000,
    id_end: int = 199999,
    seed: int = 42,
    output_formats: Optional[List[str]] = None,
    max_snps: Optional[int] = 10000,
    demographics_path: Optional[str] = None,
    mask_family_ids: bool = True,
    preserve_relationships: bool = True,
    add_family_ids_to_id_map: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Returns:
      syn_df (index = synthetic participant IDs)
      updated_global_id_map_df (if global_id_map_df provided or created)
    """
    if output_formats is None:
        output_formats = ["matrix", "plink"]

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    _log(f"Reading PLINK prefix: {plink_prefix}")
    geno_df, bim_df, fam_df = read_plink_prefix(plink_prefix)

    _log(f"Input genotype shape: {geno_df.shape}")
    _log(f"Input FAM rows: {fam_df.shape[0]} unique IIDs: {fam_df['iid'].astype(str).nunique()}")
    _log(f"Input BIM rows: {bim_df.shape[0]} first SNP: {bim_df['snp'].iloc[0]}")

    if max_snps and geno_df.shape[1] > int(max_snps):
        max_snps = int(max_snps)
        geno_df = geno_df.iloc[:, :max_snps].copy()
        bim_df = bim_df.iloc[:max_snps].copy()
        _log(f"Truncated SNPs to max_snps={max_snps}")

    _log("Fitting GaussianCopula and sampling synthetic genotypes")
    syn_df = synthesize_gaussiancopula(geno_df, out_dir=out_dir, seed=seed, n_rows=len(geno_df))

    syn_df.index = fam_df["iid"].astype(str).tolist()
    syn_df.columns = bim_df["snp"].astype(str).tolist()

    _log(f"Post-sample synthetic shape: {syn_df.shape}")
    _log("Normalizing and imputing synthetic genotypes")
    syn_df = normalize_and_impute_genotypes(syn_df)

    demo_df = _load_demographics(demographics_path)
    sex_by_syn_id: Dict[str, int] = {}
    sex_dist: Dict[int, float] = {1: 0.5, 2: 0.5}
    if demo_df is not None:
        sex_by_syn_id, sex_dist = _build_sex_lookup(demo_df, id_col_name=id_col_name)

    updated_map: Optional[pd.DataFrame] = None
    iid_map: Dict[str, str] = {}

    if map_ids:
        fam_df.columns = ["fid", "iid", "pid", "mid", "sex", "pheno"]

        all_plink_ids = set(fam_df["iid"].astype(str).tolist())
        all_plink_ids |= set(fam_df["pid"].astype(str).tolist())
        all_plink_ids |= set(fam_df["mid"].astype(str).tolist())
        all_plink_ids = {x for x in all_plink_ids if x and x != "0"}

        _log(f"Participant-like IDs found in PLINK FAM (IID+parents): {len(all_plink_ids)}")

        if global_id_map_df is None or global_id_map_df.empty:
            sorted_ids = sorted(all_plink_ids)
            if len(sorted_ids) > (id_end - id_start + 1):
                raise ValueError("Too many IDs for reserved range.")
            updated_map = pd.DataFrame(
                {
                    "original_id": sorted_ids,
                    "synthetic_id": [str(id_start + i) for i in range(len(sorted_ids))],
                    "id_col_name": id_col_name,
                }
            )
            _log(f"Created new global ID map with {len(updated_map)} rows")
        else:
            _log("Extending provided global_id_map_df to include PLINK IDs")
            tmp = global_id_map_df.copy()
            tmp = extend_id_map_inplace(tmp, sorted(all_plink_ids), start_id=id_start, end_id=id_end)

            tmp["id_col_name"] = id_col_name
            updated_map = tmp

        lookup = dict(
            zip(
                updated_map["original_id"].astype(str),
                updated_map["synthetic_id"].astype(str),
            )
        )

        src_iids = fam_df["iid"].astype(str).tolist()
        iid_map = {iid: lookup[iid] for iid in src_iids}

        _log(f"Applied global IID masking. Example: {src_iids[0]} -> {iid_map[src_iids[0]]}")

        syn_df.index = [iid_map[iid] for iid in syn_df.index.astype(str)]
        syn_df.index.name = id_col_name
    else:
        syn_df.index.name = id_col_name

    # -------------------------
    # Write matrix
    # -------------------------
    if "matrix" in output_formats:
        matrix_path = out_dir_p / "synthetic_genotype_matrix.csv"
        syn_df.to_csv(matrix_path, index=True, index_label=id_col_name)
        _log(f"Wrote matrix: {matrix_path}")

    # -------------------------
    # Write PLINK (BIM preserved, FAM real)
    # -------------------------
    if "plink" in output_formats:
        _log("Building synthetic FAM (IDs + sex assignment)")

        if not map_ids:
            raise RuntimeError("plink output requires map_ids=True to avoid dummy IDs.")

        fam_out, family_map = _build_synthetic_fam(
            fam_df=fam_df,
            iid_map=iid_map,
            sex_by_syn_id=sex_by_syn_id,
            sex_dist=sex_dist,
            seed=seed,
            mask_family_ids=mask_family_ids,
            preserve_relationships=preserve_relationships,
        )

        _log(f"Synthetic FAM example row: {fam_out.iloc[0].tolist()}")

        if add_family_ids_to_id_map and updated_map is not None and family_map:
            updated_map = _opt_add_family_rows_to_id_map(updated_map, family_map, id_col_name=id_col_name)
            _log(f"Added family IDs to ID map (family rows now included). Total rows: {len(updated_map)}")

        _log("Writing PLINK .bed/.bim/.fam (BIM preserved exactly)")
        bed_prefix = str(out_dir_p / "synthetic")
        write_plink(bed_prefix, syn_df, bim_df, fam_out=fam_out)
        _log("Wrote PLINK outputs: synthetic.bed/.bim/.fam")

    # -------------------------
    # Write PED (RESTORED)
    # -------------------------
    if "ped" in output_formats:
        ped_path = out_dir_p / "synthetic.ped"
        _log(f"Writing PED: {ped_path}")
        write_ped(syn_df, bim_df, ped_path, fam_out=fam_out)
        _log("Wrote PED: synthetic.ped")

    # -------------------------
    # Write VCF (RESTORED, minimal GT-only)
    # -------------------------
    if "vcf" in output_formats:
        vcf_path = out_dir_p / "synthetic.vcf"
        _log(f"Writing VCF: {vcf_path}")
        write_vcf_simple(syn_df, bim_df, vcf_path)
        _log("Wrote VCF: synthetic.vcf")

    # -------------------------
    # QC
    # -------------------------
    qc_path = out_dir_p / "synthetic_qc_report.csv"
    _run_qc_report(syn_df, qc_path)

    # Optional debug verification of final fam on disk
    fam_path = out_dir_p / "synthetic.fam"
    if fam_path.exists():
        st = fam_path.stat()
        print(f"[DEBUG] END-OF-RUN fam mtime={st.st_mtime} size={st.st_size}", flush=True)
        check2 = pd.read_csv(fam_path, sep=r"\s+", header=None, dtype=str, engine="python")
        print(f"[DEBUG] END-OF-RUN fam first row: {check2.iloc[0].tolist()}", flush=True)

    _log("synth_wgs_minimal complete")
    return syn_df, updated_map
