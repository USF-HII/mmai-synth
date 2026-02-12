"""
plink_utils.py

PLINK I/O helpers for synthetic genotype pipelines.

Responsibilities
- Read PLINK prefix (.bed/.bim/.fam) into DataFrames
- Write PLINK (.bed/.bim/.fam) using a provided genotype matrix and BIM/FAM tables
- Write PED for inspection/debugging

Notes
- Genotypes are expected in {0,1,2} (dosage) before writing.

Author: Kenneth Young, PhD (USF-HII)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from bed_reader import open_bed, create_bed


def read_plink_prefix(prefix: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prefix = str(prefix)
    bed_path = prefix + ".bed"
    bim_path = prefix + ".bim"
    fam_path = prefix + ".fam"

    fam_df = pd.read_csv(
        fam_path,
        sep=r"\s+",
        header=None,
        names=["fid", "iid", "pid", "mid", "sex", "pheno"],
        engine="python",
        dtype=str,
    )

    bim_df = pd.read_csv(
        bim_path,
        sep=r"\s+",
        header=None,
        names=["chrom", "snp", "cm", "pos", "a1", "a2"],
        engine="python",
        dtype=str,
    )

    sample_ids = fam_df["iid"].astype(str).tolist()
    snp_ids = bim_df["snp"].astype(str).tolist()

    with open_bed(bed_path) as bed:
        geno = bed.read()  # shape (n_samples, n_snps)

    geno_df = pd.DataFrame(geno, index=sample_ids, columns=snp_ids)

    # Coerce to numeric dosage domain
    geno_df = (
        geno_df.apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .round()
        .clip(0, 2)
        .astype("Int8")
    )

    geno_df.index.name = "iid"
    return geno_df, bim_df, fam_df


def normalize_and_impute_genotypes(geno_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure {0,1,2} int domain and impute missing per SNP with modal genotype.
    """
    df = geno_df.apply(pd.to_numeric, errors="coerce").clip(0, 2)

    # Discretize to 0/1/2
    df = df.apply(
        lambda col: pd.cut(
            col,
            bins=[-np.inf, 0.5, 1.5, np.inf],
            labels=[0, 1, 2],
        )
    ).astype("Int8")

    # Impute missing per SNP with mode
    for c in df.columns:
        if df[c].isna().any():
            mode = df[c].mode(dropna=True)
            fill = int(mode.iloc[0]) if not mode.empty else 0
            df[c] = df[c].fillna(fill)

    return df.astype(np.int8)


def write_plink(
    prefix: str,
    geno_df: pd.DataFrame,
    bim_df: pd.DataFrame,
    fam_out: Optional[pd.DataFrame] = None,
) -> None:
    """
    Write PLINK .bed/.bim/.fam.

    Critical detail:
    bed_reader.create_bed() may auto-generate/overwrite .bim/.fam with dummy values
    when the BED writer closes. Therefore:
      1) write BED first
      2) then overwrite BIM and FAM last (authoritative)

    Preconditions
    - geno_df columns match bim_df['snp'] exactly (order and identity)
    - fam_out, if provided, has columns: fid,iid,pid,mid,sex,pheno
    """
    prefix = Path(prefix)

    # Enforce BIM alignment
    bim_snps = bim_df["snp"].astype(str).tolist()
    geno_snps = [str(c) for c in geno_df.columns.tolist()]
    if geno_snps != bim_snps:
        raise ValueError("BIM SNP order does not match genotype columns.")

    # Build FAM if not provided
    if fam_out is None:
        fam_out = pd.DataFrame(
            {
                "fid": geno_df.index.astype(str),
                "iid": geno_df.index.astype(str),
                "pid": "0",
                "mid": "0",
                "sex": "0",
                "pheno": "-9",
            }
        )
    else:
        fam_out = fam_out.copy()
        fam_out.columns = ["fid", "iid", "pid", "mid", "sex", "pheno"]
        for c in ["fid", "iid", "pid", "mid", "sex", "pheno"]:
            fam_out[c] = fam_out[c].astype(str)

    # ------------------------------------------------------------------
    # 1) Write BED FIRST (may generate dummy .bim/.fam internally)
    # ------------------------------------------------------------------
    bed_path = prefix.with_suffix(".bed")

    geno_array = geno_df.to_numpy(dtype=np.int8)
    geno_array = np.ascontiguousarray(geno_array, dtype=np.float32)
    n_samples, n_snps = geno_array.shape

    with create_bed(bed_path, n_samples, n_snps) as bed_writer:
        for snp_idx in range(n_snps):
            bed_writer.write(geno_array[:, snp_idx])

    # ------------------------------------------------------------------
    # 2) Overwrite BIM LAST (authoritative)
    # ------------------------------------------------------------------
    bim_path = prefix.with_suffix(".bim")
    bim_df[["chrom", "snp", "cm", "pos", "a1", "a2"]].to_csv(
        bim_path,
        sep="\t",
        index=False,
        header=False,
    )

    # ------------------------------------------------------------------
    # 3) Overwrite FAM LAST (authoritative)
    # ------------------------------------------------------------------
    fam_path = prefix.with_suffix(".fam")
    fam_out[["fid", "iid", "pid", "mid", "sex", "pheno"]].to_csv(
        fam_path,
        sep="\t",
        index=False,
        header=False,
    )


def write_ped(
    geno_df: pd.DataFrame,
    bim_df: pd.DataFrame,
    out_path: Path,
    fam_out: Optional[pd.DataFrame] = None,
) -> None:
    """
    Write PED for inspection/debugging.

    - If fam_out is provided, use its fid/iid/pid/mid/sex/pheno columns
      for the first 6 PED fields.
    - Otherwise, fall back to legacy behavior (FID=IID, sex=0, etc.)
    """
    out_path = Path(out_path)

    # Validate SNP order
    bim_snps = bim_df["snp"].astype(str).tolist()
    geno_snps = [str(c) for c in geno_df.columns.tolist()]
    if geno_snps != bim_snps:
        raise ValueError("BIM SNP order does not match genotype columns.")

    # Prepare genotype matrix as floats so NaN works
    gmat = geno_df.apply(pd.to_numeric, errors="coerce").to_numpy()

    # Header fields
    if fam_out is None:
        fam_out2 = pd.DataFrame(
            {
                "fid": geno_df.index.astype(str),
                "iid": geno_df.index.astype(str),
                "pid": "0",
                "mid": "0",
                "sex": "0",
                "pheno": "-9",
            }
        )
    else:
        fam_out2 = fam_out.copy()
        fam_out2 = fam_out2[["fid", "iid", "pid", "mid", "sex", "pheno"]]
        for c in ["fid", "iid", "pid", "mid", "sex", "pheno"]:
            fam_out2[c] = fam_out2[c].astype(str)

        # Ensure ordering matches geno_df rows by IID (geno_df index is IID)
        fam_out2 = fam_out2.set_index("iid", drop=False).reindex(geno_df.index.astype(str))
        if fam_out2["fid"].isna().any():
            missing = fam_out2.index[fam_out2["fid"].isna()].tolist()[:10]
            raise RuntimeError(f"PED write: fam_out missing rows for IIDs (examples): {missing}")

    # Normalize allele strings (PED missing allele is "0")
    def _allele(x: str) -> str:
        s = str(x)
        if s.lower() in ("", "0", "nan", "none", "."):
            return "0"
        return s

    a1 = [_allele(x) for x in bim_df["a1"].tolist()]
    a2 = [_allele(x) for x in bim_df["a2"].tolist()]

    # Stream write
    with open(out_path, "w", newline="\n") as f:
        for row_idx in range(gmat.shape[0]):
            h = fam_out2.iloc[row_idx]
            ped_fields = [h["fid"], h["iid"], h["pid"], h["mid"], h["sex"], h["pheno"]]

            g_row = gmat[row_idx, :]
            for j, g in enumerate(g_row):
                if np.isnan(g):
                    ped_fields.extend(["0", "0"])
                else:
                    gi = int(round(float(g)))
                    if gi == 0:
                        ped_fields.extend([a2[j], a2[j]])
                    elif gi == 1:
                        ped_fields.extend([a1[j], a2[j]])
                    elif gi == 2:
                        ped_fields.extend([a1[j], a1[j]])
                    else:
                        ped_fields.extend(["0", "0"])

            f.write("\t".join(ped_fields) + "\n")

