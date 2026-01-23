"""
plink_utils.py

Utilities for reading and writing PLINK genotype data.

This module provides:
- Safe, explicit PLINK (.bed/.bim/.fam) reading
- Fully compliant PLINK writing
- Lightweight QC metrics for synthetic genotypes

Design principles:
- Explicit return values (no hidden globals)
- Stable interfaces
- ASCII only
- Clear separation of concerns

Responsibilities:
- Read PLINK .bed/.bim/.fam into a pandas DataFrame
- Write fully compliant PLINK (.bed/.bim/.fam)
- Write text-based PED for inspection/debugging
- Fast QC metrics (MAF, missingness, genotype counts)

Assumptions:
- Genotypes encoded as 0/1/2 (A1 dosage)

Author: Kenneth Young, PhD (USF HII)
"""

from typing import Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from bed_reader import open_bed, create_bed


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def normalize_genotypes(geno_df: pd.DataFrame) -> np.ndarray:
    """
    Normalize genotype matrix to PLINK-compatible int8 A1 counts.
    Missing is encoded as -1.
    """
    arr = (
        geno_df
        .apply(pd.to_numeric, errors="coerce")
        .round()
        .clip(0, 2)
        .astype("Int8")   # allows NA
        .to_numpy()
    )

    # Convert pandas NA to PLINK missing code (-1)
    arr = np.where(pd.isna(arr), -1, arr).astype(np.int8)

    # Ensure C-contiguous memory
    return np.ascontiguousarray(arr)


def normalize_genotypes_for_output(geno_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize synthetic genotypes to valid PLINK domain:
    - values in {0,1,2}
    - missing preserved as NA
    """
    out = (
        geno_df
        .apply(pd.to_numeric, errors="coerce")
        .clip(lower=0, upper=2)
    )

    # Discretize safely
    out = out.apply(
        lambda col: pd.cut(
            col,
            bins=[-np.inf, 0.5, 1.5, np.inf],
            labels=[0, 1, 2]
        )
    )

    return out.astype("Int8")  # allows NA

# Synthetic genotypes are fully observed by design.
# Missing values from the synthesis model are imputed
# using per-SNP modal genotype.
def normalize_and_impute_genotypes(geno_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize synthetic genotypes:
    - coerce to numeric
    - clip to [0,2]
    - discretize
    - impute missing per SNP using mode
    """

    df = (
        geno_df
        .apply(pd.to_numeric, errors="coerce")
        .clip(lower=0, upper=2)
    )

    # Discretize
    df = df.apply(
        lambda col: pd.cut(
            col,
            bins=[-np.inf, 0.5, 1.5, np.inf],
            labels=[0, 1, 2]
        )
    )

    # Impute missing per SNP
    for c in df.columns:
        if df[c].isna().any():
            mode = df[c].mode(dropna=True)
            fill = int(mode.iloc[0]) if not mode.empty else 0
            df[c] = df[c].fillna(fill)

    return df.astype(np.int8)


# ---------------------------------------------------------------------
# PLINK READER
# ---------------------------------------------------------------------
def read_plink_prefix(prefix: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read a PLINK .bed/.bim/.fam triple using a filename prefix.

    Parameters
    ----------
    prefix : str
        Path prefix without extension (e.g. "/path/GWAS_masked")

    Returns
    -------
    geno_df : pd.DataFrame
        Genotype matrix (samples x SNPs), values in {0,1,2}
    bim_df : pd.DataFrame
        Variant metadata with columns:
        [chrom, snp, cm, pos, a1, a2]
    fam_df : pd.DataFrame
        Sample metadata with standard 6 PLINK columns
    """

    prefix = str(prefix)
    bed_path = prefix + ".bed"
    bim_path = prefix + ".bim"
    fam_path = prefix + ".fam"

    # --- Read FAM ---
    fam_df = pd.read_csv(
        fam_path,
        sep=r"\s+",
        header=None,
        names=["fid", "iid", "pid", "mid", "sex", "pheno"],
        engine="python"
    )

    sample_ids = fam_df["iid"].astype(str).tolist()

    # --- Read BIM ---
    bim_df = pd.read_csv(
        bim_path,
        sep=r"\s+",
        header=None,
        names=["chrom", "snp", "cm", "pos", "a1", "a2"],
        engine="python"
    )

    snp_ids = bim_df["snp"].astype(str).tolist()

    # --- Read BED ---
    with open_bed(bed_path) as bed:
        geno = bed.read()  # shape: (n_samples, n_snps)

    geno_df = pd.DataFrame(
        geno,
        index=sample_ids,
        columns=snp_ids
    )

    # Normalize genotype values
    geno_df = (
        geno_df
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)  
        .round()
        .clip(0, 2)
        .astype(np.int8)
    )

    geno_df.index.name = "sample_id"

    return geno_df, bim_df, fam_df



# ---------------------------------------------------------------------
# PED WRITER
# ---------------------------------------------------------------------
def write_ped(
    geno_df: pd.DataFrame,
    bim_df: pd.DataFrame,
    out_path: Path
) -> None:
    """
    Write a fully PLINK-compatible PED file.

    geno_df:
        samples x SNPs, values in {0,1,2} or NA
    bim_df:
        must align exactly to geno_df.columns
    """

    # --- Safety check ---
    if list(geno_df.columns) != list(bim_df["snp"]):
        raise ValueError("BIM SNP order does not match genotype columns")

    rows = []

    for sample_id, row in geno_df.iterrows():
        ped_row = [
            sample_id,  # FID
            sample_id,  # IID
            "0",        # PID
            "0",        # MID
            "0",        # SEX
            "-9",       # PHENO
        ]

        for g, row_bim in zip(row.values, bim_df.itertuples()):
            a1 = row_bim.a1
            a2 = row_bim.a2

            if pd.isna(g):
                ped_row.extend(["0", "0"])
            elif g == 0:
                ped_row.extend([a2, a2])
            elif g == 1:
                ped_row.extend([a1, a2])
            elif g == 2:
                ped_row.extend([a1, a1])
            else:
                ped_row.extend(["0", "0"])


        rows.append(ped_row)

    ped_df = pd.DataFrame(rows)
    ped_df.to_csv(out_path, sep="\t", index=False, header=False)



# ---------------------------------------------------------------------
# PLINK WRITER
# ---------------------------------------------------------------------

def write_plink(
    prefix: str,
    geno_df: pd.DataFrame,
    bim_df: pd.DataFrame,
    fam_df: pd.DataFrame
) -> None:
    """
    Write a fully compliant PLINK dataset (.bed/.bim/.fam).

    Parameters
    ----------
    prefix : str
        Output prefix (without extension)
    geno_df : pd.DataFrame
        Genotype matrix (samples x SNPs), values in {0,1,2}
    bim_df : pd.DataFrame
        Variant metadata aligned to geno_df columns
    fam_df : pd.DataFrame
        Sample metadata aligned to geno_df rows
    """

    # Enforce exact column order match
    if list(geno_df.columns) != list(bim_df["snp"]):
        raise ValueError("BIM SNP order does not match genotype columns")


    prefix = Path(prefix)

    # --- Write BIM ---
    bim_df[["chrom", "snp", "cm", "pos", "a1", "a2"]].to_csv(
        prefix.with_suffix(".bim"),
        sep="\t",
        index=False,
        header=False,
    )


    fam_out = pd.DataFrame({
        0: geno_df.index,   # FID
        1: geno_df.index,   # IID
        2: 0,               # PID
        3: 0,               # MID
        4: 0,               # SEX
        5: -9,              # PHENO
    })

    fam_out.to_csv(
        prefix.with_suffix(".fam"),
        sep="\t",
        index=False,
        header=False
    )

    # --- Write BED ---
    
    # Normalize and ensure int8
    geno_array = normalize_genotypes(geno_df)
    geno_array = np.ascontiguousarray(geno_array, dtype=np.float32)

    n_samples, n_snps  = geno_array.shape

    with create_bed(prefix.with_suffix(".bed"), n_samples, n_snps) as bed_writer:
        for snp_idx in range(n_snps):
            # bed_reader expects one SNP (all individuals) at a time
            bed_writer.write(geno_array[:, snp_idx])


# ---------------------------------------------------------------------
# QC METRICS (FAST)
# ---------------------------------------------------------------------

def compute_basic_qc(geno_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute fast QC metrics:
    - MAF
    - Missingness
    - Genotype counts
    """
    n = geno_df.shape[0]

    maf = geno_df.sum(axis=0) / (2 * n)
    maf = maf.where(maf <= 0.5, 1 - maf)

    report = pd.DataFrame({
        "snp": geno_df.columns,
        "maf": maf.values,
        "missing_rate": geno_df.isna().mean(axis=0).values,
        "n_hom_ref": (geno_df == 0).sum(axis=0).values,
        "n_het": (geno_df == 1).sum(axis=0).values,
        "n_hom_alt": (geno_df == 2).sum(axis=0).values,
    })

    return report
