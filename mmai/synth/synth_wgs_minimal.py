"""
Generate synthetic WGS genotype data using masked PLINK input.
Outputs synthetic genotypes in tabular (CSV), VCF, PED, and PLINK formats,
with a basic QC report.

Author: Kenneth Young, Ph.D. (USF-HII)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
from mmai.synth.plink_utils import read_plink_prefix, write_plink, write_ped, normalize_and_impute_genotypes
from mmai.synth.tabular import synthesize_gaussiancopula
from mmai.synth.vcf_writer import write_vcf_from_df


def mask_ids(original_ids: List[str], start_mask_id: int = 100000, end_mask_id: int = 199999) -> dict:
    """Map original sample IDs to synthetic IDs in a specified integer range."""
    if len(original_ids) > (end_mask_id - start_mask_id + 1):
        raise ValueError("Too many synthetic IDs required for the specified range.")
    return {pid: str(start_mask_id + i) for i, pid in enumerate(original_ids)}


def run_qc_report(geno_df: pd.DataFrame, out_path: str):
    """Generate a quick QC report: MAF, missingness, and genotype counts."""
    print("Generating QC report...")
    report = []

    for snp in geno_df.columns:
        g = geno_df[snp].dropna()
        n = len(g)
        if n == 0:
            continue
        counts = g.value_counts().to_dict()
        maf = min(counts.get(1, 0) + 2 * counts.get(2, 0), counts.get(1, 0) + 2 * counts.get(0, 0)) / (2 * n)
        missing = 1.0 - n / geno_df.shape[0]
        report.append({
            "SNP": snp,
            "MAF": round(maf, 4),
            "Missingness": round(missing, 4),
            "N_0/0": counts.get(0, 0),
            "N_0/1": counts.get(1, 0),
            "N_1/1": counts.get(2, 0),
            "N_missing": geno_df.shape[0] - n
        })

    qc_df = pd.DataFrame(report)
    qc_df.to_csv(out_path, index=False)
    print(f"[INFO] Saved QC report: {out_path}")


def generate_synthetic_wgs(
    plink_prefix: str,
    bim_path: str,
    out_dir: str,
    map_ids: bool = True,
    id_start: int = 100000,
    id_end: int = 199999,
    output_formats: List[str] = ["matrix", "vcf", "plink"],
    max_snps: Optional[int] = 1000,
):
    """
    Main entry point for generating synthetic WGS genotype data.

    Args:
        plink_prefix (str): Path prefix to .bed/.bim/.fam files.
        bim_path (str): Path to .bim file (for variant metadata).
        out_dir (str): Output directory.
        map_ids (bool): Whether to anonymize participant IDs.
        id_start (int): Starting synthetic ID integer.
        id_end (int): Ending synthetic ID integer.
        output_formats (list): Any subset of ["matrix", "vcf", "plink"].
        max_snps (int): Maximum SNPs to load (for memory control).
    """

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Reading PLINK data from: {plink_prefix}")
    geno_df, bim_df, fam_df = read_plink_prefix(plink_prefix)
    print(f"[INFO] Original genotype shape: {geno_df.shape}")

    # Truncate SNPs for memory
    if max_snps and geno_df.shape[1] > max_snps:
        geno_df = geno_df.iloc[:, :max_snps]
        bim_df = bim_df.iloc[:max_snps].copy()
        print(f"[INFO] SNPs truncated to max_snps={max_snps}")

    print("[INFO] Generating synthetic genotype matrix...")
    syn_df = synthesize_gaussiancopula(geno_df, seed=42)
    print("[INFO] Normalizing and imputing synthetic genotypes...")
    syn_df = normalize_and_impute_genotypes(syn_df)
    print(f"[INFO] Synthetic genotype shape: {syn_df.shape}")

    # Optional ID masking
    if map_ids:
        id_map = mask_ids(syn_df.index.tolist(), start_mask_id=id_start, end_mask_id=id_end)
        syn_df.index = [id_map.get(pid, pid) for pid in syn_df.index]
    else:
        id_map = {}

    # Save matrix
    if "matrix" in output_formats:
        print("[INFO] Writing Matrix...")
        matrix_path = Path(out_dir) / "synthetic_genotype_matrix.csv"
        syn_df.to_csv(matrix_path, index=True)
        print(f"[INFO] Saved Matrix: {matrix_path.name}")

    # Save PLINK and PED files
    if "plink" in output_formats:
        
        # Write  PLINK format files
        print("[INFO] Writing PLINK .bed/.bim/.fam files...")
        bed_prefix = str(Path(out_dir) / "synthetic")
        write_plink(bed_prefix, syn_df, bim_df, fam_df)
        print("[INFO] Saved PLINK files: synthetic.bed/.bim/.fam")
        # Write synthetic PED file
        print(f"[INFO] Writing PED file...")
        ped_path = Path(out_dir) / "synthetic.ped"
        write_ped(syn_df, bim_df, ped_path)
        print(f"[INFO] Saved PED: {ped_path.name}")
    
    

    # Write synthetic VCF
    if "vcf" in output_formats:
        bim_df = pd.read_csv(
            bim_path,
            sep=r"\s+",
            header=None,
            names=["chrom", "snp", "cm", "pos", "a1", "a2"],
            engine="python"
        )

        # Align BIM to synthetic SNPs (order + identity)
        print("[INFO] Aligning BIM to synthetic SNPs.")

        #bim_df = bim_df.set_index("snp").loc[syn_df.columns].reset_index()
        # Build an explicit ordering map
        snp_order = pd.Index(syn_df.columns)

        bim_df = bim_df.loc[bim_df["snp"].isin(snp_order)].copy()
        bim_df["__order"] = bim_df["snp"].map({snp: i for i, snp in enumerate(snp_order)})
        bim_df = bim_df.sort_values("__order").drop(columns="__order").reset_index(drop=True)


        if bim_df.shape[0] != syn_df.shape[1]:
            raise ValueError("BIM and genotype SNP order mismatch")


        vcf_path = Path(out_dir) / "synthetic.vcf"
        write_vcf_from_df(syn_df, bim_df, str(vcf_path), id_map=id_map)
        print(f"[INFO] Saved VCF files: {vcf_path.name}")

    # Generate QC report
    qc_path = Path(out_dir) / "synthetic_qc_report.csv"
    run_qc_report(syn_df, str(qc_path))
    print(f"[INFO] QC Report Generated.")

    print("[INFO] synth_wgs_minimal pipeline complete.")
