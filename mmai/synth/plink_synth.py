"""
plink_synth.py

Synthetic PLINK genotype generator with cross-modality ID consistency.

Author: Kenneth Young, PhD (USF HII)
"""

from pathlib import Path
import pandas as pd
import numpy as np

from mmai.synth.plink_utils import (
    read_plink_prefix,
    write_plink,
    write_ped,
)
from mmai.synth.tabular import synthesize_gaussiancopula
from mmai.synth.vcf_writer import write_vcf_from_df


def synthesize_plink(
    plink_prefix: str,
    out_dir: Path,
    id_map: dict[str, str],
    *,
    max_snps: int | None = None,
    seed: int = 42,
    output_formats: list[str] = ["plink", "ped", "vcf", "matrix"],
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------
    # Load PLINK
    # --------------------
    geno_df, bim_df, fam_df = read_plink_prefix(plink_prefix)

    # --------------------
    # Enforce ID coverage
    # --------------------
    orig_ids = geno_df.index.astype(str).tolist()
    missing = [i for i in orig_ids if i not in id_map]
    if missing:
        raise ValueError(
            f"{len(missing)} PLINK samples missing from synthetic ID map"
        )

    # --------------------
    # Optional SNP truncation
    # --------------------
    if max_snps and geno_df.shape[1] > max_snps:
        geno_df = geno_df.iloc[:, :max_snps]
        bim_df = bim_df.iloc[:max_snps].copy()

    # --------------------
    # Synthesize genotypes
    # --------------------
    syn_df = synthesize_gaussiancopula(geno_df, seed=seed)

    # --------------------
    # Apply synthetic IDs
    # --------------------
    syn_df.index = syn_df.index.map(lambda x: id_map[str(x)])

    # --------------------
    # Outputs
    # --------------------
    if "matrix" in output_formats:
        syn_df.to_csv(out_dir / "synthetic_genotype_matrix.csv")

    if "plink" in output_formats:
        write_plink(out_dir / "synthetic", syn_df, bim_df, fam_df=None)

    if "ped" in output_formats:
        write_ped(syn_df, bim_df, out_dir / "synthetic.ped")

    if "vcf" in output_formats:
        write_vcf_from_df(
            syn_df,
            bim_df,
            str(out_dir / "synthetic.vcf"),
            id_map=None  # already applied
        )
