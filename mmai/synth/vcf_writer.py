# mmai/synth/vcf_writer.py

import pandas as pd
from typing import Optional, Dict


def write_vcf_from_df(
    geno_df: pd.DataFrame,
    bim_df: pd.DataFrame,
    path: str,
    id_map: Optional[Dict[str, str]] = None,
):
    """
    Write a minimal but valid VCF from a genotype matrix.

    geno_df:
        samples x SNPs, values in {0,1,2}
    bim_df:
        must be in SAME ORDER as geno_df.columns
    """
    
    # Ensure 'snp' column exists for alignment
    if "snp" not in bim_df.columns and bim_df.index.name == "snp":
        bim_df = bim_df.reset_index()

    # Enforce exact column order match
    if not list(bim_df["snp"]) == list(geno_df.columns):
        raise ValueError("BIM and genotype SNP order mismatch")

    samples = [
        id_map.get(s, s) if id_map else s
        for s in geno_df.index
    ]

    with open(path, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write("##source=mmai-synth\n")
        f.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")

        header = [
            "#CHROM", "POS", "ID", "REF", "ALT",
            "QUAL", "FILTER", "INFO", "FORMAT"
        ]
        f.write("\t".join(header + samples) + "\n")

        for i, snp in enumerate(geno_df.columns):
            row = bim_df.iloc[i]

            calls = []
            for g in geno_df.iloc[:, i]:
                if g == 0:
                    calls.append("0/0")
                elif g == 1:
                    calls.append("0/1")
                elif g == 2:
                    calls.append("1/1")
                else:
                    calls.append("./.")

            f.write("\t".join([
                str(row.chrom),
                str(int(row.pos)),
                row.snp,
                row.a1,
                row.a2,
                ".",
                ".",
                ".",
                "GT",
                *calls
            ]) + "\n")
