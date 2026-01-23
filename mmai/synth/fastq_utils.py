"""
fastq_utils.py

Utilities for renaming FASTQ files to synthetic participant identifiers.

This module performs filename-level transformations only and does not
inspect sequence contents.

FASTQ files are not synthesized at the read level. Instead, original FASTQ files 
are retained and filenames are deterministically rewritten to use synthetic participant 
identifiers. This approach preserves workflow realism and cross-modal linkage while 
avoiding the substantial technical and privacy risks associated with raw sequencing data 
synthesis. Synthetic genotype and phenotype data provide the primary analytic substrate, 
while FASTQ renaming supports integration testing and pipeline validation.

Author: Kenneth Young, PhD
"""

from pathlib import Path
from typing import List, Tuple
import shutil
import pandas as pd


def rename_fastqs(
    fastq_paths: List[Path],
    outdir: Path,
    id_map: pd.DataFrame,
    overwrite: bool = False,
) -> List[Tuple[Path, Path, str]]:
    """
    Create renamed copies of FASTQ files using synthetic participant IDs.

    This function performs filename-level anonymization only.
    FASTQ file contents are NOT modified.

    Parameters
    ----------
    fastq_paths : list of Path
        Input FASTQ or FASTQ.GZ files.
    outdir : Path
        Output directory where renamed FASTQs will be written.
    id_map : pd.DataFrame
        DataFrame with columns:
            - original_id
            - synthetic_id
    overwrite : bool
        If True, overwrite existing files. Default False.

    Returns
    -------
    List of tuples:
        (source_path, destination_path, status)

    Status values:
        - "renamed"
        - "skipped_exists"
        - "no_id_match"
    """

    results = []

    if id_map is None or id_map.empty:
        print("[INFO] No ID map provided. FASTQ renaming skipped.")
        return results

    # Ensure output directory exists
    fastq_outdir = outdir / "fastq"
    fastq_outdir.mkdir(parents=True, exist_ok=True)

    # Build string-safe ID mapping
    id_pairs = [
        (str(orig), str(syn))
        for orig, syn in zip(
            id_map["original_id"],
            id_map["synthetic_id"]
        )
    ]

    print(f"[INFO] Renaming {len(fastq_paths)} FASTQ files")

    for src in fastq_paths:
        src = Path(src)
        name = src.name

        matched = False
        dst = None

        # Attempt to match original ID in filename
        for orig_id, syn_id in id_pairs:
            if orig_id in name:
                new_name = name.replace(orig_id, syn_id)
                dst = fastq_outdir / new_name
                matched = True
                break

        if not matched:
            results.append((src, src, "no_id_match"))
            continue

        if dst.exists() and not overwrite:
            results.append((src, dst, "skipped_exists"))
            continue

        shutil.copy2(src, dst)
        results.append((src, dst, "renamed"))

    print(f"[INFO] FASTQ renaming complete ({len(results)} files processed)")
    return results
