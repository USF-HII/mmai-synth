#!/usr/bin/env python3
"""
synthesize_all.py

Master orchestration script for the MAI-T1D synthetic data generation pipeline.

This script coordinates synthetic data generation across multiple modalities
while enforcing deterministic execution order and cross-modal identifier
consistency.

SUPPORTED MODALITIES
--------------------
1. Genome-scale synthetic WGS data (PLINK / VCF / matrix)
2. Tabular CSV data using SDV-based generative models
3. FASTQ file renaming for cross-modal linkage validation

ARCHITECTURAL PRINCIPLES
------------------------
- This file is an orchestration layer, not an algorithmic layer.
- Genome-scale PLINK synthesis is owned exclusively by synth_wgs_minimal.
- CSV synthesis is optional and explicitly gated.
- FASTQ renaming is optional and explicitly gated.
- All execution flows through run_pipeline(args).

Author: Kenneth Young, PhD
        Dena Tewey, MPH
Affiliation: USF Health Informatics Institute
"""

import argparse
from pathlib import Path

from mmai.synth.synth_wgs_minimal import generate_synthetic_wgs
from mmai.synth.id_utils import load_or_build_id_map
from mmai.synth.csv_utils import synthesize_csv_table
from mmai.synth.fastq_utils import rename_fastqs


# ---------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAI-T1D synthetic data pipeline")

    p.add_argument("--input", default="example_data")
    p.add_argument("--output", default="synthetic_data")

    # Default ID column hint (used to detect participant IDs in CSVs))
    p.add_argument("--id-col-hint", default="maskid") 
    # Default synthetic ID column name (used in output CSVs)
    p.add_argument("--id-col-name", default="MAI_T1D_maskid") 
    p.add_argument("--force", action="store_true")

    p.add_argument("--synth-wgs-minimal", action="store_true")
    p.add_argument("--process-csvs", action="store_true")
    p.add_argument("--process-fastq", action="store_true")

    p.add_argument("--model-csv", default="copulagan")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--rows", type=int, default=0)

    return p.parse_args()


# ---------------------------------------------------------------------
# Pipeline execution (authoritative entry point)
# ---------------------------------------------------------------------
def run_pipeline(args: argparse.Namespace) -> None:
    print("[INFO] ============================================================")
    print("[INFO] Starting MAI-T1D synthetic data pipeline")
    print("[INFO] ============================================================")

    indir = Path(args.input).resolve()
    outdir = Path(args.output).resolve()

    print(f"[INFO] Input directory : {indir}")
    print(f"[INFO] Output directory: {outdir}")

    csv_paths = sorted(indir.glob("*.csv"))
    bed_paths = sorted(indir.glob("*.bed"))
    fastq_paths = sorted(indir.glob("*.fastq")) + sorted(indir.glob("*.fastq.gz"))

    print(
        f"[INFO] Discovered files -> "
        f"{len(csv_paths)} CSV, {len(bed_paths)} BED, {len(fastq_paths)} FASTQ"
    )

    # -------------------------------------------------------------
    # ID mapping
    # -------------------------------------------------------------
    print("[INFO] ------------------------------------------------------------")
    print("[INFO] Building or loading synthetic ID map")
    print("[INFO] ------------------------------------------------------------")

    id_map, id_col_name = load_or_build_id_map(
        outdir=outdir,
        id_col_hint=args.id_col_hint,
        id_col_name=args.id_col_name,
        csv_paths=csv_paths,
        force_new=args.force,
    )

    if id_map.empty:
        print("[WARNING] No participant IDs detected; cross-modal linkage disabled")
    else:
        print(
            f"[INFO] Loaded ID map with {len(id_map)} participants "
            f"(synthetic ID range: {id_map['synthetic_id'].iloc[0]}-"
            f"{id_map['synthetic_id'].iloc[-1]})"
        )

    # =============================================================
    # Phase 1: Genome-scale WGS synthesis
    # =============================================================
    print("[INFO] ============================================================")
    print("[INFO] Phase 1: Genome-scale WGS synthesis")
    print("[INFO] ============================================================")

    if args.synth_wgs_minimal:
        print("[INFO] WGS synthesis enabled")

        if len(bed_paths) != 1:
            raise RuntimeError(
                "Exactly one BED file is required for WGS synthesis"
            )

        bed = bed_paths[0]
        print(f"[INFO] Using PLINK prefix: {bed.stem}")

        generate_synthetic_wgs(
            plink_prefix=str(bed.with_suffix("")),
            bim_path=str(bed.with_suffix(".bim")),
            out_dir=str(outdir),
            map_ids=True,
            id_start=100000,
            output_formats=["matrix", "vcf", "plink"],
            max_snps=10000,
        )

        print("[INFO] Phase 1 complete: WGS synthesis finished")
    else:
        print("[INFO] Phase 1 skipped: --synth-wgs-minimal not set")

    # =============================================================
    # Phase 2: CSV synthesis
    # =============================================================
    print("[INFO] ============================================================")
    print("[INFO] Phase 2: CSV synthesis")
    print("[INFO] ============================================================")

    if args.process_csvs:
        if not csv_paths:
            print("[INFO] No CSV files found; skipping CSV synthesis")
        else:
            print(f"[INFO] Starting CSV synthesis for {len(csv_paths)} tables")

            for p in csv_paths:
                print(f"[INFO] START CSV synthesis: {p.name}")

                synthesize_csv_table(
                    path=p,
                    outdir=outdir,
                    id_map=id_map,
                    id_col_hint=args.id_col_hint,
                    id_col_name=args.id_col_name,
                    model=args.model_csv,
                    fallback_model="gaussian",
                    epochs=args.epochs,
                    rows=args.rows,
                )

                print(f"[INFO] DONE  CSV synthesis: {p.name}")

            print("[INFO] Phase 2 complete: CSV synthesis finished")
    else:
        print("[INFO] Phase 2 skipped: --process-csvs not set")

    # =============================================================
    # Phase 3: FASTQ renaming
    # =============================================================
    print("[INFO] ============================================================")
    print("[INFO] Phase 3: FASTQ renaming")
    print("[INFO] ============================================================")

    if args.process_fastq:
        if not fastq_paths:
            print("[INFO] No FASTQ files found; skipping FASTQ renaming")
        elif id_map.empty:
            print("[WARNING] FASTQ renaming requested but no ID map available")
        else:
            print(f"[INFO] Renaming {len(fastq_paths)} FASTQ files")
            rename_fastqs(fastq_paths, outdir, id_map)
            print("[INFO] Phase 3 complete: FASTQ renaming finished")
    else:
        print("[INFO] Phase 3 skipped: --process-fastq not set")

    # =============================================================
    # Pipeline completion
    # =============================================================
    print("[INFO] ============================================================")
    print("[INFO] Synthetic data pipeline completed successfully")
    print("[INFO] ============================================================")


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
