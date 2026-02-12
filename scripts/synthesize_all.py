#!/usr/bin/env python3
"""
synthesize_all.py

Master orchestration script for the MAI-T1D synthetic data generation pipeline.

Author: Kenneth Young, PhD
        Dena Tewey, MPH
"""

import argparse
from pathlib import Path

from mmai.synth.synth_wgs_minimal import generate_synthetic_wgs
from mmai.synth.id_utils import load_or_build_id_map, write_id_map
from mmai.synth.csv_utils import synthesize_csv_table
from mmai.synth.fastq_utils import rename_fastqs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAI-T1D synthetic data pipeline")

    p.add_argument("--input", default="example_data")
    p.add_argument("--output", default="synthetic_data")

    # INPUT detection (example CSVs)
    p.add_argument("--id-col-hint", default="maskid")

    # OUTPUT canonical column name + what we store in synthetic_id_map.csv
    p.add_argument("--id-col-name", default="MAI_T1D_maskid")

    p.add_argument("--force", action="store_true")

    p.add_argument("--synth-wgs-minimal", action="store_true")
    p.add_argument("--process-csvs", action="store_true")
    p.add_argument("--process-fastq", action="store_true")

    p.add_argument("--model-csv", default="copulagan")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--rows", type=int, default=0)

    return p.parse_args()


def run_pipeline(args: argparse.Namespace) -> None:
    print("[INFO] ============================================================")
    print("[INFO] Starting MAI-T1D synthetic data pipeline")
    print("[INFO] ============================================================")

    indir = Path(args.input).resolve()
    outdir = Path(args.output).resolve()

    ##################################################
    # Create run folder for output with date time
    from datetime import datetime
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir_run = outdir / f"run_{run_tag}"
    outdir_run.mkdir(parents=True, exist_ok=True)
    ##################################################

    print(f"[INFO] Input directory : {indir}")
    print(f"[INFO] Output directory: {outdir}")
    print(f"[INFO] Output run directory: {outdir_run}")
    print(f"[INFO] ID column hint  : {args.id_col_hint}")
    print(f"[INFO] ID column name  : {args.id_col_name}")

    csv_paths = sorted(indir.glob("*.csv"))
    bed_paths = sorted(indir.glob("*.bed"))
    fastq_paths = sorted(indir.glob("*.fastq")) + sorted(indir.glob("*.fastq.gz"))

    print(
        f"[INFO] Discovered files -> "
        f"{len(csv_paths)} CSV, {len(bed_paths)} BED, {len(fastq_paths)} FASTQ"
    )
    if bed_paths:
        for b in bed_paths:
            print(f"[INFO] Found BED: {b.name}")

    # -------------------------------------------------------------
    # ID mapping
    # -------------------------------------------------------------
    print("[INFO] ------------------------------------------------------------")
    print("[INFO] Building or loading synthetic ID map")
    print("[INFO] ------------------------------------------------------------")

    id_map, id_col_name_loaded = load_or_build_id_map(
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
            f"[INFO] Loaded ID map with {len(id_map)} rows "
            f"(entity types: {sorted(id_map['entity_type'].unique().tolist()) if 'entity_type' in id_map.columns else ['participant']})"
        )
        print(f"[INFO] synthetic_id_map id_col_name: {args.id_col_name}")

    # =============================================================
    # Phase 1: WGS synthesis
    # =============================================================
    print("[INFO] ============================================================")
    print("[INFO] Phase 1: Genome-scale WGS synthesis")
    print("[INFO] ============================================================")

    if args.synth_wgs_minimal:
        print("[INFO] WGS synthesis enabled")

        if len(bed_paths) != 1:
            raise RuntimeError(
                f"Exactly one BED file is required for WGS synthesis, found {len(bed_paths)}"
            )

        bed = bed_paths[0]
        plink_prefix = str(bed.with_suffix(""))
        print(f"[INFO] Using PLINK prefix: {plink_prefix}")

        # Try to pick a demographics file:
        # Prefer the synthetic one if it exists, else use input demographics.csv
        # Note: You may need to run CSV only to generate the demographics file 
        # and place in synthetic data folder.
        demo1 = outdir / "Demographics.synthetic.csv"
        demo2 = indir / "demographics.csv"
        demographics_path = str(demo1) if demo1.exists() else (str(demo2) if demo2.exists() else None)
        print(f"[INFO] Demographics path: {demographics_path}")

        # Generate Synthetic WGS files.
        syn_df, updated_map = generate_synthetic_wgs(
            plink_prefix=plink_prefix,
            out_dir=str(outdir_run),
            global_id_map_df=id_map,
            id_col_hint=args.id_col_hint,
            id_col_name=args.id_col_name,
            map_ids=True,
            id_start=100000,
            id_end=199999,
            output_formats=["matrix", "plink", "ped", "vcf"],
            max_snps=10000,
            seed=42,
            demographics_path=demographics_path,
        )

        if updated_map is not None and not updated_map.empty:
            write_id_map(outdir, updated_map, id_col_hint=args.id_col_hint, id_col_name=args.id_col_name) # canonical persistent
            write_id_map(outdir_run, updated_map, id_col_hint=args.id_col_hint, id_col_name=args.id_col_name) # per-run snapshot
            
            id_map = updated_map
            print("[INFO] Updated synthetic_id_map.csv written")

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
                    outdir=outdir_run,
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

    print("[INFO] ============================================================")
    print("[INFO] Synthetic data pipeline completed successfully")
    print("[INFO] ============================================================")


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
