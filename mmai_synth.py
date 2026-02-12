"""
mmai_synth.py

Canonical programmatic entry point for the MAI-T1D synthetic data pipeline.

This module exists to decouple orchestration logic from command-line parsing.
All execution flows through synthesize_all.run_pipeline() using an explicitly
constructed argparse.Namespace.

This ensures deterministic behavior across:
- IDE execution
- Notebooks
- Batch workflows
- CI pipelines

Author: Kenneth Young, PhD
        Dena Tewey, MPH

Affiliation: USF Health Informatics Institute
"""

from pathlib import Path
from scripts.synthesize_all import parse_args, run_pipeline


def run_synthetic_pipeline(
    input_dir: str = "example_data",
    output_dir: str = "synthetic_data",
    synth_wgs_minimal: bool = True,
    process_csvs: bool = False,
    process_fastq: bool = False,
):
    args = parse_args()

    # INPUT detection for example CSVs
    args.id_col_hint = "maskid"

    # OUTPUT canonical ID column name
    args.id_col_name = "MAI_T1D_maskid"

    args.input = str(Path(input_dir).resolve())
    args.output = str(Path(output_dir).resolve())

    args.synth_wgs_minimal = synth_wgs_minimal
    args.process_csvs = process_csvs
    args.process_fastq = process_fastq

    run_pipeline(args)


if __name__ == "__main__":
    run_synthetic_pipeline()