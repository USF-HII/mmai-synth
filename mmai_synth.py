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
    process_csvs: bool = True,
    process_fastq: bool = True,
):
    """
    Run the MAI-T1D synthetic data pipeline programmatically.

    Parameters
    ----------
    input_dir : str
        Directory containing source CSV, PLINK, and FASTQ data.
    output_dir : str
        Directory where synthetic outputs will be written.
    synth_wgs_minimal : bool
        Whether to run genome-scale WGS synthesis.
    process_csvs : bool
        Whether to run CSV table synthesis.
    process_fastq : bool
        Whether to rename FASTQ files using the synthetic ID map.
    """

    args = parse_args()

    # Override CLI defaults
    
    args.id_col_hint = "maskid"
    args.id_col_name = "MAI_T1D_maskid"
    args.input = str(Path(input_dir).resolve())
    args.output = str(Path(output_dir).resolve())
    args.synth_wgs_minimal = synth_wgs_minimal
    args.process_csvs = process_csvs
    args.process_fastq = process_fastq
    

    # Sensible defaults for WGS-focused runs
    if synth_wgs_minimal:
        args.no_plink = True

    run_pipeline(args)


if __name__ == "__main__":
    run_synthetic_pipeline()
