"""
Smoke test for the MAI-T1D synthetic pipeline.

This test validates that the orchestration layer executes without error
when provided with minimal example data.
"""

from pathlib import Path
from scripts.synthesize_all import run_pipeline, parse_args


def test_pipeline_wgs_only(tmp_path):
    args = parse_args()

    args.input = "example_data"
    args.output = str(tmp_path)
    args.synth_wgs_minimal = True
    args.process_csvs = False
    args.process_fastq = False
    args.force = True

    run_pipeline(args)

    # Verify expected outputs
    assert (tmp_path / "synthetic.bed").exists()
    assert (tmp_path / "synthetic.bim").exists()
    assert (tmp_path / "synthetic.fam").exists()