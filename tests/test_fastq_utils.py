from pathlib import Path
import pandas as pd
from mmai.synth.fastq_utils import rename_fastqs

# Purpose: Test the FASTQ renaming utility function
def test_fastq_renaming(tmp_path):
    fq = tmp_path / "A_read1.fastq"
    fq.write_text("@SEQ\nACGT\n+\n!!!!\n")

    id_map = pd.DataFrame({
        "original_id": ["A"],
        "synthetic_id": ["100000"],
    })

    rename_fastqs([fq], tmp_path, id_map)

    outdir = tmp_path / "fastq"
    assert outdir.exists()
    assert any("100000" in p.name for p in outdir.iterdir())
