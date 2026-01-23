import pandas as pd
from pathlib import Path
from mmai.synth.csv_utils import synthesize_csv_table

# Purpose: Test CSV synthesis functionality
def test_csv_synthesis(tmp_path):
    csv = tmp_path / "input.csv"
    csv.write_text("MASK_ID,x,y\nA,1,2\nB,3,4\n")

    id_map = pd.DataFrame({
        "original_id": ["A", "B"],
        "synthetic_id": ["100000", "100001"],
    })

    out_path, n = synthesize_csv_table(
        path=csv,
        outdir=tmp_path,
        id_map=id_map,
        id_col_hint="MASK_ID",
        model="gaussian",
        fallback_model="gaussian",
        epochs=1,
        rows=2,
    )

    assert out_path.exists()
    assert n == 2
