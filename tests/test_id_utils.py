import pandas as pd
from pathlib import Path
from mmai.synth.id_utils import load_or_build_id_map

# Purpose: Validate ID map construction.
def test_id_map_creation(tmp_path):
    csv = tmp_path / "test.csv"
    csv.write_text("MASK_ID,value\nA,1\nB,2\n")

    id_map, id_col = load_or_build_id_map(
        outdir=tmp_path,
        id_col_hint="MASK_ID",
        csv_paths=[csv],
        force_new=True,
    )

    assert not id_map.empty
    assert id_col == "MASK_ID"
    assert len(id_map) == 2
    assert "synthetic_id" in id_map.columns