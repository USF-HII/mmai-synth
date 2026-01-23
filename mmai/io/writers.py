from pathlib import Path
import pandas as pd

def write_csv(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

def write_fastq(reads, out_path: Path):
    """reads: iterable of (id, seq, qual_list[int])"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rid, seq, qual in reads:
            f.write(f"@{rid}\n{seq}\n+\n{''.join(chr(q+33) for q in qual)}\n")
