from pathlib import Path
from typing import Literal, Tuple, Optional
import numpy as np
import pandas as pd
from bed_reader import open_bed
from Bio import SeqIO

FileType = Literal["csv", "plink", "fastq"]

def detect_type(path: Path) -> FileType:
    ext = path.suffix.lower()
    if ext == ".csv":
        return "csv"
    if ext == ".bed":
        return "plink"
    if ext in (".fq", ".fastq"):
        return "fastq"
    raise ValueError(f"Unsupported file type: {ext}")

def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def load_plink(bed_path: Path):
    """
    Load PLINK .bed using bed_reader.open_bed.
    Returns:
        geno_df: DataFrame shape (n_samples, n_snps) with values in {0,1,2} or NaN
        meta: dict with 'samples' and 'snps'
    """
    bed = open_bed(str(bed_path))
    X = bed.read()  # numpy array, typically float32 with NaNs for missing
    # bed.iid: array of tuples or strings for sample identifiers
    # bed.sid: array of SNP identifiers
    # Normalize sample ids to simple strings
    def _iid_to_str(iid_item):
        if isinstance(iid_item, (list, tuple)) and len(iid_item) >= 1:
            return str(iid_item[0])
        return str(iid_item)

    samples = np.array([_iid_to_str(i) for i in bed.iid])
    snps = np.array([str(s) for s in bed.sid])

    geno_df = pd.DataFrame(X, index=samples, columns=snps)
    # Optionally round to nearest genotype count
    geno_df = geno_df.round().clip(0, 2)

    meta = {"samples": samples, "snps": snps}
    return geno_df.reset_index(drop=False).rename(columns={"index": "sample_id"}), meta

def iter_fastq(path: Path):
    """Yield (id, seq, qual) tuples."""
    with path.open("r") as handle:
        for record in SeqIO.parse(handle, "fastq"):
            yield record.id, str(record.seq), record.letter_annotations["phred_quality"]

def fastq_to_dataframe(path: Path, max_reads: Optional[int] = None) -> pd.DataFrame:
    rows = []
    for i, (rid, seq, qual) in enumerate(iter_fastq(path)):
        if max_reads is not None and i >= max_reads:
            break
        rows.append({"read_id": rid, "seq": seq, "len": len(seq), "mean_qual": sum(qual)/len(qual)})
    return pd.DataFrame(rows)
