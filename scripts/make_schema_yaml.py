#!/usr/bin/env python3
"""
synthesize_all.py

End-to-end synthetic data generator for a mixed folder of:
- CSV tables (single-table synthesis per file)
- PLINK genetics (BED/BIM/FAM)
- FASTQ files (filenames rewritten from original to synthetic IDs)

Design goals
------------
1) Deterministic, cross-file ID mapping:
   One original subject ID (e.g., MASK_ID) maps to one synthetic ID across all outputs.

2) Safe table synthesis:
   - Do not model the ID column.
   - Model the remaining columns.
   - Re-attach a synthetic ID column whose per-row distribution mirrors the original table.

3) Skippable work:
   - If an expected output already exists, skip unless --force is passed.

4) Practical PLINK handling:
   - Read BED via bed_reader, BIM/FAM via pandas.
   - Optionally cap SNP count (--plink-max-snps) for speed.
   - Produce a synthetic genotype matrix via per-SNP shuffling (privacy-safe and fast).
   - Write a CSV with synthetic genotypes and a linked sample ID column.

5) FASTQ renaming:
   - If FASTQ filenames embed the original ID, produce copies with synthetic ID in the name
     (no sequence editing).

Outputs
-------
- synthetic_data/<TableName>.synthetic.csv for CSV tables
- synthetic_data/<BedBase>.synthetic_genotypes.csv for PLINK
- synthetic_data/fastq/<renamed fastq files>
- synthetic_data/synthetic_id_map.csv (original_id -> synthetic_id)

Dependencies
------------
- pandas
- numpy
- sdv (supports both new `Metadata` and legacy `SingleTableMetadata`)
- bed_reader (for PLINK)
- python 3.9+

Examples
--------
python synthesize_all.py ^
  --input example_data ^
  --output synthetic_data ^
  --anchor-csv Demographics.csv ^
  --id-col MASK_ID ^
  --rows 0 ^
  --skip-existing ^
  --plink-max-snps 1500
"""

from __future__ import annotations

import sys
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --- SDV synthesizers (single-table) ---
from sdv.single_table import (
    CTGANSynthesizer,
    CopulaGANSynthesizer,
    GaussianCopulaSynthesizer,
)

# New API + legacy fallback for metadata
from sdv.metadata import Metadata, SingleTableMetadata

# PLINK BED reader (optional)
try:
    from bed_reader import open_bed  # pip install bed-reader
    BED_READER_AVAILABLE = True
except Exception:
    BED_READER_AVAILABLE = False


# ----------------------------
# Logging (simple, stdout only)
# ----------------------------

def log_info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)

def log_warn(msg: str) -> None:
    print(f"[WARNING] {msg}", flush=True)

def log_error(msg: str) -> None:
    print(f"[ERROR] {msg}", flush=True)


# ----------------------------
# File discovery
# ----------------------------

def discover_files(indir: Path) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Return lists of CSVs, BEDs, and FASTQs in indir (non-recursive).
    """
    csvs = sorted([p for p in indir.glob("*.csv") if p.is_file()])
    beds = sorted([p for p in indir.glob("*.bed") if p.is_file()])
    fastqs = sorted([p for p in indir.glob("*.fastq")] + [p for p in indir.glob("*.fastq.gz")])
    return csvs, beds, fastqs


# ----------------------------
# CSV safe reading (Unicode)
# ----------------------------

def safe_read_csv(path: Path) -> pd.DataFrame:
    """
    Robust CSV reader:
      1) Try UTF-8-SIG (handles BOM). 
      2) Fallback to latin-1 if decoding fails.
    """
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


# ----------------------------
# ID mapping helpers
# ----------------------------

ID_CANDIDATES = [
    "MASK_ID", "mask_id", "Mask_ID", "MASKID", "maskid",
    "participant_id", "participant", "subject_id", "subject",
    "patient_id", "record_id", "EXAMP_maskid"
]

def detect_id_column(df: pd.DataFrame, requested: Optional[str]) -> Optional[str]:
    """
    Pick an ID column to use.
    Priority:
      1) requested argument (if present in df)
      2) common name heuristics (case-insensitive)
      3) None if not found
    """
    if requested and requested in df.columns:
        return requested
    lowered = {c.lower(): c for c in df.columns}
    for cand in ID_CANDIDATES:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    # "contains maskid" heuristic
    for c in df.columns:
        if "maskid" in c.lower():
            return c
    return None


def infer_id_map_from_csvs(
    csv_paths: List[Path],
    id_col_hint: Optional[str],
    id_prefix: str
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Scan all CSVs to find an ID column and build a global original->synthetic map.
    If multiple tables have an ID column, we union all unique IDs.
    Returns (id_map, resolved_id_col_name or None).
    """
    unique_ids: set[str] = set()
    resolved_name: Optional[str] = None

    for p in csv_paths:
        try:
            df = safe_read_csv(p)
        except Exception:
            continue
        id_col = detect_id_column(df, id_col_hint)
        if not id_col or id_col not in df.columns:
            continue
        # Remember a reasonable column name to write in metadata
        resolved_name = resolved_name or id_col
        ids = df[id_col].astype(str).dropna().unique().tolist()
        unique_ids.update(ids)

    if not unique_ids:
        return pd.DataFrame(columns=["original_id", "synthetic_id"]), None

    uniques = sorted(list(unique_ids))
    width = max(6, len(str(len(uniques))))
    synthetic = [f"{id_prefix}{i:0{width}d}" for i in range(1, len(uniques) + 1)]
    id_map = pd.DataFrame({"original_id": uniques, "synthetic_id": synthetic})
    return id_map, resolved_name


def load_or_build_id_map(
    outdir: Path,
    csv_paths: List[Path],
    anchor_csv_path: Optional[Path],
    id_col_hint: Optional[str],
    id_prefix: str = "SYN",
    force_new: bool = False
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Load existing id_map or create one.
    Order of operations:
      1) If synthetic_id_map.csv exists AND not --force, try to load it.
      2) Else, if --anchor-csv is provided and exists, build map from it.
      3) Else, infer from input CSVs by scanning for ID columns.
      4) If none found, return empty map.

    Returns (id_map_df, resolved_id_col_name or None).
    """
    idmap_path = outdir / "synthetic_id_map.csv"

    # 1) Load existing
    if idmap_path.exists() and not force_new:
        try:
            df = pd.read_csv(idmap_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(idmap_path, encoding="latin-1")
        if {"original_id", "synthetic_id"}.issubset(df.columns):
            log_info(f"Loaded existing ID map: {idmap_path}")
            # try to capture the original id column name if stored
            id_col_name = None
            if "id_col_name" in df.columns:
                vals = df["id_col_name"].dropna().unique().tolist()
                if vals:
                    id_col_name = str(vals[0])
            return df[["original_id", "synthetic_id"]].copy(), id_col_name
        else:
            log_warn("Existing synthetic_id_map.csv is missing required columns. Rebuilding...")

    outdir.mkdir(parents=True, exist_ok=True)

    # 2) Build from anchor if available
    if anchor_csv_path and anchor_csv_path.exists():
        anchor_df = safe_read_csv(anchor_csv_path)
        id_col = detect_id_column(anchor_df, id_col_hint)
        if id_col:
            originals = anchor_df[id_col].astype(str).dropna().unique()
            originals = np.array(sorted(originals))
            width = max(6, len(str(len(originals))))
            synthetic = [f"{id_prefix}{i:0{width}d}" for i in range(1, len(originals) + 1)]
            id_map = pd.DataFrame({"original_id": originals, "synthetic_id": synthetic})
            id_map["id_col_name"] = id_col
            id_map.to_csv(idmap_path, index=False)
            log_info(f"Wrote ID map: {idmap_path} (anchor id_col={id_col})")
            return id_map[["original_id", "synthetic_id"]].copy(), id_col
        else:
            log_warn(f"Anchor CSV provided but no ID column found: {anchor_csv_path.name}. Falling back to inference.")

    # 3) Infer from CSVs in the input folder
    id_map, resolved = infer_id_map_from_csvs(csv_paths, id_col_hint, id_prefix)
    if not id_map.empty:
        if resolved:
            id_map["id_col_name"] = resolved
        id_map.to_csv(idmap_path, index=False)
        log_info(f"Wrote ID map (inferred from CSVs): {idmap_path}")
        return id_map[["original_id", "synthetic_id"]].copy(), resolved

    # 4) None found
    log_warn("No anchor or detectable ID columns found in CSVs. Proceeding without cross-file ID mapping.")
    return pd.DataFrame(columns=["original_id", "synthetic_id"]), None


def attach_synthetic_ids(
    output_rows: int,
    original_ids: pd.Series,
    id_map: pd.DataFrame
) -> pd.Series:
    """
    Produce a synthetic ID column for a synthetic table:
    - Mirrors the original table's per-row ID frequency distribution
    - Replaces each original ID with its synthetic mapped ID
    - Upsamples or downsamples to match output_rows
    """
    if id_map is None or id_map.empty:
        width = max(6, len(str(output_rows)))
        syn_ids = [f"SYNROW{i:0{width}d}" for i in range(1, output_rows + 1)]
        return pd.Series(syn_ids)

    map_dict = dict(zip(id_map["original_id"].astype(str), id_map["synthetic_id"].astype(str)))
    base = original_ids.astype(str).map(map_dict)

    # Fill any missing with random from the known synthetic IDs
    missing = base.isna()
    if missing.any():
        pool = list(map_dict.values()) or [f"SYNFALL{i:06d}" for i in range(1, 1001)]
        base.loc[missing] = np.random.choice(pool, size=missing.sum(), replace=True)

    base = base.astype(str).values

    # Match output size
    if len(base) == output_rows:
        chosen = base
    elif len(base) > output_rows:
        idx = np.random.choice(len(base), size=output_rows, replace=False)
        chosen = base[idx]
    else:
        extra = np.random.choice(base, size=(output_rows - len(base)), replace=True)
        chosen = np.concatenate([base, extra])

    # Shuffle to avoid implying original order
    rng = np.random.default_rng()
    rng.shuffle(chosen)
    return pd.Series(chosen)


# ----------------------------
# Metadata + synthesizer (new API with fallback)
# ----------------------------

def build_metadata(model_df: pd.DataFrame, _id_col_unused: Optional[str] = None) -> tuple[object, Optional[str]]:
    """
    Build SDV metadata for a single table using the new Metadata API.
    Returns (metadata_object, table_name).

    - Preferred path: Metadata().detect_from_dataframe(data, table_name="table")
    - Fallback: SingleTableMetadata().detect_from_dataframe(data)
    """
    # Try new multi-table Metadata API
    try:
        table_name = "table"
        meta = Metadata()
        meta.detect_from_dataframe(data=model_df, table_name=table_name)
        # sanity: ensure table is present
        meta_dict = meta.to_dict()
        if "tables" not in meta_dict or table_name not in meta_dict["tables"]:
            raise RuntimeError("New Metadata did not register a single table as expected.")
        return meta, table_name
    except Exception:
        # Legacy SingleTableMetadata
        stm = SingleTableMetadata()
        stm.detect_from_dataframe(model_df)
        return stm, None


def choose_synthesizer(model: str, metadata: object, table_name: Optional[str], device: str = "cpu"):
    """
    Create a single-table synthesizer instance.
    - If `table_name` is provided (new Metadata API), pass it to the constructor.
    - Otherwise, assume legacy SingleTableMetadata and omit table_name.
    """
    model = model.lower()
    synth_cls = {
        "ctgan": CTGANSynthesizer,
        "copulagan": CopulaGANSynthesizer,
        "gaussian": GaussianCopulaSynthesizer,
    }.get(model)

    if synth_cls is None:
        raise ValueError(f"Unknown model: {model}")

    # New API style
    if table_name is not None:
        try:
            return synth_cls(metadata=metadata, table_name=table_name)
        except TypeError:
            # Some SDV builds may not accept table_name; fall through
            pass

    # Legacy style
    return synth_cls(metadata=metadata)


# ----------------------------
# CSV synthesis
# ----------------------------

def synthesize_csv_table(
    path: Path,
    outdir: Path,
    id_map: pd.DataFrame,
    id_col_hint: Optional[str],
    model: str,
    fallback_model: str,
    epochs: int,
    batch_size: int,
    rows: int,
    device: str,
    skip_existing: bool,
    force: bool
) -> Tuple[Path, int, str]:
    """
    Synthesize one CSV table, writing synthetic csv to outdir with suffix .synthetic.csv.
    Returns (output_path, n_rows_written, status).
    """
    out_path = outdir / f"{path.stem}.synthetic.csv"
    if out_path.exists() and skip_existing and not force:
        log_info(f"Skipping {path.name}: output exists -> {out_path}")
        try:
            n = max(0, sum(1 for _ in open(out_path, "r", encoding="utf-8")) - 1)
        except Exception:
            n = 0
        return out_path, n, "skipped_existing"

    # Read input
    df = safe_read_csv(path)
    id_col = detect_id_column(df, id_col_hint)

    # Drop ID col for modeling (re-attach later)
    model_df = df.copy()
    if id_col and id_col in model_df.columns:
        model_df = model_df.drop(columns=[id_col])

    # Build metadata (new API w/ table_name, or legacy)
    metadata, table_name = build_metadata(model_df)

    # Target number of rows
    n_target = rows if rows > 0 else len(model_df)

    def try_fit_and_sample(synth_name: str) -> Optional[pd.DataFrame]:
        try:
            synth = choose_synthesizer(synth_name, metadata, table_name, device=device)

            # Best-effort knobs (if present). Silently ignored if not supported.
            if hasattr(synth, "epochs"):
                setattr(synth, "epochs", epochs)
            if hasattr(synth, "batch_size"):
                setattr(synth, "batch_size", batch_size)

            synth.fit(model_df)
            fake = synth.sample(n_target)
            return fake
        except Exception as e:
            log_warn(f"Model {synth_name} failed on {path.name}: {e}")
            return None

    # Try primary and fallbacks
    fake = try_fit_and_sample(model)
    if fake is None and fallback_model:
        fake = try_fit_and_sample(fallback_model)
    if fake is None and fallback_model.lower() != "gaussian":
        fake = try_fit_and_sample("gaussian")

    if fake is None:
        log_error("All synthesizers failed for this table.")
        return out_path, 0, "failed"

    # Re-attach synthetic IDs
    if id_col:
        syn_ids = attach_synthetic_ids(
            output_rows=len(fake),
            original_ids=df[id_col],
            id_map=id_map
        )
        # insert at column 0 to keep familiar layout
        fake.insert(0, id_col, syn_ids)

    outdir.mkdir(parents=True, exist_ok=True)
    fake.to_csv(out_path, index=False)
    log_info(f"Wrote: {out_path}")
    return out_path, len(fake), "wrote"


# ----------------------------
# PLINK synthesis
# ----------------------------

def process_plink_bed(
    bed_path: Path,
    outdir: Path,
    id_map: pd.DataFrame,
    id_col_name: Optional[str],
    max_snps: int,
    skip_existing: bool,
    force: bool
) -> Tuple[Optional[Path], int, str]:
    """
    Load PLINK BED/BIM/FAM, synthesize genotype calls, and write a CSV.

    Output name: <bed_base>.synthetic_genotypes.csv
    Columns:
      SYN_SAMPLE_ID, SNP_<index_1>, SNP_<index_2>, ...

    Notes:
      - For speed and privacy, we shuffle genotypes per SNP (marginals preserved).
      - For very large panels, use --plink-max-snps to limit columns.
    """
    if not BED_READER_AVAILABLE:
        log_warn("bed_reader is not available. Skipping PLINK.")
        return None, 0, "skipped_no_reader"

    fam = bed_path.with_suffix(".fam")
    bim = bed_path.with_suffix(".bim")
    if not fam.exists() or not bim.exists():
        log_warn(f"Missing BIM/FAM neighbor files for {bed_path.name}. Skipping.")
        return None, 0, "skipped_missing_neighbors"

    out_path = outdir / f"{bed_path.stem}.synthetic_genotypes.csv"
    if out_path.exists() and skip_existing and not force:
        log_info(f"Skipping PLINK {bed_path.name}: output exists -> {out_path}")
        try:
            n = max(0, sum(1 for _ in open(out_path, "r", encoding="utf-8")) - 1)
        except Exception:
            n = 0
        return out_path, n, "skipped_existing"

    # Read FAM (sample IDs) and BIM (SNP count)
    log_info(f"Loading fam file {fam}")
    fam_df = pd.read_csv(fam, sep=r"\s+", header=None, engine="python")
    sample_ids = fam_df[1].astype(str).tolist()

    log_info(f"Loading bim file {bim}")
    #bim_df = pd.read_csv(bim, sep=r"\s+", header=None, engine="python")
    bim_df = pd.read_csv(
        bim,
        sep=r"\s+",
        header=None,
        names=["chrom", "snp", "cm", "pos", "a1", "a2"],
        engine="python"
    )
    total_snps = len(bim_df)

    # Read genotypes with bed_reader
    with open_bed(str(bed_path)) as bed:
        X = bed.read()  # shape: n_samples x n_snps, values in {0,1,2,nan}

    geno = pd.DataFrame(X)  # columns are 0..n_snps-1

    # cap SNPs if requested
    if max_snps and max_snps > 0 and geno.shape[1] > max_snps:
        cols = np.random.choice(geno.columns, size=max_snps, replace=False)
        cols = sorted(cols.tolist())
        geno = geno[cols]

    # Clean numeric dtypes, impute small missingness with column mode, and discretize to 0/1/2
    for c in geno.columns:
        col = pd.to_numeric(geno[c], errors="coerce")
        if col.isna().any():
            if col.notna().any():
                mode_val = col.dropna().round().astype(int).mode()
                fill_val = int(mode_val.iloc[0]) if not mode_val.empty else 0
            else:
                fill_val = 0
            col = col.fillna(fill_val)
        col = col.clip(lower=0.0, upper=2.0).round().astype(int)
        geno[c] = col

    # Simple synthetic via column-wise shuffle (preserve per-SNP marginals)
    syn = pd.DataFrame(index=geno.index, columns=geno.columns, dtype=int)
    rng = np.random.default_rng()
    for c in geno.columns:
        syn[c] = rng.permutation(geno[c].values)

    # Attach synthetic sample IDs (map originals if possible)
    if id_map is not None and not id_map.empty:
        map_dict = dict(zip(id_map["original_id"].astype(str), id_map["synthetic_id"].astype(str)))
        syn_ids = [map_dict.get(sid, f"SYN{idx:06d}") for idx, sid in enumerate(sample_ids, start=1)]
    else:
        syn_ids = [f"SYN{idx:06d}" for idx, _ in enumerate(sample_ids, start=1)]
    syn.insert(0, "SYN_SAMPLE_ID", syn_ids)

    outdir.mkdir(parents=True, exist_ok=True)
    syn.to_csv(out_path, index=False)
    log_info(f"Wrote: {out_path} (samples={syn.shape[0]}, snps={syn.shape[1]-1}/{total_snps})")
    return out_path, syn.shape[0], "wrote"


# ----------------------------
# FASTQ renaming
# ----------------------------

def rename_fastqs(
    fastq_paths: List[Path],
    outdir: Path,
    id_map: pd.DataFrame,
    skip_existing: bool,
    force: bool
) -> List[Tuple[Path, Path, str]]:
    """
    For each FASTQ, if any original ID from id_map appears in the filename,
    produce a copy with the synthetic ID in the filename.

    Returns a list of (src_path, dst_path, status):
      "renamed", "skipped_existing", "no_id_match", "skipped_no_map"
    """
    results = []
    fq_outdir = outdir / "fastq"
    fq_outdir.mkdir(parents=True, exist_ok=True)

    if id_map is None or id_map.empty:
        for p in fastq_paths:
            results.append((p, p, "skipped_no_map"))
        return results

    pairs = list(zip(id_map["original_id"].astype(str), id_map["synthetic_id"].astype(str)))

    for p in fastq_paths:
        name = p.name
        matched = False
        dst = None
        for orig, syn in pairs:
            if orig in name:
                new_name = name.replace(orig, syn)
                dst = fq_outdir / new_name
                matched = True
                break
        if not matched:
            results.append((p, p, "no_id_match"))
            continue

        if dst.exists() and skip_existing and not force:
            results.append((p, dst, "skipped_existing"))
            continue

        shutil.copy2(p, dst)
        results.append((p, dst, "renamed"))

    return results


# ----------------------------
# Summary printing
# ----------------------------

def print_summary(rows_info: List[Tuple[str, int, str]]) -> None:
    """
    Pretty-print a summary of work: file, rows written, status.
    """
    log_info("")
    log_info("Summary")
    log_info("-------")
    for fname, nrows, status in rows_info:
        log_info(f"{fname:35s} rows={nrows:8d}  status={status}")
    log_info("-------")


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synthesize CSV tables, PLINK genotypes, and FASTQs with cross-file ID consistency.",
        add_help=True
    )
    # Paths
    p.add_argument("--input", type=str, default="", help="Input directory with CSV/BED/FASTQ. If omitted, tries 'example_data' then '.'")
    p.add_argument("--output", type=str, default="synthetic_data", help="Output directory.")

    # ID mapping
    p.add_argument("--anchor-csv", type=str, default="", help="CSV filename in --input to build the ID map.")
    p.add_argument("--id-col", type=str, default="MASK_ID", help="Preferred name of the ID column if present.")
    p.add_argument("--id-prefix", type=str, default="SYN", help="Prefix for generated synthetic IDs.")

    # Execution control
    p.add_argument("--force", action="store_true", help="Force regeneration even if outputs already exist.")
    p.add_argument("--skip-existing", action="store_true", help="Skip files whose outputs already exist.")
    p.add_argument("--rows", type=int, default=0, help="Override rows to generate per CSV (0 means match input).")

    # CSV model controls
    p.add_argument("--model-csv", type=str, default="copulagan",
                   choices=["ctgan", "copulagan", "gaussian"],
                   help="Primary model for CSV tables.")
    p.add_argument("--model-fallback", type=str, default="gaussian",
                   choices=["ctgan", "copulagan", "gaussian"],
                   help="Fallback model for CSV tables.")
    p.add_argument("--epochs", type=int, default=50, help="Epochs for GAN-based models.")
    p.add_argument("--batch-size", type=int, default=500, help="Batch size for GAN-based models.")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device hint for GAN models.")

    # PLINK handling
    p.add_argument("--no-plink", action="store_true", help="Skip PLINK BED processing.")
    p.add_argument("--plink-max-snps", type=int, default=1500, help="Cap number of SNPs to synthesize (0 means use all).")

    # FASTQ handling
    p.add_argument("--no-fastq", action="store_true", help="Skip FASTQ renaming.")

    args = p.parse_args()

    # Default input directory
    if not args.input:
        candidate = Path("example_data")
        args.input = str(candidate if candidate.exists() else Path("."))

    return args


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    args = parse_args()

    # Resolve paths
    indir = Path(args.input).resolve()
    outdir = Path(args.output).resolve()

    if not indir.exists() or not indir.is_dir():
        log_error(f"Input directory does not exist: {indir}")
        log_info("Tip: pass --input <path> or place your data in ./example_data/")
        sys.exit(1)

    log_info("Effective configuration")
    log_info("-----------------------")
    log_info(f"input           : {indir}")
    log_info(f"output          : {outdir}")
    log_info(f"anchor-csv      : {args.anchor_csv or '(none)'}")
    log_info(f"id-col          : {args.id_col}")
    log_info(f"id-prefix       : {args.id_prefix}")
    log_info(f"skip-existing   : {args.skip_existing}")
    log_info(f"force           : {args.force}")
    log_info(f"rows            : {args.rows}")
    log_info(f"model-csv       : {args.model_csv}")
    log_info(f"model-fallback  : {args.model_fallback}")
    log_info(f"epochs          : {args.epochs}")
    log_info(f"batch-size      : {args.batch_size}")
    log_info(f"device          : {args.device}")
    log_info(f"no-plink        : {args.no_plink}")
    log_info(f"plink-max-snps  : {args.plink_max_snps}")
    log_info(f"no-fastq        : {args.no_fastq}")
    log_info("-----------------------")

    # Discover files
    csv_paths, bed_paths, fastq_paths = discover_files(indir)
    log_info(f"Found {len(csv_paths)} CSV, {len(bed_paths)} BED, {len(fastq_paths)} FASTQ in {indir}")

    # Build/load ID map
    anchor_path = (indir / args.anchor_csv) if args.anchor_csv else None
    id_map, id_col_name = load_or_build_id_map(
        outdir=outdir,
        csv_paths=csv_paths,
        anchor_csv_path=anchor_path,
        id_col_hint=args.id_col,
        id_prefix=args.id_prefix,
        force_new=args.force  # rebuild if --force
    )

    if id_map.empty:
        log_warn("Proceeding without ID map. IDs will be synthetic but not linked across files.")

    # Process CSVs
    summary_rows: List[Tuple[str, int, str]] = []
    for p in csv_paths:
        try:
            out_path, n, status = synthesize_csv_table(
                path=p,
                outdir=outdir,
                id_map=id_map,
                id_col_hint=(id_col_name or args.id_col),
                model=args.model_csv,
                fallback_model=args.model_fallback,
                epochs=args.epochs,
                batch_size=args.batch_size,
                rows=args.rows,
                device=args.device,
                skip_existing=args.skip_existing,
                force=args.force
            )
            summary_rows.append((p.name, n, status))
        except Exception as e:
            log_error(f"CSV failed for {p.name}: {e}")
            summary_rows.append((p.name, 0, "failed"))

    # Process PLINK
    if not args.no_plink and bed_paths:
        for bed in bed_paths:
            try:
                out_path, n, status = process_plink_bed(
                    bed_path=bed,
                    outdir=outdir,
                    id_map=id_map,
                    id_col_name=(id_col_name or args.id_col),
                    max_snps=args.plink_max_snps,
                    skip_existing=args.skip_existing,
                    force=args.force
                )
                summary_rows.append((bed.name, n, status))
            except Exception as e:
                log_error(f"PLINK failed for {bed.name}: {e}")
                summary_rows.append((bed.name, 0, "failed"))
    else:
        if args.no_plink:
            log_info("PLINK processing skipped by --no-plink")

    # FASTQ renaming
    if not args.no_fastq and fastq_paths:
        try:
            results = rename_fastqs(
                fastq_paths=fastq_paths,
                outdir=outdir,
                id_map=id_map,
                skip_existing=args.skip_existing,
                force=args.force
            )
            for src, dst, status in results:
                summary_rows.append((src.name, 0, status))
        except Exception as e:
            log_error(f"FASTQ renaming failed: {e}")
            for p in fastq_paths:
                summary_rows.append((p.name, 0, "failed"))
    else:
        if args.no_fastq:
            log_info("FASTQ renaming skipped by --no-fastq")

    # Final summary
    print_summary(summary_rows)


if __name__ == "__main__":
    main()
