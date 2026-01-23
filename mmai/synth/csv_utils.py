"""
csv_utils.py

Synthetic tabular data generation utilities using SDV-based generative models.

This module owns CSV synthesis only. It does not manage identifiers globally
and does not coordinate pipeline execution.

Author: Kenneth Young, PhD
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import (
    CTGANSynthesizer,
    CopulaGANSynthesizer,
    GaussianCopulaSynthesizer,
)

from mmai.synth.id_utils import detect_id_column

def build_metadata(df: pd.DataFrame) -> SingleTableMetadata:
    meta = SingleTableMetadata()
    meta.detect_from_dataframe(df)
    return meta


def choose_synthesizer(model: str, metadata: SingleTableMetadata):
    model = model.lower()
    if model == "copulagan":
        return CopulaGANSynthesizer(metadata=metadata, enforce_rounding=False)
    if model == "ctgan":
        return CTGANSynthesizer(metadata=metadata, enforce_rounding=False)
    if model == "gaussian":
        return GaussianCopulaSynthesizer(metadata=metadata, enforce_rounding=False)
    raise ValueError(f"Unknown model: {model}")


def synthesize_csv_table(
    path: Path,
    outdir: Path,
    id_map: pd.DataFrame,
    id_col_hint: Optional[str],
    id_col_name: Optional[str],
    model: str,
    fallback_model: str,
    epochs: int,
    rows: int,
) -> Tuple[Path, int]:
    """
    Generate a synthetic version of a single CSV table.

    - Detects the source identifier column (e.g. EXAMP_maskid)
    - Replaces real IDs with synthetic IDs deterministically
    - Uses a canonical identifier column name in outputs
    - Prevents ID leakage into generative modeling
    """

    df = pd.read_csv(path)

    # ------------------------------------------------------------------
    # Step 1: Detect source ID column (e.g. EXAMP_maskid)
    # ------------------------------------------------------------------
    source_id_col = detect_id_column(df, id_col_hint)

    # Canonical output ID column name
    if id_col_name is None:
        id_col_name = id_col_hint or "MASK_ID"

    # ------------------------------------------------------------------
    # Step 2: Map real IDs -> synthetic IDs (row-wise)
    # ------------------------------------------------------------------
    synthetic_ids = None
    if source_id_col and not id_map.empty:
        mapping = dict(
            zip(
                id_map["original_id"].astype(str),
                id_map["synthetic_id"].astype(str),
            )
        )

        synthetic_ids = (
            df[source_id_col]
            .astype(str)
            .map(mapping)
        )

        # Defensive check
        if synthetic_ids.isna().any():
            raise ValueError(
                f"Unmapped IDs detected in {path.name}. "
                "Ensure synthetic_id_map is complete."
            )

    # ------------------------------------------------------------------
    # Step 3: Prepare modeling dataframe (NO IDs)
    # ------------------------------------------------------------------
    model_df = df.drop(columns=[source_id_col]) if source_id_col else df.copy()

    n_target = rows if rows > 0 else len(model_df)

    metadata = build_metadata(model_df)
    synth = choose_synthesizer(model, metadata)

    if hasattr(synth, "epochs"):
        synth.epochs = epochs

    synth.fit(model_df)
    fake = synth.sample(n_target)

    # ------------------------------------------------------------------
    # Step 4: Reattach synthetic ID column with canonical name
    # ------------------------------------------------------------------
    if synthetic_ids is not None:
        # Resize deterministically if row counts differ
        if len(synthetic_ids) != len(fake):
            synthetic_ids = np.random.choice(
                synthetic_ids.values,
                size=len(fake),
                replace=True,
            )

        fake.insert(0, id_col_name, synthetic_ids)

    # ------------------------------------------------------------------
    # Step 5: Write output
    # ------------------------------------------------------------------
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{path.stem}.synthetic.csv"
    fake.to_csv(out_path, index=False)

    return out_path, len(fake)
