# mmai/synth/tabular.py

"""
Wrapper for SDV single-table GaussianCopula synthesis.

Provides reproducible synthetic generation for numeric genotype matrices.
"""

import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata


def synthesize_gaussiancopula(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic dataframe using GaussianCopula model.

    Parameters:
        df (pd.DataFrame): Input genotype matrix
        seed (int): Reproducibility seed

    Returns:
        pd.DataFrame: Synthetic data
    """
    meta = SingleTableMetadata()
    meta.detect_from_dataframe(df)

    synth = GaussianCopulaSynthesizer(metadata=meta, enforce_rounding=True)
    synth.fit(df)
    return synth.sample(len(df))
