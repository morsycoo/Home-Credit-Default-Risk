"""
Data preprocessing utilities for the Credit Risk Intelligence System.

This module contains reusable preprocessing functions that are shared
between training and inference.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def fix_days_employed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace the anomalous DAYS_EMPLOYED value (365243) with NaN.

    In the Home Credit dataset, the value 365243 is a placeholder
    representing missing employment information.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """

    df = df.copy()

    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(
            365243,
            np.nan
        )

    return df


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize feature names by removing unsupported characters.

    Example:
        DAYS_BIRTH -> DAYS_BIRTH
        NAME_CONTRACT_TYPE -> NAME_CONTRACT_TYPE
        EXT_SOURCE_1 -> EXT_SOURCE_1

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with cleaned column names.
    """

    df = df.copy()

    df.columns = [
        re.sub(
            r"[^A-Za-z0-9_]+",
            "_",
            str(col)
        )
        for col in df.columns
    ]

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the complete preprocessing pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe.

    Returns
    -------
    pd.DataFrame
        Preprocessed dataframe.
    """

    df = fix_days_employed(df)
    df = clean_columns(df)

    return df