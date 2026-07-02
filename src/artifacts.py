"""
Artifact loading utilities.

This module is responsible for loading all artifacts
required during inference.
"""

from functools import lru_cache

import joblib

from src.config import (
    FEATURES_PATH,
    THRESHOLD_PATH,
)


@lru_cache(maxsize=1)
def load_selected_features() -> list[str]:
    """
    Load the selected features used during model training.

    Returns
    -------
    list[str]
        List of selected feature names.
    """

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Selected features not found: {FEATURES_PATH}"
        )

    return joblib.load(FEATURES_PATH)


@lru_cache(maxsize=1)
def load_threshold() -> float:
    """
    Load the optimal prediction threshold.

    Returns
    -------
    float
        Optimal classification threshold.
    """

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Threshold not found: {THRESHOLD_PATH}"
        )

    return float(joblib.load(THRESHOLD_PATH))