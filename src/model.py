"""
Model loading utilities.

This module is responsible for loading the trained
machine learning model used during inference.
"""

from functools import lru_cache

import joblib

from src.config import MODEL_PATH


@lru_cache(maxsize=1)
def load_model():
    """
    Load the trained machine learning model.

    Returns
    -------
    Any
        Loaded model instance.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)