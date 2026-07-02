"""
Inference pipeline for the Credit Risk Intelligence System.
"""

from __future__ import annotations

import pandas as pd

from src.artifacts import (
    load_selected_features,
    load_threshold,
)
from src.model import load_model
from src.preprocessing import preprocess_data


def prepare_dataframe(data: dict) -> pd.DataFrame:
    """
    Convert request data into a pandas DataFrame.
    """

    return pd.DataFrame([data])


def align_features(
    df: pd.DataFrame,
    selected_features: list[str],
) -> pd.DataFrame:
    """
    Align dataframe columns with the features used during training.
    """

    df = df.copy()

    # Find missing features
    missing_features = [
        feature
        for feature in selected_features
        if feature not in df.columns
    ]

    # Add all missing features at once
    if missing_features:
        missing_df = pd.DataFrame(
            0,
            index=df.index,
            columns=missing_features,
        )

        df = pd.concat(
            [df, missing_df],
            axis=1,
        )

    # Keep only the features used during training
    df = df[selected_features]

    return df


def predict_probability(
    model,
    df: pd.DataFrame,
) -> float:
    """
    Predict probability of default.
    """

    probability = model.predict_proba(df)[0][1]

    return float(probability)


def build_response(
    probability: float,
    threshold: float,
) -> dict:
    """
    Build prediction response.
    """

    prediction = int(probability >= threshold)

    return {
        "prediction": prediction,
        "risk_score": round(probability, 4),
        "threshold": round(threshold, 4),
    }


def predict(data: dict) -> dict:
    """
    Complete inference pipeline.
    """

    # Convert request to DataFrame
    df = prepare_dataframe(data)

    # Apply preprocessing
    df = preprocess_data(df)

    # Load training artifacts
    selected_features = load_selected_features()
    threshold = load_threshold()

    # Align dataframe with training features
    df = align_features(
        df,
        selected_features,
    )

    # Load trained model
    model = load_model()

    # Predict
    probability = predict_probability(
        model,
        df,
    )

    # Build response
    return build_response(
        probability,
        threshold,
    )


if __name__ == "__main__":

    sample_customer = {
        "AMT_INCOME_TOTAL": 180000,
        "AMT_CREDIT": 450000,
        "AMT_ANNUITY": 21000,
        "DAYS_EMPLOYED": -1500,
    }

    try:

        result = predict(sample_customer)

        print("\nPrediction Result")
        print("-------------------------")
        print(result)

    except Exception as e:

        print("\nInference Failed")
        print("-------------------------")
        print(type(e).__name__)
        print(e)