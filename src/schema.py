from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Input schema for prediction requests.
    """

    features: dict[str, Any] = Field(
        ...,
        description="Customer features used for inference.",
        example={
            "AMT_INCOME_TOTAL": 180000,
            "AMT_CREDIT": 450000,
            "AMT_ANNUITY": 21000,
            "DAYS_EMPLOYED": -1500
        },
    )


class PredictionResponse(BaseModel):
    """
    Output schema.
    """

    prediction: int = Field(
        ...,
        description="Predicted class.",
        example=1,
    )

    risk_score: float = Field(
        ...,
        description="Probability of default.",
        example=0.6801,
    )

    threshold: float = Field(
        ...,
        description="Decision threshold.",
        example=0.65,
    )