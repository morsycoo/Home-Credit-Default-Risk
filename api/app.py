"""
FastAPI application for the Credit Risk Intelligence System.
"""

from fastapi import FastAPI, HTTPException
from src.logger import logger
import time

from src.config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
)
from src.inference import predict
from src.schema import (
    PredictionRequest,
    PredictionResponse,
)

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)


@app.get(
    "/",
    tags=["Health Check"],
)
def root() -> dict:

    logger.info("Health check requested.")

    return {
        "message": "Credit Risk Intelligence API is running."
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
def predict_credit_risk(
    request: PredictionRequest,
) -> PredictionResponse:
    """
    Predict customer credit default risk.
    """
    start_time = time.perf_counter()
    logger.info("Prediction request received.")

    try:

        result = predict(request.features)

        execution_time = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Prediction completed successfully. "
            f"Prediction={result['prediction']}, "
            f"Risk Score={result['risk_score']:.4f}, "
            f"Execution Time={execution_time:.2f} ms"
)

        return PredictionResponse(**result)

    except FileNotFoundError as e:

        logger.exception("Model file not found.")

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    except ValueError as e:

        logger.exception("Validation error.")

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:

        logger.exception("Prediction failed.")

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )