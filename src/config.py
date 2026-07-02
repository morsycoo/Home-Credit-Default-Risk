"""
Application configuration.
"""

from pathlib import Path


# ============================================================================
# Project Directories
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SRC_DIR = BASE_DIR / "src"

API_DIR = BASE_DIR / "api"

MODELS_DIR = BASE_DIR / "models"

DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

LOGS_DIR = BASE_DIR / "logs"

ARTIFACTS_DIR = BASE_DIR / "artifacts"


# ============================================================================
# Model Files
# ============================================================================

MODEL_NAME = "home_credit_optuna_lgb.pkl"

MODEL_PATH = MODELS_DIR / MODEL_NAME

FEATURES_PATH = MODELS_DIR / "selected_features.pkl"

THRESHOLD_PATH = MODELS_DIR / "best_threshold.pkl"


# ============================================================================
# API
# ============================================================================

API_TITLE = "Credit Risk Intelligence API"

API_DESCRIPTION = (
    "Predict customer credit default risk using a trained LightGBM model."
)

API_VERSION = "1.0.0"


# ============================================================================
# Logging
# ============================================================================

LOG_LEVEL = "INFO"

LOG_FILE_NAME = "app.log"

LOG_MAX_BYTES = 5 * 1024 * 1024      # 5 MB

LOG_BACKUP_COUNT = 5