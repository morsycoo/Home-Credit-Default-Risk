"""
Application logging configuration.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from src.config import (
    LOGS_DIR,
    LOG_FILE_NAME,
    LOG_LEVEL,
    LOG_MAX_BYTES,
    LOG_BACKUP_COUNT,
)

LOGS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_FILE = LOGS_DIR / LOG_FILE_NAME

logger = logging.getLogger("credit_risk")

logger.setLevel(LOG_LEVEL)

if not logger.handlers:

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )

    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False