"""Logging setup shared by the app, scripts and tests."""

from __future__ import annotations

import logging
import os

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure_logging(level: str | None = None) -> None:
    """Configure the root logger once. Level comes from GPS_LOG_LEVEL if not given."""
    level_name = (level or os.getenv("GPS_LOG_LEVEL", "INFO")).upper()
    root = logging.getLogger()
    if not any(getattr(h, "_gps_seville", False) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        handler._gps_seville = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.setLevel(getattr(logging, level_name, logging.INFO))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
