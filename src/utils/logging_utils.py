"""
Logging configuration for the VLM audit pipeline.
Call setup_logging() once at the start of each script.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    name: str = "vlmaudit",
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure root logger with console + optional file handler.
    Returns the named logger.
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
