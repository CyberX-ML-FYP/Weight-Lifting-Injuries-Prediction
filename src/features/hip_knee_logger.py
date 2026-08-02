"""
Hip & Knee Analysis — Structured logging utilities. (Improvement 7)

Provides a single ``get_logger`` factory that writes structured, timestamped
log records (processing steps, errors, predictions, scores and execution
time) to both the console and a rotating log file under
``reports/logs/hip_knee_analysis.log``. Using one shared logger avoids
scattering ``print`` statements throughout the pipeline and makes it possible
to audit what happened for a given lift after the fact.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler

from src.features.hip_knee_config import LOGS_DIR

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LOG_FILE_NAME = "hip_knee_analysis.log"


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger that writes to console and a rotating file.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A ``logging.Logger`` instance with a console handler and a rotating
        file handler already attached (attached only once, even if this
        function is called repeatedly for the same logger name).
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured — avoid attaching duplicate handlers.
        return logger

    logger.setLevel(logging.INFO)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        LOGS_DIR / _LOG_FILE_NAME,
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter(_LOG_FORMAT)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger


@contextmanager
def log_execution_time(logger: logging.Logger, task_name: str) -> Iterator[None]:
    """
    Context manager that logs the wall-clock execution time of a code block.

    Example:
        >>> logger = get_logger(__name__)
        >>> with log_execution_time(logger, "video frame extraction"):
        ...     extract_frames(video_path, output_dir)

    Args:
        logger: Logger to write the timing message to.
        task_name: Human readable description of the timed task.
    """
    start = time.perf_counter()
    logger.info("START | %s", task_name)
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - start
        logger.exception("FAILED | %s | elapsed=%.3fs", task_name, elapsed)
        raise
    else:
        elapsed = time.perf_counter() - start
        logger.info("DONE  | %s | elapsed=%.3fs", task_name, elapsed)
