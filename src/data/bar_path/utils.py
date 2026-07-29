from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def compute_euclidean_distance(
    x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray
) -> np.ndarray:
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def compute_perpendicular_distances(
    x: np.ndarray, y: np.ndarray, start: np.ndarray, end: np.ndarray
) -> np.ndarray:
    if np.allclose(start, end):
        return np.zeros_like(x, dtype=float)

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    numerator = np.abs(dy * x - dx * y + end[0] * start[1] - end[1] * start[0])
    denominator = np.hypot(dx, dy)
    return numerator / denominator


def ensure_path(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_coordinate(value: float, divisor: float) -> float:
    if divisor == 0:
        return 0.0
    return float(value / divisor)
