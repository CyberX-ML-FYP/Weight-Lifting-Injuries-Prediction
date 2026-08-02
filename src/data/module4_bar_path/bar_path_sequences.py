"""
Module 4 (bar_path) — LSTM sequence preparation.

RF/XGBoost train on 7 hand-summarised numbers per lift (bar_path_features
.csv); an LSTM instead needs the raw-ish per-frame (x, y) bar trajectory
for every lift, all at the SAME fixed length T so they can be stacked into
one (n_lifts, T, 2) array.

Lift durations vary a lot (69-418 frames, median ~246 across the 58
cleaned videos), so every trajectory is linearly resampled -- not padded
or truncated -- to exactly T points along its own normalised progress
(0.0 = first lift-phase frame, 1.0 = last). Resampling keeps the full
shape of every lift regardless of how long it took, which matters because
truncating a 418-frame lift to T=150 would silently discard everything
after the first third of it (interpolation was chosen over pad/truncate
for exactly this reason).

Output: X of shape (n_lifts, T, 2) [x, y per frame] and y of shape
(n_lifts,) [0=good, 1=bad], saved as a single .npz so bar_path_train_lstm
.py doesn't need to touch the raw CSVs at all.

Run:
    python -m src.data.module4_bar_path.bar_path_sequences
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import BarPathConfig
from .feature_extractor import _extract_lift_label
from .utils import setup_logger

logger = setup_logger(__name__)

SEQUENCE_LENGTH = 150  # T, per the README's Attention-LSTM spec
SEQUENCES_PATH_NAME = "bar_path_sequences.npz"


def resample_trajectory(x: np.ndarray, y: np.ndarray, length: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Linearly resample an (n,) x and (n,) y trajectory to exactly
    `length` points along normalised progress through the lift. Returns
    an (length, 2) array. Requires at least 2 points to interpolate."""
    n = len(x)
    if n < 2:
        raise ValueError(f"Need at least 2 points to resample, got {n}")

    original_progress = np.linspace(0.0, 1.0, n)
    target_progress = np.linspace(0.0, 1.0, length)

    x_resampled = np.interp(target_progress, original_progress, x)
    y_resampled = np.interp(target_progress, original_progress, y)
    return np.stack([x_resampled, y_resampled], axis=1)


def build_sequence_dataset(config: BarPathConfig, length: int = SEQUENCE_LENGTH):
    """Read every <video_id>_cleaned.csv, resample its (x, y) trajectory,
    and stack into X (n_lifts, length, 2) / y (n_lifts,) / video_ids."""
    cleaned_paths = sorted(config.processed_output_dir.glob("*_cleaned.csv"))
    logger.info("Found %s cleaned coordinate CSV(s) in %s", len(cleaned_paths), config.processed_output_dir)

    sequences = []
    labels = []
    video_ids = []
    skipped = []

    for path in cleaned_paths:
        video_id = path.stem.replace("_cleaned", "")
        df = pd.read_csv(path)
        if len(df) < 2:
            logger.warning("Skipping %s: only %s usable frame(s)", video_id, len(df))
            skipped.append(video_id)
            continue

        label = _extract_lift_label(video_id)
        if label is None:
            logger.warning("Skipping %s: could not infer good/bad label from name", video_id)
            skipped.append(video_id)
            continue

        df = df.sort_values("frame_id").reset_index(drop=True)
        seq = resample_trajectory(df["x"].to_numpy(dtype=float), df["y"].to_numpy(dtype=float), length)

        sequences.append(seq)
        labels.append(label)
        video_ids.append(video_id)

    if not sequences:
        raise RuntimeError("No usable sequences found -- check data/processed/module4/*_cleaned.csv")

    X = np.stack(sequences, axis=0)
    y = np.array(labels, dtype=int)
    logger.info(
        "Built sequence dataset: X=%s y=%s (%s good / %s bad); skipped %s",
        X.shape, y.shape, int((y == 0).sum()), int((y == 1).sum()), skipped,
    )
    return X, y, video_ids


def main() -> None:
    config = BarPathConfig()
    X, y, video_ids = build_sequence_dataset(config)

    out_path = config.root_dir / "data" / "processed" / "module4" / SEQUENCES_PATH_NAME
    np.savez(out_path, X=X, y=y, video_ids=np.array(video_ids))
    logger.info("Saved sequence dataset to %s", out_path)


if __name__ == "__main__":
    main()
