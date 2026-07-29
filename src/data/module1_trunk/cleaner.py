from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .config import TrunkConfig
from .utils import setup_logger

logger = setup_logger(__name__)

COORDINATE_COLUMNS = [
    "left_shoulder_x", "left_shoulder_y",
    "right_shoulder_x", "right_shoulder_y",
    "left_hip_x", "left_hip_y",
    "right_hip_x", "right_hip_y",
]
VISIBILITY_COLUMNS = [
    "left_shoulder_visibility",
    "right_shoulder_visibility",
    "left_hip_visibility",
    "right_hip_visibility",
]


def clean_coordinates(raw_df: pd.DataFrame, config: TrunkConfig, video_id: str) -> pd.DataFrame:
    """Filter low-visibility frames, interpolate gaps, and smooth trunk landmarks."""
    pdf = raw_df.sort_values("frame_id").reset_index(drop=True).copy()

    low_visibility_mask = (pdf[VISIBILITY_COLUMNS] < config.visibility_threshold).any(axis=1)
    pdf.loc[low_visibility_mask, COORDINATE_COLUMNS] = np.nan

    if pdf[COORDINATE_COLUMNS].isna().all().all():
        logger.warning("No valid frames left after visibility filtering for %s", video_id)
        return pdf

    pdf[COORDINATE_COLUMNS] = pdf[COORDINATE_COLUMNS].interpolate(
        method="linear", limit_direction="both"
    )
    pdf[COORDINATE_COLUMNS] = (
        pdf[COORDINATE_COLUMNS].ffill().bfill()
    )

    if len(pdf) >= config.smoothing_window:
        for column in COORDINATE_COLUMNS:
            pdf[column] = savgol_filter(
                pdf[column].to_numpy(dtype=float),
                window_length=config.smoothing_window,
                polyorder=config.smoothing_polyorder,
                mode="interp",
            )

    # Outlier rejection: a frame-to-frame jump larger than the threshold in any
    # tracked point is treated as a landmark glitch rather than real motion.
    outlier_mask = pd.Series(False, index=pdf.index)
    for prefix in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
        dx = np.diff(pdf[f"{prefix}_x"], prepend=pdf[f"{prefix}_x"].iloc[0])
        dy = np.diff(pdf[f"{prefix}_y"], prepend=pdf[f"{prefix}_y"].iloc[0])
        displacement = np.sqrt(dx**2 + dy**2)
        outlier_mask |= displacement > config.outlier_threshold

    if outlier_mask.any():
        logger.info(
            "Detected %s outlier frames in %s", int(outlier_mask.sum()), video_id
        )
        pdf.loc[outlier_mask, COORDINATE_COLUMNS] = np.nan
        pdf[COORDINATE_COLUMNS] = pdf[COORDINATE_COLUMNS].interpolate(
            method="linear", limit_direction="both"
        )
        pdf[COORDINATE_COLUMNS] = pdf[COORDINATE_COLUMNS].ffill().bfill()

    pdf["low_visibility"] = low_visibility_mask.astype(int)
    return pdf


def save_cleaned_coordinates(cleaned_df: pd.DataFrame, video_id: str, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_id}_cleaned.csv"

    cleaned_df.to_csv(output_path, index=False)
    logger.info("Saved cleaned trunk coordinates for %s to %s", video_id, output_path)
    return output_path
