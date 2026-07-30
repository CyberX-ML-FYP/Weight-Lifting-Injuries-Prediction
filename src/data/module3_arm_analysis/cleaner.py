"""
Module 3 — Cleaner
Pass 2 of the pipeline: takes the raw per-frame landmark rows from
landmark_extractor.py and filters low-visibility frames, interpolates
gaps, and smooths the pixel-space coordinates before any angle is
computed from them.
"""
import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

POINT_NAMES = [
    "left_shoulder", "left_elbow", "left_wrist", "left_hip",
    "right_shoulder", "right_elbow", "right_wrist", "right_hip",
]
COORDINATE_COLUMNS = [f"{name}_{axis}" for name in POINT_NAMES for axis in ("x", "y")]
VISIBILITY_COLUMNS = [f"{name}_visibility" for name in POINT_NAMES]

VISIBILITY_THRESHOLD = 0.5
OUTLIER_THRESHOLD_PX = 80.0  # pixel-space frame-to-frame jump treated as a glitch
SMOOTHING_WINDOW = 7
SMOOTHING_POLYORDER = 2


def clean_landmarks(raw_rows):
    """Filter, interpolate, and smooth raw per-frame landmark rows.

    Returns a DataFrame with the same coordinate columns as the input,
    fully filled (no NaNs) wherever at least one frame had a detection,
    plus a `low_visibility` column marking frames that needed correction.
    """
    pdf = pd.DataFrame(raw_rows).sort_values("frame").reset_index(drop=True)

    if pdf.empty or pdf[COORDINATE_COLUMNS].isna().all().all():
        pdf["low_visibility"] = 1
        return pdf

    low_visibility_mask = (pdf[VISIBILITY_COLUMNS] < VISIBILITY_THRESHOLD).any(axis=1)
    pdf.loc[low_visibility_mask, COORDINATE_COLUMNS] = np.nan

    pdf[COORDINATE_COLUMNS] = pdf[COORDINATE_COLUMNS].interpolate(
        method="linear", limit_direction="both"
    )
    pdf[COORDINATE_COLUMNS] = pdf[COORDINATE_COLUMNS].ffill().bfill()

    if len(pdf) >= SMOOTHING_WINDOW:
        for column in COORDINATE_COLUMNS:
            pdf[column] = savgol_filter(
                pdf[column].to_numpy(dtype=float),
                window_length=SMOOTHING_WINDOW,
                polyorder=SMOOTHING_POLYORDER,
                mode="interp",
            )

    # Outlier rejection: a frame-to-frame jump larger than the threshold in
    # any tracked point is treated as a landmark glitch rather than real
    # motion (e.g. pose detector briefly locking onto a spectator).
    outlier_mask = pd.Series(False, index=pdf.index)
    for name in POINT_NAMES:
        dx = np.diff(pdf[f"{name}_x"], prepend=pdf[f"{name}_x"].iloc[0])
        dy = np.diff(pdf[f"{name}_y"], prepend=pdf[f"{name}_y"].iloc[0])
        displacement = np.sqrt(dx**2 + dy**2)
        outlier_mask |= displacement > OUTLIER_THRESHOLD_PX

    if outlier_mask.any():
        pdf.loc[outlier_mask, COORDINATE_COLUMNS] = np.nan
        pdf[COORDINATE_COLUMNS] = pdf[COORDINATE_COLUMNS].interpolate(
            method="linear", limit_direction="both"
        )
        pdf[COORDINATE_COLUMNS] = pdf[COORDINATE_COLUMNS].ffill().bfill()

    pdf["low_visibility"] = low_visibility_mask.astype(int)
    return pdf


def save_cleaned_landmarks(cleaned_df, output_path):
    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    return output_path
