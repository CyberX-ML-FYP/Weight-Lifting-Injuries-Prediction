from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import setup_logger

logger = setup_logger(__name__)


def _extract_lift_label(video_id: str) -> int | None:
    name = str(video_id).lower()
    if "good" in name:
        return 0
    if "bad" in name:
        return 1
    logger.warning(
        "Could not infer label from video_id %r (expected 'good' or 'bad' in name)",
        video_id,
    )
    return None


def extract_bar_path_features(cleaned_df, video_id: str) -> pd.DataFrame:
    if hasattr(cleaned_df, "compute"):
        pdf = cleaned_df.compute()
    else:
        pdf = cleaned_df.copy()

    if pdf.empty:
        logger.warning("No cleaned coordinates available for %s", video_id)
        return pd.DataFrame(
            columns=[
                "video_id",
                "frame_index",
                "timestamp_ms",
                "bar_x",
                "bar_y",
                "displacement",
                "bar_velocity",
                "bar_deviation",
                "normalized_flag",
            ]
        )

    pdf = pdf.sort_values("frame_id").reset_index(drop=True).copy()
    pdf["frame_index"] = pdf["frame_id"].astype(int)
    pdf["timestamp_ms"] = (
        pdf["frame_index"] * 1000.0 / pdf["fps"].replace(0, np.nan)
    ).fillna(0.0)

    x = pdf["x"].to_numpy(dtype=float)
    y = pdf["y"].to_numpy(dtype=float)

    displacements = np.zeros(len(pdf), dtype=float)
    if len(pdf) > 1:
        displacements[1:] = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)

    time_delta = np.diff(pdf["timestamp_ms"].to_numpy(dtype=float), prepend=0.0)
    velocity = np.zeros(len(pdf), dtype=float)
    velocity[1:] = displacements[1:] / np.maximum(time_delta[1:], 1e-6)

    center_x = float(np.nanmean(x))
    center_y = float(np.nanmean(y))
    bar_deviation = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    normalized_flag = (bar_deviation < 0.1).astype(int)

    pdf["video_id"] = video_id
    pdf["bar_x"] = x
    pdf["bar_y"] = y
    pdf["displacement"] = displacements
    pdf["bar_velocity"] = velocity
    pdf["bar_deviation"] = bar_deviation
    pdf["normalized_flag"] = normalized_flag

    output_columns = [
        "video_id",
        "frame_index",
        "timestamp_ms",
        "bar_x",
        "bar_y",
        "displacement",
        "bar_velocity",
        "bar_deviation",
        "normalized_flag",
    ]

    logger.info("Extracted %s frame-level bar path rows for %s", len(pdf), video_id)
    return pdf[output_columns]


def summarize_bar_path_features(frame_df: pd.DataFrame, video_id: str) -> pd.DataFrame:
    """Convert frame-level bar-path data into one row per lift."""
    if frame_df is None or frame_df.empty:
        return pd.DataFrame(
            columns=[
                "video_id",
                "lift_id",
                "label",
                "max_deviation",
                "avg_deviation",
                "path_smoothness",
                "peak_vertical_velocity",
                "time_to_peak_velocity",
                "total_displacement",
                "jerk_like_movements",
            ]
        )

    pdf = frame_df.copy()
    if "bar_velocity" not in pdf.columns or "bar_deviation" not in pdf.columns:
        return pd.DataFrame(
            columns=[
                "video_id",
                "lift_id",
                "label",
                "max_deviation",
                "avg_deviation",
                "path_smoothness",
                "peak_vertical_velocity",
                "time_to_peak_velocity",
                "total_displacement",
                "jerk_like_movements",
            ]
        )

    pdf = pdf.sort_values("frame_index").reset_index(drop=True)
    velocity = pdf["bar_velocity"].to_numpy(dtype=float)
    deviation = pdf["bar_deviation"].to_numpy(dtype=float)

    if len(pdf) > 1:
        smoothness = float(np.nanmean(np.abs(np.diff(velocity))))
        peak_idx = int(np.argmax(velocity))
        peak_vertical_velocity = float(velocity[peak_idx])
        time_to_peak_velocity = float(peak_idx / max(len(pdf) - 1, 1))
    else:
        smoothness = 0.0
        peak_idx = 0
        peak_vertical_velocity = float(velocity[0]) if len(velocity) else 0.0
        time_to_peak_velocity = 0.0

    row = {
        "video_id": video_id,
        "lift_id": video_id,
        "label": _extract_lift_label(video_id),
        "max_deviation": float(np.nanmax(deviation)) if len(deviation) else np.nan,
        "avg_deviation": float(np.nanmean(deviation)) if len(deviation) else np.nan,
        "path_smoothness": smoothness,
        "peak_vertical_velocity": peak_vertical_velocity,
        "time_to_peak_velocity": time_to_peak_velocity,
        "total_displacement": float(
            np.nansum(pdf.get("displacement", pd.Series(0.0, index=pdf.index)))
        )
        if len(pdf)
        else np.nan,
        "jerk_like_movements": int(
            np.sum(
                np.abs(np.diff(velocity)) > np.nanmedian(np.abs(np.diff(velocity))) * 2
            )
        )
        if len(velocity) > 1
        else 0,
    }

    return pd.DataFrame([row])
