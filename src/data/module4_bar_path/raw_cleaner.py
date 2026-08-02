"""
Module 4 (bar_path) — Pass 2: clean the raw landmark CSVs from
raw_extractor.py and turn them into per-lift bar-path features.

This is a separate cleaning path from cleaner.py, which was built for the
older landmark_extractor.py/pipeline.py schema (left_visibility /
right_visibility = wrist visibility only, no lift_phase column). Our raw
CSVs carry per-landmark visibility columns (left_wrist_visibility, etc.)
and a lift_phase flag from find_lift_window(), so cleaning needs an extra
first step: drop walk-in/walk-out frames before any interpolation or
smoothing touches the signal, otherwise those phases would bleed into the
lift's bar-path features.

Pipeline: lift_phase filter -> wrist-visibility filter -> interpolate gaps
-> outlier removal (rolling-median distance) -> re-interpolate ->
Savitzky-Golay smoothing -> the existing feature_extractor
.extract_bar_path_features / summarize_bar_path_features (unchanged,
already schema-compatible once we hand it fps + normalised x/y).

Outlier removal runs on a rolling-median distance rather than cleaner.py's
frame-to-frame diff: select_lifter() can occasionally lock onto a
bystander for several consecutive frames (observed up to ~3 frames in one
video), and a bad run that long is invisible to a check against only the
immediate previous/next frame. A point far from the median of a ~9-frame
window is a bad selection almost by definition, as long as bad frames stay
a minority of that window -- confirmed against several videos with known
bad-selection runs (all showed the bad frames at 600-850px from the local
median, while every legitimate motion peak found across the dataset,
including fast bar drops and close-up motion blur, stayed under ~300px).

Run:
    python -m src.data.module4_bar_path.raw_cleaner
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .config import BarPathConfig
from .feature_extractor import extract_bar_path_features, summarize_bar_path_features
from .storage import save_frame_level_features
from .utils import setup_logger

logger = setup_logger(__name__)

COORDINATE_COLUMNS = ["bar_x", "bar_y"]
WRIST_VISIBILITY_COLUMNS = ["left_wrist_visibility", "right_wrist_visibility"]

# Distance (px) from the rolling-median bar position beyond which a frame
# is treated as a bad selection rather than real (if fast) bar motion.
# See module docstring for how this was calibrated.
OUTLIER_DISTANCE_THRESHOLD = 300.0
OUTLIER_WINDOW = 9


WRIST_SWAP_REFERENCE_WINDOW = 5   # frames of recent history used as the "expected" position
WRIST_SWAP_MARGIN = 0.5           # swapped distance must be <= this fraction of as-is distance


def fix_wrist_label_swaps(pdf: pd.DataFrame) -> pd.DataFrame:
    """MediaPipe occasionally swaps its left_wrist/right_wrist landmark
    labels for a frame or two -- most often when the arms cross near the
    face during the catch. bar_x/bar_y (the L+R midpoint) is unaffected
    since swapping two values doesn't change their average, which is why
    this went unnoticed until a chart that reads each wrist individually
    (render_wrist_deviation_chart) showed sharp, suspiciously symmetric
    spikes at the same few frames in an otherwise smooth trajectory.

    Fix: walk forward frame by frame; at each frame, compare "as labelled"
    vs "swapped" distance to a short rolling median of the last few
    ALREADY-CORRECTED frames (not just frame i-1). A median reference
    avoids poisoning every later comparison from a single wrong call, the
    way comparing to i-1 alone would. Only swap when it's a clear win
    (swapped distance <= WRIST_SWAP_MARGIN of as-is distance) -- when the
    two wrists are genuinely close together (hands near each other through
    much of the lift), "swap" and "no swap" are nearly equally plausible
    move-for-move, and forcing a choice there just flip-flops on noise.
    That exact failure mode showed up as 200-380 "corrections" on a few
    videos before this margin was added, versus 1-10 on lifts with an
    isolated genuine swap.
    """
    pdf = pdf.copy()
    left_x = pdf["left_wrist_x"].to_numpy(dtype=float)
    left_y = pdf["left_wrist_y"].to_numpy(dtype=float)
    right_x = pdf["right_wrist_x"].to_numpy(dtype=float)
    right_y = pdf["right_wrist_y"].to_numpy(dtype=float)

    n_swapped = 0
    for i in range(1, len(pdf)):
        if np.isnan(left_x[i]) or np.isnan(right_x[i]):
            continue

        window_start = max(0, i - WRIST_SWAP_REFERENCE_WINDOW)
        ref_left_x = np.nanmedian(left_x[window_start:i])
        ref_left_y = np.nanmedian(left_y[window_start:i])
        ref_right_x = np.nanmedian(right_x[window_start:i])
        ref_right_y = np.nanmedian(right_y[window_start:i])
        if np.isnan(ref_left_x) or np.isnan(ref_right_x):
            continue

        dist_as_is = (
            (left_x[i] - ref_left_x) ** 2 + (left_y[i] - ref_left_y) ** 2
            + (right_x[i] - ref_right_x) ** 2 + (right_y[i] - ref_right_y) ** 2
        )
        dist_swapped = (
            (right_x[i] - ref_left_x) ** 2 + (right_y[i] - ref_left_y) ** 2
            + (left_x[i] - ref_right_x) ** 2 + (left_y[i] - ref_right_y) ** 2
        )

        if dist_as_is > 1e-9 and dist_swapped <= WRIST_SWAP_MARGIN * dist_as_is:
            left_x[i], right_x[i] = right_x[i], left_x[i]
            left_y[i], right_y[i] = right_y[i], left_y[i]
            n_swapped += 1

    if n_swapped:
        pdf["left_wrist_x"] = left_x
        pdf["left_wrist_y"] = left_y
        pdf["right_wrist_x"] = right_x
        pdf["right_wrist_y"] = right_y

    return pdf, n_swapped


def clean_raw_landmarks(raw_df: pd.DataFrame, config: BarPathConfig, video_id: str) -> pd.DataFrame:
    """Clean one video's raw landmark rows into a smoothed, normalised
    bar-path coordinate series covering only the detected lift phase."""
    pdf = raw_df.copy()

    if "lift_phase" in pdf.columns:
        before = len(pdf)
        pdf = pdf[pdf["lift_phase"]].copy()
        logger.info(
            "%s: kept %s/%s lift-phase rows (dropped walk-in/walk-out)",
            video_id, len(pdf), before,
        )

    pdf = pdf[
        (pdf["left_wrist_visibility"] >= config.visibility_threshold)
        & (pdf["right_wrist_visibility"] >= config.visibility_threshold)
    ]

    pdf = pdf.sort_values("frame_id").reset_index(drop=True)
    if pdf.empty:
        logger.warning("No valid frames left after visibility filtering for %s", video_id)
        return pdf

    pdf, n_swapped = fix_wrist_label_swaps(pdf)
    if n_swapped:
        logger.info("%s: corrected %s left/right wrist label swap(s)", video_id, n_swapped)

    pdf[COORDINATE_COLUMNS] = pdf[COORDINATE_COLUMNS].interpolate(
        method="linear", limit_direction="both"
    )
    pdf[COORDINATE_COLUMNS] = pdf[COORDINATE_COLUMNS].ffill().bfill()

    # Outlier removal MUST run before smoothing: a Savitzky-Golay filter
    # smears a genuine bad frame (e.g. a brief wrong-person selection) into
    # its neighbours, turning a sharp jump into a gradual ramp that no
    # longer trips a displacement threshold. A single bad frame next to
    # its immediate neighbour is not enough either -- select_lifter() can
    # occasionally lock onto a bystander for several consecutive frames
    # (e.g. 3 frames straight in one observed case), so comparing only to
    # frame N-1/N+1 misses runs longer than 1. Instead compare each point
    # to the rolling median position over a wider window: as long as bad
    # selections are a minority within that window, the median stays
    # anchored to the real bar position and any point far from it -- lone
    # frame or short run -- gets flagged.
    x = pdf["bar_x"].to_numpy(dtype=float)
    y = pdf["bar_y"].to_numpy(dtype=float)
    window = min(OUTLIER_WINDOW, len(pdf) if len(pdf) % 2 else len(pdf) - 1)
    if window >= 3:
        med_x = pd.Series(x).rolling(window, center=True, min_periods=3).median().to_numpy()
        med_y = pd.Series(y).rolling(window, center=True, min_periods=3).median().to_numpy()
        dist_from_median = np.sqrt((x - med_x) ** 2 + (y - med_y) ** 2)
        outlier_mask = dist_from_median > OUTLIER_DISTANCE_THRESHOLD
    else:
        outlier_mask = np.zeros(len(pdf), dtype=bool)

    if outlier_mask.any():
        logger.info("Detected %s outlier frames in %s", int(outlier_mask.sum()), video_id)
        pdf.loc[outlier_mask, COORDINATE_COLUMNS] = np.nan
        pdf[COORDINATE_COLUMNS] = pdf[COORDINATE_COLUMNS].interpolate(
            method="linear", limit_direction="both"
        )
        pdf[COORDINATE_COLUMNS] = pdf[COORDINATE_COLUMNS].ffill().bfill()

    if len(pdf) >= config.smoothing_window:
        for column in COORDINATE_COLUMNS:
            pdf[column] = savgol_filter(
                pdf[column].to_numpy(dtype=float),
                window_length=config.smoothing_window,
                polyorder=config.smoothing_polyorder,
                mode="interp",
            )

    pdf["x"] = pdf["bar_x"] / pdf["frame_width"]
    pdf["y"] = pdf["bar_y"] / pdf["frame_height"]

    return pdf


def process_all_videos() -> None:
    config = BarPathConfig()
    raw_paths = sorted(config.interim_output_dir.glob("*_raw.csv"))
    logger.info("Found %s raw landmark CSV(s) in %s", len(raw_paths), config.interim_output_dir)

    if not raw_paths:
        logger.warning("No raw landmark CSVs found in %s", config.interim_output_dir)
        return

    failed_videos: List[str] = []
    for raw_path in raw_paths:
        video_id = raw_path.stem.replace("_raw", "")
        try:
            raw_df = pd.read_csv(raw_path)
            if raw_df.empty:
                logger.warning("Raw CSV is empty for %s", video_id)
                continue

            cleaned_df = clean_raw_landmarks(raw_df, config, video_id)
            if cleaned_df.empty:
                logger.warning("Cleaned coordinates are empty for %s", video_id)
                continue

            cleaned_path = config.processed_output_dir / f"{video_id}_cleaned.csv"
            cleaned_df.to_csv(cleaned_path, index=False)
            logger.info("Saved cleaned coordinates for %s to %s", video_id, cleaned_path)

            feature_df = extract_bar_path_features(cleaned_df, video_id)
            frame_feature_path = config.interim_output_dir / f"{video_id}_bar_path.csv"
            save_frame_level_features(feature_df, frame_feature_path)
        except Exception:
            logger.exception("Failed to clean/extract features for %s", raw_path.name)
            failed_videos.append(raw_path.name)

    succeeded = len(raw_paths) - len(failed_videos)
    if failed_videos:
        logger.warning(
            "Processed %s/%s video(s); failed: %s",
            succeeded, len(raw_paths), ", ".join(failed_videos),
        )
    else:
        logger.info("Processed %s/%s video(s) successfully", succeeded, len(raw_paths))

    build_lift_feature_table(config)


def build_lift_feature_table(config: BarPathConfig) -> pd.DataFrame:
    """Build one-row-per-lift bar-path features from the per-video frame CSVs."""
    feature_rows = []
    for csv_path in sorted(config.interim_output_dir.glob("*_bar_path.csv")):
        try:
            frame_df = pd.read_csv(csv_path)
        except Exception:
            logger.exception("Could not read %s", csv_path)
            continue

        if frame_df.empty:
            continue

        video_id = csv_path.stem.replace("_bar_path", "")
        feature_rows.append(summarize_bar_path_features(frame_df, video_id))

    if feature_rows:
        combined = pd.concat(feature_rows, ignore_index=True)
        combined.to_csv(config.features_output_path, index=False)
        logger.info("Saved bar-path lift features to %s", config.features_output_path)
        return combined

    empty_df = pd.DataFrame(
        columns=[
            "video_id", "lift_id", "label", "max_deviation", "avg_deviation",
            "path_smoothness", "peak_vertical_velocity", "time_to_peak_velocity",
            "total_displacement", "jerk_like_movements",
        ]
    )
    empty_df.to_csv(config.features_output_path, index=False)
    return empty_df


if __name__ == "__main__":
    process_all_videos()
