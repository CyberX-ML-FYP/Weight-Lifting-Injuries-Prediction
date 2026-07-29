from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TrunkConfig
from .utils import setup_logger

logger = setup_logger(__name__)


def _extract_lift_label(video_id: str) -> int | None:
    name = str(video_id).lower()
    if "good" in name:
        return 0
    if "bad" in name:
        return 1
    return None


def calculate_angle(A: tuple[float, float], B: tuple[float, float], C: tuple[float, float]) -> float:
    """Angle at vertex B, in degrees, using the vector dot product."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    BA = A - B
    BC = C - B
    denom = np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-6
    cosine = np.clip(np.dot(BA, BC) / denom, -1.0, 1.0)
    return round(float(np.degrees(np.arccos(cosine))), 1)


def classify_lift_phase(spine_angle: float) -> str:
    """Assign a coarse phase label based on the spine angle."""
    if abs(spine_angle) < 10.0:
        return "clean_catch"
    if spine_angle >= 35.0:
        return "first_pull"
    if spine_angle >= 10.0:
        return "jerk_dip"
    return "jerk_drive"


def _row_features(row: pd.Series, config: TrunkConfig) -> dict:
    required = (
        "left_shoulder_x", "left_shoulder_y",
        "right_shoulder_x", "right_shoulder_y",
        "left_hip_x", "left_hip_y",
        "right_hip_x", "right_hip_y",
    )
    if row[list(required)].isna().any():
        return {
            "shoulder_mid_x": np.nan,
            "shoulder_mid_y": np.nan,
            "hip_mid_x": np.nan,
            "hip_mid_y": np.nan,
            "spine_angle": np.nan,
            "lean_deviation": np.nan,
            "postural_deviation": np.nan,
            "shoulder_asymmetry_flag": 0,
            "lift_phase": "unknown",
            "low_visibility": 1,
        }

    shoulder_mid = (
        (row["left_shoulder_x"] + row["right_shoulder_x"]) / 2.0,
        (row["left_shoulder_y"] + row["right_shoulder_y"]) / 2.0,
    )
    hip_mid = (
        (row["left_hip_x"] + row["right_hip_x"]) / 2.0,
        (row["left_hip_y"] + row["right_hip_y"]) / 2.0,
    )

    trunk_vector = (shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1])
    vertical_reference = (0.0, -1.0)
    trunk_norm = np.linalg.norm(trunk_vector) + 1e-6
    vertical_norm = np.linalg.norm(vertical_reference) + 1e-6
    cosine = np.clip(
        np.dot(trunk_vector, vertical_reference) / (trunk_norm * vertical_norm), -1.0, 1.0
    )
    spine_angle = round(float(np.degrees(np.arccos(cosine))), 1)

    # Sign is preserved so forward lean produces a positive angle.
    spine_angle = float(spine_angle) if trunk_vector[1] < 0 else float(-spine_angle)

    shoulder_y_diff = abs(row["left_shoulder_y"] - row["right_shoulder_y"])
    asymmetry_flag = int(shoulder_y_diff > config.shoulder_asym_threshold)

    lean_deviation = round(abs(spine_angle - 45.0), 2)
    postural_deviation = round(
        abs(shoulder_mid[0] - hip_mid[0]) + abs(shoulder_mid[1] - hip_mid[1]), 3
    )

    return {
        "shoulder_mid_x": round(shoulder_mid[0], 4),
        "shoulder_mid_y": round(shoulder_mid[1], 4),
        "hip_mid_x": round(hip_mid[0], 4),
        "hip_mid_y": round(hip_mid[1], 4),
        "spine_angle": spine_angle,
        "lean_deviation": lean_deviation,
        "postural_deviation": postural_deviation,
        "shoulder_asymmetry_flag": asymmetry_flag,
        "lift_phase": classify_lift_phase(spine_angle),
        "low_visibility": 0,
    }


OUTPUT_COLUMNS = [
    "video_id",
    "frame_index",
    "timestamp_ms",
    "shoulder_mid_x",
    "shoulder_mid_y",
    "hip_mid_x",
    "hip_mid_y",
    "spine_angle",
    "lean_deviation",
    "postural_deviation",
    "shoulder_asymmetry_flag",
    "lift_phase",
    "low_visibility",
]


def extract_trunk_features(raw_df: pd.DataFrame, video_id: str, config: TrunkConfig) -> pd.DataFrame:
    """Convert raw per-frame landmark rows into trunk/spine biomechanical features."""
    if raw_df.empty:
        logger.warning("No landmark rows available for %s", video_id)
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = raw_df.sort_values("frame_id").reset_index(drop=True).copy()
    feature_rows = df.apply(lambda row: _row_features(row, config), axis=1, result_type="expand")

    df = pd.concat([df, feature_rows], axis=1)
    df["video_id"] = video_id
    df["frame_index"] = df["frame_id"].astype(int)
    df["timestamp_ms"] = (
        df["frame_index"] * 1000.0 / df["fps"].replace(0, np.nan)
    ).fillna(0.0)

    logger.info("Extracted %s trunk feature rows for %s", len(df), video_id)
    return df[OUTPUT_COLUMNS]


SUMMARY_COLUMNS = [
    "video_id",
    "lift_id",
    "label",
    "avg_spine_angle",
    "max_lean_deviation",
    "avg_postural_deviation",
    "asymmetry_ratio",
    "dominant_phase",
]


def summarize_trunk_features(frame_df: pd.DataFrame, video_id: str) -> pd.DataFrame:
    """Convert frame-level trunk data into one row per lift."""
    if frame_df is None or frame_df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    valid = frame_df[frame_df["low_visibility"] == 0]
    if valid.empty:
        valid = frame_df

    row = {
        "video_id": video_id,
        "lift_id": video_id,
        "label": _extract_lift_label(video_id),
        "avg_spine_angle": float(valid["spine_angle"].mean()) if len(valid) else np.nan,
        "max_lean_deviation": float(valid["lean_deviation"].max()) if len(valid) else np.nan,
        "avg_postural_deviation": float(valid["postural_deviation"].mean()) if len(valid) else np.nan,
        "asymmetry_ratio": float(frame_df["shoulder_asymmetry_flag"].mean()) if len(frame_df) else np.nan,
        "dominant_phase": valid["lift_phase"].mode().iat[0] if len(valid) and not valid["lift_phase"].mode().empty else "unknown",
    }
    return pd.DataFrame([row])
