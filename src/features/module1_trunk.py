"""Module 1 — Trunk and Spine Analysis.

This module extracts trunk/posture features from MediaPipe pose landmarks for
Clean & Jerk video frames. The implementation follows the project README and
produces one row per processed frame with trunk angle, asymmetry, and phase
labels.
"""

from __future__ import annotations

import argparse
import os
import urllib.request
from typing import Any, Dict, Optional, Sequence

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from scipy.signal import savgol_filter


LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

SHOULDER_ASYM_THRESHOLD = 0.03
FRAME_SKIP = 3
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MAX_POSES_PER_FRAME = 5

# Shared with Module 3 — same landmarker model, downloaded once.
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "module3_arm_analysis",
    "pose_landmarker_full.task",
)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)


def download_model_if_needed() -> None:
    """Download the MediaPipe pose landmarker model if not already present."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def select_lifter_pose(pose_landmarks_list: Any) -> Optional[Any]:
    """Pick whichever detected pose is the lifter, not a bystander.

    A single video can contain several people (spotters, coaches, mirror
    reflections). MediaPipe's legacy single-person tracker can lock onto
    whichever person it detects first and never let go, even once the actual
    lifter is clearly in frame. Detecting up to MAX_POSES_PER_FRAME candidates
    per frame and picking the one whose torso is horizontally closest to
    frame-center (where the lifting platform is framed) avoids that failure
    mode, at the cost of running full detection every frame instead of cheap
    inter-frame tracking.
    """
    if not pose_landmarks_list:
        return None

    torso_indices = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
    best_pose = None
    best_distance = float("inf")

    for landmarks in pose_landmarks_list:
        if not landmarks or max(torso_indices) >= len(landmarks):
            continue
        avg_x = sum(landmarks[idx].x for idx in torso_indices) / len(torso_indices)
        distance_to_center = abs(avg_x - 0.5)
        if distance_to_center < best_distance:
            best_distance = distance_to_center
            best_pose = landmarks

    return best_pose


def _coerce_point(landmarks: Any, idx: int) -> Optional[tuple[float, float]]:
    """Convert a landmark entry into a (x, y) coordinate pair."""
    if landmarks is None:
        return None

    if isinstance(landmarks, dict):
        item = landmarks.get(idx)
        if item is None:
            return None
        if isinstance(item, dict):
            return float(item.get("x", 0.0)), float(item.get("y", 0.0))
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            return float(item[0]), float(item[1])
        return None

    if isinstance(landmarks, (list, tuple)):
        if idx >= len(landmarks):
            return None
        item = landmarks[idx]
        if item is None:
            return None
        if hasattr(item, "x") and hasattr(item, "y"):
            return float(item.x), float(item.y)
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            return float(item[0]), float(item[1])
        return None

    if hasattr(landmarks, "landmark"):
        try:
            item = landmarks.landmark[idx]
        except Exception:
            return None
        if item is None:
            return None
        if hasattr(item, "x") and hasattr(item, "y"):
            return float(item.x), float(item.y)

    if hasattr(landmarks, "__getitem__"):
        try:
            item = landmarks[idx]
        except Exception:
            return None
        if item is None:
            return None
        if hasattr(item, "x") and hasattr(item, "y"):
            return float(item.x), float(item.y)
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            return float(item[0]), float(item[1])

    return None


def calculate_angle(A: Sequence[float], B: Sequence[float], C: Sequence[float]) -> float:
    """Calculate the angle at vertex B in degrees using the dot product."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    BA = A - B
    BC = C - B
    denom = np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-6
    cosine = np.dot(BA, BC) / denom
    cosine = np.clip(cosine, -1.0, 1.0)
    return round(np.degrees(np.arccos(cosine)), 1)


def compute_trunk_features_from_landmarks(landmarks: Any, *, shoulder_asym_threshold: float = SHOULDER_ASYM_THRESHOLD) -> Dict[str, float | int | str]:
    """Compute trunk angle and posture features from a single frame's landmarks."""
    left_shoulder = _coerce_point(landmarks, LEFT_SHOULDER)
    right_shoulder = _coerce_point(landmarks, RIGHT_SHOULDER)
    left_hip = _coerce_point(landmarks, LEFT_HIP)
    right_hip = _coerce_point(landmarks, RIGHT_HIP)

    if None in (left_shoulder, right_shoulder, left_hip, right_hip):
        raise ValueError("Missing required landmarks for trunk analysis")

    shoulder_mid = (
        (left_shoulder[0] + right_shoulder[0]) / 2.0,
        (left_shoulder[1] + right_shoulder[1]) / 2.0,
    )
    hip_mid = ((left_hip[0] + right_hip[0]) / 2.0, (left_hip[1] + right_hip[1]) / 2.0)

    trunk_vector = (shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1])
    vertical_reference = (0.0, -1.0)
    trunk_norm = np.linalg.norm(trunk_vector) + 1e-6
    vertical_norm = np.linalg.norm(vertical_reference) + 1e-6
    cosine = np.dot(trunk_vector, vertical_reference) / (trunk_norm * vertical_norm)
    cosine = np.clip(cosine, -1.0, 1.0)
    spine_angle = round(np.degrees(np.arccos(cosine)), 1)

    # The README defines the trunk vector from hip-mid to shoulder-mid and
    # describes the spine angle as a deviation from vertical. The sign is
    # preserved so forward lean produces a positive angle.
    if trunk_vector[1] < 0:
        spine_angle = float(spine_angle)
    else:
        spine_angle = float(-spine_angle)

    shoulder_y_diff = abs(left_shoulder[1] - right_shoulder[1])
    asymmetry_flag = int(shoulder_y_diff > shoulder_asym_threshold)

    lean_deviation = round(abs(spine_angle - 45.0), 2)
    postural_deviation = round(abs(shoulder_mid[0] - hip_mid[0]) + abs(shoulder_mid[1] - hip_mid[1]), 3)

    return {
        "shoulder_mid_x": round(shoulder_mid[0], 4),
        "shoulder_mid_y": round(shoulder_mid[1], 4),
        "hip_mid_x": round(hip_mid[0], 4),
        "hip_mid_y": round(hip_mid[1], 4),
        "spine_angle": spine_angle,
        "lean_deviation": lean_deviation,
        "postural_deviation": postural_deviation,
        "shoulder_asymmetry_flag": asymmetry_flag,
    }


def _boolean_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end_exclusive) index pairs for each contiguous True run."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(int), [0]))))
    return list(zip(edges[0::2], edges[1::2]))


def _drop_short_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    out = mask.copy()
    for start, end in _boolean_runs(mask):
        if end - start < min_len:
            out[start:end] = False
    return out


def _bridge_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = mask.copy()
    runs = _boolean_runs(mask)
    for (_, prev_end), (next_start, _) in zip(runs, runs[1:]):
        if next_start - prev_end <= max_gap:
            out[prev_end:next_start] = True
    return out


def _dilate_runs(mask: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return mask
    out = mask.copy()
    for start, end in _boolean_runs(mask):
        out[max(0, start - pad):min(len(mask), end + pad)] = True
    return out


def compute_lift_phase_flags(
    df: pd.DataFrame,
    *,
    k_std: float = 1.2,
    min_floor_velocity: float = 0.03,
    smooth_ms: float = 150.0,
    min_burst_ms: float = 200.0,
    merge_gap_ms: float = 2000.0,
    pad_ms: float = 250.0,
) -> pd.Series:
    """Flag frames belonging to the dynamic lift attempt (pull/catch/dip/drive).

    Uses the vertical velocity of the torso midpoint (average of shoulder_mid_y
    and hip_mid_y) as a proxy for explosive lifting motion: standing, walking to
    the platform, and racking the bar between the clean and the jerk all produce
    much smaller and slower vertical movement than the lift itself. The threshold
    is adaptive per video, so a clip that is pre-trimmed to start mid-motion can
    under-flag its opening frames if a later phase in the same clip is even more
    explosive — worth a manual spot-check on short or heavily-trimmed clips.
    """
    n = len(df)
    if n == 0:
        return pd.Series([], dtype=bool, name="in_lift_phase")

    body_y = df[["shoulder_mid_y", "hip_mid_y"]].mean(axis=1)
    body_y = body_y.interpolate(limit_direction="both")

    timestamps = df["timestamp_ms"].astype(float)
    row_dt = timestamps.diff().median()
    row_dt = row_dt if row_dt and row_dt > 0 else 33.0

    dt_sec = timestamps.diff().replace(0, np.nan) / 1000.0
    velocity = (body_y.diff().abs() / dt_sec).fillna(0.0)

    smooth_rows = max(1, int(round(smooth_ms / row_dt)))
    smoothed = velocity.rolling(window=smooth_rows, center=True, min_periods=1).mean()

    threshold = max(min_floor_velocity, smoothed.mean() + k_std * smoothed.std())
    active = (smoothed > threshold).to_numpy()

    active = _drop_short_runs(active, max(1, int(round(min_burst_ms / row_dt))))
    active = _bridge_short_gaps(active, max(1, int(round(merge_gap_ms / row_dt))))
    active = _dilate_runs(active, max(0, int(round(pad_ms / row_dt))))

    return pd.Series(active, index=df.index, name="in_lift_phase")


def classify_lift_phase(spine_angle: float) -> str:
    """Assign a coarse phase label based on the spine angle."""
    if abs(spine_angle) < 10.0:
        return "clean_catch"
    if spine_angle >= 35.0:
        return "first_pull"
    if spine_angle >= 10.0:
        return "jerk_dip"
    return "jerk_drive"


CLEANABLE_FEATURE_COLUMNS = (
    "shoulder_mid_x",
    "shoulder_mid_y",
    "hip_mid_x",
    "hip_mid_y",
    "spine_angle",
    "lean_deviation",
    "postural_deviation",
)

# spine_angle's sign flips whenever the trunk vector's vertical component
# crosses zero, producing artificial jumps of up to ~360 degrees between
# consecutive frames even though the underlying rotation is continuous.
ANGLE_WRAP_COLUMNS = frozenset({"spine_angle"})


def _unwrap_degrees(series: pd.Series) -> pd.Series:
    radians = np.deg2rad(series.to_numpy())
    unwrapped = np.unwrap(radians)
    return pd.Series(np.rad2deg(unwrapped), index=series.index)


def _wrap_degrees(series: pd.Series) -> pd.Series:
    return ((series + 180.0) % 360.0) - 180.0


def _savgol_window_length(n_rows: int, target_ms: float, row_dt_ms: float, polyorder: int) -> Optional[int]:
    """Pick an odd Savitzky-Golay window length near target_ms, or None if too few rows."""
    min_window = polyorder + 1 + (polyorder + 1) % 2  # smallest odd window > polyorder
    if n_rows < min_window:
        return None

    window = max(min_window, int(round(target_ms / row_dt_ms)))
    if window % 2 == 0:
        window += 1

    max_window = n_rows if n_rows % 2 == 1 else n_rows - 1
    return min(window, max_window)


def clean_module1_features(
    df: pd.DataFrame,
    *,
    smooth_ms: float = 200.0,
    polyorder: int = 2,
    outlier_k: float = 3.5,
) -> pd.DataFrame:
    """Clean raw per-frame Module 1 features for downstream modeling.

    Pipeline per continuous feature column: interpolate missing/failed-detection
    values -> Savitzky-Golay smoothing -> remove residual outliers (points where
    the pre-smoothing value deviates far from the smoothed trend, via a robust
    MAD-based threshold) -> interpolate again to fill the gaps left by removed
    outliers. ``lift_phase`` and ``in_lift_phase`` are recomputed from the
    cleaned signal so they stay consistent with it; ``shoulder_asymmetry_flag``
    and ``low_visibility`` are carried over unchanged since they depend on raw
    per-side landmarks not retained in this per-frame table.

    Note: ``spine_angle`` can jump sharply (e.g. -160 -> +74) when the trunk
    vector's vertical component crosses zero, since the sign convention flips
    at that boundary. This pipeline does not unwrap that discontinuity, so a
    handful of frames right at such a transition may smooth oddly — worth a
    spot-check if a video passes through a near-horizontal trunk position.
    """
    df = df.sort_values("frame_index").reset_index(drop=True)
    cleaned = df.copy()

    timestamps = df["timestamp_ms"].astype(float)
    row_dt = timestamps.diff().median()
    row_dt = row_dt if row_dt and row_dt > 0 else 33.0

    n_rows = len(df)
    window = _savgol_window_length(n_rows, smooth_ms, row_dt, polyorder)

    for col in CLEANABLE_FEATURE_COLUMNS:
        series = df[col].astype(float)
        is_angle = col in ANGLE_WRAP_COLUMNS

        interpolated = series.interpolate(limit_direction="both")
        working = _unwrap_degrees(interpolated) if is_angle else interpolated

        if window is None or interpolated.isna().all():
            smoothed = working
        else:
            smoothed = pd.Series(
                savgol_filter(working.to_numpy(), window_length=window, polyorder=polyorder),
                index=working.index,
            )

        residual = working - smoothed
        median_residual = residual.median()
        mad = (residual - median_residual).abs().median()
        robust_scale = 1.4826 * mad if mad > 0 else residual.std()

        if not robust_scale or np.isnan(robust_scale):
            outlier_mask = pd.Series(False, index=series.index)
        else:
            outlier_mask = (residual - median_residual).abs() > outlier_k * robust_scale

        final = smoothed.where(~outlier_mask, np.nan).interpolate(limit_direction="both")
        if is_angle:
            final = _wrap_degrees(final)
        cleaned[col] = final

    cleaned["lift_phase"] = cleaned["spine_angle"].apply(
        lambda angle: classify_lift_phase(angle) if np.isfinite(angle) else "unknown"
    )
    cleaned["in_lift_phase"] = compute_lift_phase_flags(cleaned)

    return cleaned


def extract_frame_features(frame_index: int, timestamp_ms: int, landmarks: Any) -> Dict[str, Any]:
    """Create a one-row feature dictionary for a single frame."""
    try:
        features = compute_trunk_features_from_landmarks(landmarks)
    except ValueError:
        features = {
            "shoulder_mid_x": np.nan,
            "shoulder_mid_y": np.nan,
            "hip_mid_x": np.nan,
            "hip_mid_y": np.nan,
            "spine_angle": np.nan,
            "lean_deviation": np.nan,
            "postural_deviation": np.nan,
            "shoulder_asymmetry_flag": 0,
        }

    features.update(
        {
            "frame_index": frame_index,
            "timestamp_ms": timestamp_ms,
            "lift_phase": classify_lift_phase(float(features["spine_angle"])) if np.isfinite(features["spine_angle"]) else "unknown",
            "low_visibility": 1 if not np.isfinite(features["spine_angle"]) else 0,
        }
    )
    return features


def analyze_video(video_path: str, output_csv_path: str, show_display: bool = False) -> pd.DataFrame:
    """Process a video file and write a Module 1 CSV."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    download_model_if_needed()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_index = 0
    rows: list[dict[str, Any]] = []

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=MAX_POSES_PER_FRAME,
        min_pose_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index % FRAME_SKIP != 0:
                    frame_index += 1
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int((frame_index / fps) * 1000) if fps > 0 else int(frame_index * 1000 / 30)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                landmarks = select_lifter_pose(result.pose_landmarks)
                rows.append(extract_frame_features(frame_index, timestamp_ms, landmarks))

                if show_display:
                    overlay = frame.copy()
                    if landmarks:
                        for landmark in landmarks:
                            x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                            cv2.circle(overlay, (x, y), 3, (0, 255, 0), -1)
                    cv2.putText(overlay, f"Frame {frame_index}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow("Module 1 — Trunk and Spine Analysis", overlay)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_index += 1
    finally:
        cap.release()
        if show_display:
            cv2.destroyAllWindows()

    df = pd.DataFrame(rows)
    df["in_lift_phase"] = compute_lift_phase_flags(df)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Module 1 trunk and spine analysis")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", default="data/interim/module1/trunk_angles.csv", help="Output CSV path")
    args = parser.parse_args()
    analyze_video(args.video, args.output, show_display=False)


if __name__ == "__main__":
    main()
