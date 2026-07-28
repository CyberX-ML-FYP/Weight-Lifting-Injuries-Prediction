"""Module 1 — Trunk and Spine Analysis.

This module extracts trunk/posture features from MediaPipe pose landmarks for
Clean & Jerk video frames. The implementation follows the project README and
produces one row per processed frame with trunk angle, asymmetry, and phase
labels.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional, Sequence

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp


LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

SHOULDER_ASYM_THRESHOLD = 0.03
FRAME_SKIP = 3
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5


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


def classify_lift_phase(spine_angle: float) -> str:
    """Assign a coarse phase label based on the spine angle."""
    if abs(spine_angle) < 10.0:
        return "clean_catch"
    if spine_angle >= 35.0:
        return "first_pull"
    if spine_angle >= 10.0:
        return "jerk_dip"
    return "jerk_drive"


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

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_index = 0
    rows: list[dict[str, Any]] = []

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % FRAME_SKIP != 0:
                frame_index += 1
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_frame)
            timestamp_ms = int((frame_index / fps) * 1000) if fps > 0 else int(frame_index * 1000 / 30)
            landmarks = result.pose_landmarks if result.pose_landmarks else None
            rows.append(extract_frame_features(frame_index, timestamp_ms, landmarks))

            if show_display:
                overlay = frame.copy()
                if result.pose_landmarks:
                    for landmark in result.pose_landmarks.landmark:
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
        pose.close()

    df = pd.DataFrame(rows)
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
