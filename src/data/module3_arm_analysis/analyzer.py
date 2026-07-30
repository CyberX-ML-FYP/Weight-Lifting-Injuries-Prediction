"""
Module 3 — Arm / Shoulder / Elbow Analysis
Analyzes upper-limb biomechanics in Clean & Jerk video footage.

Three-pass pipeline:
  1. landmark_extractor.extract_raw_landmarks -- raw MediaPipe landmarks per frame
  2. cleaner.clean_landmarks -- visibility filter, interpolation, smoothing
  3. this module -- compute angles/flags from the cleaned coordinates

Author: Pasindu (214027H)
Faculty of Information Technology, University of Moratuwa
"""
import os

import numpy as np
import pandas as pd

from src.data.module3_arm_analysis.cleaner import clean_landmarks, save_cleaned_landmarks
from src.data.module3_arm_analysis.config import (
    PROCESSED_DIR,
    SYMMETRY_THRESHOLD, LOCKOUT_THRESHOLD,
)
from src.data.module3_arm_analysis.landmark_extractor import extract_raw_landmarks

# ── Pose skeleton drawing connections (kept for any future overlay/debug use) ──
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(25,27),(27,29),(27,31),(29,31),
    (24,26),(26,28),(28,30),(28,32),(30,32),
]


def calculate_angle(A, B, C):
    """
    Calculate the angle at vertex B (in degrees) given three points A, B, C.
    Uses the vector dot product formula:
        θ = cos⁻¹( (BA · BC) / (|BA| · |BC|) )
    """
    A, B, C = np.array(A), np.array(B), np.array(C)
    BA = A - B
    BC = C - B
    cosine = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-6)
    return round(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))), 1)


def draw_landmarks(frame, landmarks):
    """Draw the 33-point pose skeleton overlay on a frame."""
    import cv2
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for (a, b) in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2, cv2.LINE_AA)
    for (x, y) in pts:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 4, (255, 255, 255), 1, cv2.LINE_AA)


def _row_features(row):
    L_shoulder = (row["left_shoulder_x"], row["left_shoulder_y"])
    L_elbow = (row["left_elbow_x"], row["left_elbow_y"])
    L_wrist = (row["left_wrist_x"], row["left_wrist_y"])
    L_hip = (row["left_hip_x"], row["left_hip_y"])
    R_shoulder = (row["right_shoulder_x"], row["right_shoulder_y"])
    R_elbow = (row["right_elbow_x"], row["right_elbow_y"])
    R_wrist = (row["right_wrist_x"], row["right_wrist_y"])
    R_hip = (row["right_hip_x"], row["right_hip_y"])

    left_elbow_angle = calculate_angle(L_shoulder, L_elbow, L_wrist)
    right_elbow_angle = calculate_angle(R_shoulder, R_elbow, R_wrist)
    left_shoulder_angle = calculate_angle(L_hip, L_shoulder, L_elbow)
    right_shoulder_angle = calculate_angle(R_hip, R_shoulder, R_elbow)
    symmetry_diff = round(abs(left_elbow_angle - right_elbow_angle), 1)

    symmetry_flag = symmetry_diff > SYMMETRY_THRESHOLD
    lockout_flag = (left_elbow_angle < LOCKOUT_THRESHOLD or
                     right_elbow_angle < LOCKOUT_THRESHOLD)

    # Wrist y-pixel coordinates. In image space a smaller y = higher
    # position = bar overhead, which lets us locate the jerk lockout
    # moment downstream.
    left_wrist_y = round(L_wrist[1], 1)
    right_wrist_y = round(R_wrist[1], 1)
    avg_wrist_y = round((left_wrist_y + right_wrist_y) / 2, 1)

    return {
        "frame": int(row["frame"]),
        "left_elbow_angle": left_elbow_angle,
        "right_elbow_angle": right_elbow_angle,
        "left_shoulder_angle": left_shoulder_angle,
        "right_shoulder_angle": right_shoulder_angle,
        "symmetry_diff": symmetry_diff,
        "left_wrist_y": left_wrist_y,
        "right_wrist_y": right_wrist_y,
        "avg_wrist_y": avg_wrist_y,
        "asymmetry_flag": int(symmetry_flag),
        "lockout_flag": int(lockout_flag),
        "low_visibility_flag": int(row["low_visibility"]),
    }


def analyze_video(video_path, output_csv_path, show_display=False):
    """
    Process a Clean & Jerk video and produce a CSV of upper-limb features.

    Runs the three-pass pipeline: raw landmark extraction, cleaning
    (visibility filter + interpolation + smoothing), then angle/flag
    computation on the cleaned coordinates. Cleaned coordinates are also
    saved to PROCESSED_DIR for inspection/reuse.

    Args:
        video_path:      Input video file (.mp4 / .mov).
        output_csv_path: Where to write the per-frame analysis CSV.
        show_display:    Unused (kept for backward-compatible call sites).
            Live overlay isn't available in the two-pass pipeline since
            angles are only known after the whole video has been cleaned.

    Returns:
        pandas.DataFrame of the extracted features.
    """
    raw_rows, _fps = extract_raw_landmarks(video_path)
    if not raw_rows:
        df = pd.DataFrame(columns=[
            "frame", "left_elbow_angle", "right_elbow_angle",
            "left_shoulder_angle", "right_shoulder_angle", "symmetry_diff",
            "left_wrist_y", "right_wrist_y", "avg_wrist_y",
            "asymmetry_flag", "lockout_flag", "low_visibility_flag",
        ])
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        return df

    cleaned_df = clean_landmarks(raw_rows)

    # Use the output CSV's stem (e.g. "10good_side"), not the source video's
    # stem, so that side/front/angle45 recordings of the same lift number
    # (all literally named "10good.*") don't collide on one cleaned file.
    output_stem = os.path.splitext(os.path.basename(output_csv_path))[0]
    processed_path = os.path.join(PROCESSED_DIR, f"{output_stem}_cleaned.csv")
    save_cleaned_landmarks(cleaned_df, processed_path)

    feature_rows = [_row_features(row) for _, row in cleaned_df.iterrows()]
    df = pd.DataFrame(feature_rows)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    print("\n===== ANALYSIS SUMMARY =====")
    print(f"Total frames analysed  : {len(df)}")
    print(f"Avg Left Elbow Angle   : {df['left_elbow_angle'].mean():.1f} deg")
    print(f"Avg Right Elbow Angle  : {df['right_elbow_angle'].mean():.1f} deg")
    print(f"Avg Symmetry Diff      : {df['symmetry_diff'].mean():.1f} deg")
    print(f"Asymmetry flags        : {df['asymmetry_flag'].sum()} frames")
    print(f"Incomplete lockout     : {df['lockout_flag'].sum()} frames")
    print(f"Cleaned coords saved   -> {processed_path}")
    print(f"CSV saved -> {output_csv_path}")

    return df


if __name__ == "__main__":
    from src.data.module3_arm_analysis.config import RAW_VIDEO_DIR, INTERIM_DIR
    VIDEO_INPUT = os.path.join(RAW_VIDEO_DIR, "side", "1bad.mp4")
    CSV_OUTPUT = os.path.join(INTERIM_DIR, "1bad_side.csv")
    analyze_video(VIDEO_INPUT, CSV_OUTPUT)
