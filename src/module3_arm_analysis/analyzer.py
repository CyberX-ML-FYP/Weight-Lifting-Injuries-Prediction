"""
Module 3 — Arm / Shoulder / Elbow Analysis
Analyzes upper-limb biomechanics in Clean & Jerk video footage.

Author: Pasindu (214027H)
Faculty of Information Technology, University of Moratuwa
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import urllib.request
import numpy as np
import pandas as pd
import os

from src.module3_arm_analysis.config import (
    MODEL_PATH, MODEL_URL,
    LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST,
    RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    SYMMETRY_THRESHOLD, LOCKOUT_THRESHOLD,
    FRAME_SKIP, MIN_DETECTION_CONF, MIN_TRACKING_CONF,
)

# ── Pose skeleton drawing connections ─────────────────────────────────────────
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


def download_model_if_needed():
    """Download MediaPipe pose landmarker model if not already present."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading pose landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")


def draw_landmarks(frame, landmarks):
    """Draw the 33-point pose skeleton overlay on a frame."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for (a, b) in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2, cv2.LINE_AA)
    for (x, y) in pts:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 4, (255, 255, 255), 1, cv2.LINE_AA)


def select_center_pose(pose_landmarks_list):
    """Select the detected pose whose torso center is closest to frame center."""
    if not pose_landmarks_list:
        return None

    torso_indices = [11, 12, 23, 24]
    best_pose = None
    best_distance = float("inf")

    for landmarks in pose_landmarks_list:
        if not landmarks:
            continue

        x_values = [landmarks[idx].x for idx in torso_indices if idx < len(landmarks)]
        if len(x_values) != len(torso_indices):
            continue

        avg_x = sum(x_values) / len(x_values)
        distance_to_center = abs(avg_x - 0.5)
        if distance_to_center < best_distance:
            best_distance = distance_to_center
            best_pose = landmarks

    return best_pose


def analyze_video(video_path, output_csv_path, show_display=True):
    """
    Process a Clean & Jerk video and produce a CSV of upper-limb features.

    Args:
        video_path:     Input video file (.mp4 / .mov).
        output_csv_path: Where to write the per-frame analysis CSV.
        show_display:   If True, opens a window showing real-time overlay.

    Returns:
        pandas.DataFrame of the extracted features.
    """
    download_model_if_needed()

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=5,
        min_pose_detection_confidence=MIN_DETECTION_CONF,
        min_pose_presence_confidence=MIN_DETECTION_CONF,
        min_tracking_confidence=MIN_TRACKING_CONF,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps       = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_idx = 0
    data      = []

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_SKIP != 0:
                frame_idx += 1
                continue

            h, w = frame.shape[:2]
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((frame_idx / fps) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            frame_idx += 1

            if result.pose_landmarks:
                landmark_list = select_center_pose(result.pose_landmarks)
                if landmark_list is not None:
                    draw_landmarks(frame, landmark_list)

                    def get_xy(idx):
                        lm = landmark_list[idx]
                        return [lm.x * w, lm.y * h]

                    def get_visibility(idx):
                        lm = landmark_list[idx]
                        return float(getattr(lm, "visibility", 1.0))

                    L_shoulder = get_xy(LEFT_SHOULDER)
                    L_elbow    = get_xy(LEFT_ELBOW)
                    L_wrist    = get_xy(LEFT_WRIST)
                    L_hip      = get_xy(LEFT_HIP)
                    R_shoulder = get_xy(RIGHT_SHOULDER)
                    R_elbow    = get_xy(RIGHT_ELBOW)
                    R_wrist    = get_xy(RIGHT_WRIST)
                    R_hip      = get_xy(RIGHT_HIP)
                    L_elbow_vis = get_visibility(LEFT_ELBOW)
                    R_elbow_vis = get_visibility(RIGHT_ELBOW)
                    L_wrist_vis = get_visibility(LEFT_WRIST)
                    R_wrist_vis = get_visibility(RIGHT_WRIST)

                    left_elbow_angle     = calculate_angle(L_shoulder, L_elbow, L_wrist)
                    right_elbow_angle    = calculate_angle(R_shoulder, R_elbow, R_wrist)
                    left_shoulder_angle  = calculate_angle(L_hip, L_shoulder, L_elbow)
                    right_shoulder_angle = calculate_angle(R_hip, R_shoulder, R_elbow)
                    symmetry_diff        = round(abs(left_elbow_angle - right_elbow_angle), 1)
                    low_visibility_flag  = int(
                        min(L_elbow_vis, R_elbow_vis, L_wrist_vis, R_wrist_vis) < 0.5
                    )

                    symmetry_flag = symmetry_diff > SYMMETRY_THRESHOLD
                    lockout_flag  = (left_elbow_angle  < LOCKOUT_THRESHOLD or
                                     right_elbow_angle < LOCKOUT_THRESHOLD)

                    # Wrist y-pixel coordinates (landmarks 15 / 16). In image
                    # space a smaller y = higher position = bar overhead, which
                    # lets us locate the jerk lockout moment downstream.
                    left_wrist_y  = round(L_wrist[1], 1)
                    right_wrist_y = round(R_wrist[1], 1)
                    avg_wrist_y   = round((left_wrist_y + right_wrist_y) / 2, 1)

                    data.append({
                        "frame"                : frame_idx,
                        "left_elbow_angle"     : left_elbow_angle,
                        "right_elbow_angle"    : right_elbow_angle,
                        "left_shoulder_angle"  : left_shoulder_angle,
                        "right_shoulder_angle" : right_shoulder_angle,
                        "symmetry_diff"        : symmetry_diff,
                        "left_wrist_y"         : left_wrist_y,
                        "right_wrist_y"        : right_wrist_y,
                        "avg_wrist_y"          : avg_wrist_y,
                        "asymmetry_flag"       : int(symmetry_flag),
                        "lockout_flag"         : int(lockout_flag),
                        "low_visibility_flag"  : low_visibility_flag,
                    })

                    if show_display:
                        sym_color = (0, 0, 255) if symmetry_flag else (0, 255, 0)
                        cv2.putText(frame, f"L Elbow: {left_elbow_angle}",
                                    (int(L_elbow[0]) - 70, int(L_elbow[1]) - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                        cv2.putText(frame, f"R Elbow: {right_elbow_angle}",
                                    (int(R_elbow[0]) - 70, int(R_elbow[1]) - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                        cv2.putText(frame, f"Symmetry Diff: {symmetry_diff} deg",
                                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, sym_color, 2)
                        if symmetry_flag:
                            cv2.putText(frame, "ASYMMETRY DETECTED",
                                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                        if lockout_flag:
                            cv2.putText(frame, "INCOMPLETE LOCKOUT",
                                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 140, 255), 2)
                        cv2.putText(frame, f"Frame: {frame_idx}",
                                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            if show_display:
                display_height = max(1, int(h * 1280 / w))
                display_frame = cv2.resize(frame, (1280, display_height))
                cv2.imshow("Module 3 — Upper Limb Analysis", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    if show_display:
        cv2.destroyAllWindows()

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    print("\n===== ANALYSIS SUMMARY =====")
    print(f"Total frames analysed  : {len(df)}")
    print(f"Avg Left Elbow Angle   : {df['left_elbow_angle'].mean():.1f} deg")
    print(f"Avg Right Elbow Angle  : {df['right_elbow_angle'].mean():.1f} deg")
    print(f"Avg Symmetry Diff      : {df['symmetry_diff'].mean():.1f} deg")
    print(f"Asymmetry flags        : {df['asymmetry_flag'].sum()} frames")
    print(f"Incomplete lockout     : {df['lockout_flag'].sum()} frames")
    print(f"CSV saved -> {output_csv_path}")

    return df


if __name__ == "__main__":
    # Example usage
    # VIDEO_INPUT = "data/raw_videos/cleanjerk2.mov"
    # CSV_OUTPUT  = "data/processed/arm_analysis_sample.csv"
    VIDEO_INPUT = "data/raw_videos/side/1good.MOV"
    CSV_OUTPUT  = "data/processed/1good_side.csv"
    analyze_video(VIDEO_INPUT, CSV_OUTPUT, show_display=True)
