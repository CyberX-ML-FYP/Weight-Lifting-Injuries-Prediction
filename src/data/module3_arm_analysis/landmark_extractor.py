"""
Module 3 — Landmark Extractor
Pass 1 of the pipeline: reads a video frame-by-frame and extracts the raw
shoulder/elbow/wrist/hip landmark coordinates (plus visibility) needed for
upper-limb analysis. No angle math or cleaning happens here -- that is
cleaner.py (pass 2) and analyzer.py's feature computation (pass 3).
"""
import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from src.data.module3_arm_analysis.config import (
    MODEL_PATH, MODEL_URL,
    LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST,
    RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    FRAME_SKIP, MIN_DETECTION_CONF, MIN_TRACKING_CONF,
)


def download_model_if_needed():
    """Download MediaPipe pose landmarker model if not already present."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading pose landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")


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


_POINT_NAMES = {
    "left_shoulder": LEFT_SHOULDER,
    "left_elbow": LEFT_ELBOW,
    "left_wrist": LEFT_WRIST,
    "left_hip": LEFT_HIP,
    "right_shoulder": RIGHT_SHOULDER,
    "right_elbow": RIGHT_ELBOW,
    "right_wrist": RIGHT_WRIST,
    "right_hip": RIGHT_HIP,
}


def _empty_row(frame_idx, timestamp_ms):
    row = {"frame": frame_idx, "timestamp_ms": timestamp_ms}
    for name in _POINT_NAMES:
        row[f"{name}_x"] = None
        row[f"{name}_y"] = None
        row[f"{name}_visibility"] = 0.0
    return row


def extract_raw_landmarks(video_path):
    """
    Pass 1: read a video and return a list of per-frame landmark dicts
    (pixel-space x/y plus visibility for each tracked point). Frames with
    no detected pose get an all-None/0-visibility row so frame indexing
    stays contiguous for the cleaner's interpolation step.
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

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_idx = 0
    rows = []

    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % FRAME_SKIP != 0:
                    frame_idx += 1
                    continue

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((frame_idx / fps) * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                row = _empty_row(frame_idx, timestamp_ms)
                if result.pose_landmarks:
                    landmark_list = select_center_pose(result.pose_landmarks)
                    if landmark_list is not None:
                        for name, idx in _POINT_NAMES.items():
                            lm = landmark_list[idx]
                            row[f"{name}_x"] = lm.x * w
                            row[f"{name}_y"] = lm.y * h
                            row[f"{name}_visibility"] = float(getattr(lm, "visibility", 1.0))

                rows.append(row)
                frame_idx += 1
    finally:
        cap.release()

    return rows, fps
