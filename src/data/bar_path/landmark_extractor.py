from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import mediapipe as mp
import numpy as np

from .utils import setup_logger

logger = setup_logger(__name__)
_mp_pose = mp.solutions.pose
_pose = _mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def _extract_landmark_value(
    landmark: mp.framework.formats.landmark_pb2.NormalizedLandmark,
) -> tuple[float, float, float]:
    return float(landmark.x), float(landmark.y), float(landmark.visibility)


def extract_landmarks(
    frame: "numpy.ndarray",
    frame_id: int,
    frame_width: int,
    frame_height: int,
) -> Dict[str, float]:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = _pose.process(rgb_frame)

    left_x = 0.0
    left_y = 0.0
    left_visibility = 0.0
    right_x = 0.0
    right_y = 0.0
    right_visibility = 0.0

    if results.pose_landmarks is not None:
        landmarks = results.pose_landmarks.landmark
        if len(landmarks) > _mp_pose.PoseLandmark.LEFT_WRIST.value:
            left_x, left_y, left_visibility = _extract_landmark_value(
                landmarks[_mp_pose.PoseLandmark.LEFT_WRIST.value]
            )
        if len(landmarks) > _mp_pose.PoseLandmark.RIGHT_WRIST.value:
            right_x, right_y, right_visibility = _extract_landmark_value(
                landmarks[_mp_pose.PoseLandmark.RIGHT_WRIST.value]
            )

    bar_x = float((left_x + right_x) / 2.0)
    bar_y = float((left_y + right_y) / 2.0)

    output = {
        "frame_id": int(frame_id),
        "left_wrist_x": left_x,
        "left_wrist_y": left_y,
        "left_visibility": left_visibility,
        "right_wrist_x": right_x,
        "right_wrist_y": right_y,
        "right_visibility": right_visibility,
        "bar_x": bar_x,
        "bar_y": bar_y,
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
    }

    logger.debug("Extracted landmarks for frame %s: %s", frame_id, output)
    return output
