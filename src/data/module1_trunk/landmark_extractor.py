from __future__ import annotations

from typing import Dict, Optional

import cv2
import mediapipe as mp

from .config import TrunkConfig
from .utils import setup_logger

logger = setup_logger(__name__)
_mp_pose = mp.solutions.pose


def build_pose_detector(config: TrunkConfig) -> "mp.solutions.pose.Pose":
    return _mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    )


def extract_landmarks(
    pose: "mp.solutions.pose.Pose",
    frame: "numpy.ndarray",
    frame_id: int,
    config: TrunkConfig,
) -> Dict[str, Optional[float]]:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)

    output: Dict[str, Optional[float]] = {
        "frame_id": int(frame_id),
        "left_shoulder_x": None,
        "left_shoulder_y": None,
        "left_shoulder_visibility": 0.0,
        "right_shoulder_x": None,
        "right_shoulder_y": None,
        "right_shoulder_visibility": 0.0,
        "left_hip_x": None,
        "left_hip_y": None,
        "left_hip_visibility": 0.0,
        "right_hip_x": None,
        "right_hip_y": None,
        "right_hip_visibility": 0.0,
    }

    if result.pose_landmarks is not None:
        landmarks = result.pose_landmarks.landmark
        ls = landmarks[config.left_shoulder]
        rs = landmarks[config.right_shoulder]
        lh = landmarks[config.left_hip]
        rh = landmarks[config.right_hip]
        output.update(
            {
                "left_shoulder_x": float(ls.x),
                "left_shoulder_y": float(ls.y),
                "left_shoulder_visibility": float(ls.visibility),
                "right_shoulder_x": float(rs.x),
                "right_shoulder_y": float(rs.y),
                "right_shoulder_visibility": float(rs.visibility),
                "left_hip_x": float(lh.x),
                "left_hip_y": float(lh.y),
                "left_hip_visibility": float(lh.visibility),
                "right_hip_x": float(rh.x),
                "right_hip_y": float(rh.y),
                "right_hip_visibility": float(rh.visibility),
            }
        )

    logger.debug("Extracted trunk landmarks for frame %s: %s", frame_id, output)
    return output
