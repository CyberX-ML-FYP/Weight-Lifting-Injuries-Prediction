from __future__ import annotations

import urllib.request
from typing import Dict, Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .config import TrunkConfig
from .utils import setup_logger

logger = setup_logger(__name__)


def download_model_if_needed(config: TrunkConfig) -> None:
    """Download the shared MediaPipe pose landmarker model if missing."""
    if not config.model_path.exists():
        logger.info("Downloading pose landmarker model to %s", config.model_path)
        config.model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(config.model_url, str(config.model_path))
        logger.info("Model downloaded.")


def build_pose_detector(config: TrunkConfig) -> "vision.PoseLandmarker":
    """Multi-person pose detector -- the gym footage often has bystanders,
    spotters, or other lifters in frame, so num_poses > 1 plus
    select_center_pose() is required to reliably track the actual lifter
    rather than whichever person the model happens to find first.
    """
    download_model_if_needed(config)
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(config.model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=config.num_poses,
        min_pose_detection_confidence=config.min_detection_confidence,
        min_pose_presence_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    )
    return vision.PoseLandmarker.create_from_options(options)


def select_center_pose(pose_landmarks_list, config: TrunkConfig):
    """Select the detected pose whose torso center is closest to frame center.

    The lifter is always positioned in the middle of the recording per the
    project's filming protocol, so this reliably picks them out from
    bystanders/spotters even when several people are visible.
    """
    if not pose_landmarks_list:
        return None

    torso_indices = [config.left_shoulder, config.right_shoulder, config.left_hip, config.right_hip]
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


def extract_landmarks(
    pose: "vision.PoseLandmarker",
    frame: "numpy.ndarray",
    frame_id: int,
    timestamp_ms: int,
    config: TrunkConfig,
) -> Dict[str, Optional[float]]:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = pose.detect_for_video(mp_image, timestamp_ms)

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

    if result.pose_landmarks:
        landmarks = select_center_pose(result.pose_landmarks, config)
        if landmarks is not None:
            ls = landmarks[config.left_shoulder]
            rs = landmarks[config.right_shoulder]
            lh = landmarks[config.left_hip]
            rh = landmarks[config.right_hip]
            output.update(
                {
                    "left_shoulder_x": float(ls.x),
                    "left_shoulder_y": float(ls.y),
                    "left_shoulder_visibility": float(getattr(ls, "visibility", 1.0)),
                    "right_shoulder_x": float(rs.x),
                    "right_shoulder_y": float(rs.y),
                    "right_shoulder_visibility": float(getattr(rs, "visibility", 1.0)),
                    "left_hip_x": float(lh.x),
                    "left_hip_y": float(lh.y),
                    "left_hip_visibility": float(getattr(lh, "visibility", 1.0)),
                    "right_hip_x": float(rh.x),
                    "right_hip_y": float(rh.y),
                    "right_hip_visibility": float(getattr(rh, "visibility", 1.0)),
                }
            )

    logger.debug("Extracted trunk landmarks for frame %s: %s", frame_id, output)
    return output
