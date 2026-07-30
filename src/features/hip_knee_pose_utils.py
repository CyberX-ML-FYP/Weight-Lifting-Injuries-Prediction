"""
Hip & Knee Analysis — Pose keypoint quality, normalization and smoothing.

WHY these improvements help Hip & Knee analysis
------------------------------------------------
1. Frame quality checking rejects frames where the hip/knee/ankle keypoints
   are missing, low-confidence or the athlete is occluded *before* they can
   corrupt an angle/velocity/acceleration calculation downstream. The
   original notebook silently fed whatever YOLO returned into the angle
   maths, so a single bad detection could blow up a whole rule score.
2. Body-proportion normalization expresses hip/knee positions relative to
   the athlete's own torso length instead of raw pixel coordinates, so the
   same lift filmed closer/further from the camera (or by a taller/shorter
   athlete) produces comparable feature vectors for the LSTM classifier.
3. Temporal smoothing (Savitzky-Golay / Kalman) removes frame-to-frame pose
   jitter from the pose estimator, which otherwise appears as spurious
   velocity/acceleration spikes and destabilises Rule A-D.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from src.features.hip_knee_config import (
    CENTER_DISTANCE_THRESHOLD_PX,
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    MIN_KEYPOINT_CONFIDENCE,
    REQUIRED_KEYPOINTS,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    SAVGOL_POLYORDER,
    SAVGOL_WINDOW_LENGTH,
)
from src.features.hip_knee_logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class BodyKeypoints:
    """Lower-body keypoints extracted from a single frame."""

    left_shoulder: np.ndarray
    right_shoulder: np.ndarray
    left_hip: np.ndarray
    right_hip: np.ndarray
    left_knee: np.ndarray
    right_knee: np.ndarray
    left_ankle: np.ndarray
    right_ankle: np.ndarray
    mean_confidence: float

    def as_feature_vector(self) -> np.ndarray:
        """Flatten all keypoints into a single (16,) feature vector."""
        return np.concatenate([
            self.left_shoulder, self.right_shoulder,
            self.left_hip, self.right_hip,
            self.left_knee, self.right_knee,
            self.left_ankle, self.right_ankle,
        ]).astype(np.float32)


def check_frame_quality(
    keypoints_xy: np.ndarray,
    confidences: np.ndarray | None = None,
    min_confidence: float = MIN_KEYPOINT_CONFIDENCE,
    required_indices: tuple[int, ...] = REQUIRED_KEYPOINTS,
) -> bool:
    """
    Reject frames with missing keypoints, low confidence or occlusion.

    Args:
        keypoints_xy: Array of shape ``(num_keypoints, 2)`` from YOLO-pose.
        confidences: Optional per-keypoint confidence array of shape
            ``(num_keypoints,)``. If ``None``, confidence is not checked
            (only keypoint presence).
        min_confidence: Minimum acceptable mean confidence for the required
            hip/knee/ankle/shoulder keypoints.
        required_indices: Keypoint indices that must be present and
            confident for the frame to be usable.

    Returns:
        ``True`` if the frame passes the quality check, ``False`` otherwise.
    """
    if keypoints_xy is None or len(keypoints_xy) == 0:
        return False

    max_index = max(required_indices)
    if keypoints_xy.shape[0] <= max_index:
        return False

    required_points = keypoints_xy[list(required_indices)]
    # A missing YOLO keypoint is reported as (0, 0) — treat as occluded.
    if np.any(np.all(required_points == 0, axis=1)):
        return False

    if confidences is not None:
        if confidences.shape[0] <= max_index:
            return False
        required_conf = confidences[list(required_indices)]
        if required_conf.mean() < min_confidence:
            return False

    return True


def extract_body_keypoints(
    keypoints_xy: np.ndarray,
    confidences: np.ndarray | None = None,
    min_confidence: float = MIN_KEYPOINT_CONFIDENCE,
) -> BodyKeypoints | None:
    """
    Extract lower-body keypoints from a raw YOLO-pose detection.

    Returns ``None`` (instead of raising) when the frame fails the quality
    check, so callers can simply skip/interpolate that frame.
    """
    if not check_frame_quality(keypoints_xy, confidences, min_confidence):
        return None

    mean_conf = (
        float(confidences[list(REQUIRED_KEYPOINTS)].mean())
        if confidences is not None
        else 1.0
    )

    return BodyKeypoints(
        left_shoulder=keypoints_xy[LEFT_SHOULDER],
        right_shoulder=keypoints_xy[RIGHT_SHOULDER],
        left_hip=keypoints_xy[LEFT_HIP],
        right_hip=keypoints_xy[RIGHT_HIP],
        left_knee=keypoints_xy[LEFT_KNEE],
        right_knee=keypoints_xy[RIGHT_KNEE],
        left_ankle=keypoints_xy[LEFT_ANKLE],
        right_ankle=keypoints_xy[RIGHT_ANKLE],
        mean_confidence=mean_conf,
    )


def normalize_by_body_proportion(points: BodyKeypoints) -> np.ndarray:
    """
    Normalize keypoints using anthropometric (body-proportion) ratios.

    Translates keypoints so the hip midpoint is the origin, then scales by
    the athlete's torso length (shoulder-midpoint to hip-midpoint distance)
    instead of using raw pixel coordinates. This makes features comparable
    across athletes of different heights and camera distances.

    Args:
        points: Extracted lower-body keypoints for a single frame.

    Returns:
        A (16,) normalized feature vector (same layout as
        ``BodyKeypoints.as_feature_vector``).
    """
    hip_mid = (points.left_hip + points.right_hip) / 2.0
    shoulder_mid = (points.left_shoulder + points.right_shoulder) / 2.0
    torso_length = float(np.linalg.norm(shoulder_mid - hip_mid))
    scale = torso_length if torso_length > 1e-6 else 1.0

    raw = points.as_feature_vector().reshape(-1, 2)
    normalized = (raw - hip_mid) / scale
    return normalized.reshape(-1).astype(np.float32)


def select_person_by_center(
    all_keypoints_xy: list[np.ndarray],
    frame_shape: tuple[int, int],
    distance_threshold: float = CENTER_DISTANCE_THRESHOLD_PX,
) -> int | None:
    """
    Select which detected person is closest to the frame center.

    Mirrors the original notebook's ``get_centers``/``check_image`` logic,
    but returns ``None`` (unambiguous rejection) instead of a boolean flag
    when more than one person is within the threshold, so the caller can
    unambiguously skip an unusable frame.

    Args:
        all_keypoints_xy: List of per-person keypoint arrays for one frame.
        frame_shape: ``(height, width)`` of the source frame.
        distance_threshold: Max pixel distance from frame center to be
            considered "the athlete".

    Returns:
        Index of the selected person, or ``None`` if zero or more than one
        candidate is within the threshold (ambiguous frame).
    """
    height, width = frame_shape
    frame_center = np.array([width / 2.0, height / 2.0])

    candidates: list[int] = []
    for person_id, keypoints in enumerate(all_keypoints_xy):
        if keypoints.shape[0] <= max(LEFT_HIP, RIGHT_HIP):
            continue
        hip_center = (keypoints[LEFT_HIP] + keypoints[RIGHT_HIP]) / 2.0
        distance = float(np.linalg.norm(hip_center - frame_center))
        if distance <= distance_threshold:
            candidates.append(person_id)

    if len(candidates) != 1:
        return None
    return candidates[0]


def smooth_angle_sequence(
    signal: np.ndarray,
    window_length: int = SAVGOL_WINDOW_LENGTH,
    polyorder: int = SAVGOL_POLYORDER,
) -> np.ndarray:
    """
    Smooth a 1-D angle/time-series signal with a Savitzky-Golay filter.

    Falls back to returning the signal unchanged when it is too short for
    the requested window (rather than raising), since short clips are
    common near the start/end of a lift.

    Args:
        signal: 1-D array of angle values over time.
        window_length: Odd window size for the filter.
        polyorder: Polynomial order fit within each window.

    Returns:
        Smoothed signal with the same shape as the input.
    """
    n = len(signal)
    if n < window_length:
        return signal

    window = window_length if window_length % 2 == 1 else window_length - 1
    window = min(window, n if n % 2 == 1 else n - 1)
    if window <= polyorder:
        return signal

    return savgol_filter(signal, window_length=window, polyorder=polyorder)


class KalmanAngleFilter:
    """
    Minimal 1-D Kalman filter for smoothing a single joint-angle time series.

    An alternative to Savitzky-Golay smoothing that can be applied online
    (frame-by-frame) rather than requiring the whole sequence up front,
    which is useful for future real-time inference.
    """

    def __init__(self, process_variance: float = 1e-3, measurement_variance: float = 1e-1) -> None:
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self._estimate: float | None = None
        self._error_estimate: float = 1.0

    def update(self, measurement: float) -> float:
        """Feed one new measurement and return the filtered estimate."""
        if self._estimate is None:
            self._estimate = measurement
            return self._estimate

        # Prediction step.
        prediction_error = self._error_estimate + self.process_variance

        # Update (correction) step.
        kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self._estimate = self._estimate + kalman_gain * (measurement - self._estimate)
        self._error_estimate = (1 - kalman_gain) * prediction_error
        return self._estimate

    def filter(self, sequence: np.ndarray) -> np.ndarray:
        """Filter an entire pre-recorded sequence in one call."""
        return np.array([self.update(float(value)) for value in sequence])
