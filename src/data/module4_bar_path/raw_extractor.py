"""
Module 4 (bar_path) — Pass 1: raw landmark extraction.

Reads every front-view video in data/raw/videos/front/, runs MediaPipe Tasks
PoseLandmarker (VIDEO mode, num_poses=5), and picks out the actual lifter in
each frame using select_lifter() -- a weighted score of torso centreness,
closeness to camera (bbox height), lowness in frame (hip y), and continuity
with the previous frame's selection. This last signal is what keeps the
selection from jumping to a bystander when the lifter is crouched low (e.g.
mid-deadlift), which the module's older select_center_pose() (centreness
only) does not handle -- see reports/figures/module4/verify/ for the visual
proof this fixes.

Output: one CSV per video in data/interim/module4/<video_id>_raw.csv, with
one row per processed frame (raw pixel-space landmark coordinates plus
visibility -- no cleaning/interpolation/smoothing yet, that is cleaner.py).

Run:
    python -m src.data.module4_bar_path.raw_extractor
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .config import BarPathConfig
from .utils import setup_logger

logger = setup_logger(__name__)

_mp_pose = mp.solutions.pose
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
TORSO_INDICES = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]

FRAME_SKIP = 3
WEIGHT_CENTRENESS = 0.3
WEIGHT_CLOSENESS = 0.2
WEIGHT_LOWNESS = 0.2
WEIGHT_CONTINUITY = 0.3
CONTINUITY_RADIUS_FRACTION = 0.25  # frame-width fraction at which continuity score hits 0

# Lift-phase window detection: a video includes walk-in (approaching the
# bar), the lift itself, and walk-out (re-racking/walking away) -- only the
# middle part is a real lift and should feed feature extraction.
LIFT_BASELINE_FRAMES = 5       # frames used to establish the resting bar_y
LIFT_DEVIATION_FRACTION = 0.08  # bar_y must move this fraction of frame height to count as "lifting"
LIFT_WINDOW_PAD = 3            # extra frames kept on each side of the detected window


def download_model_if_needed(config: BarPathConfig) -> None:
    """Download the shared MediaPipe pose landmarker model if missing."""
    if not config.model_path.exists():
        logger.info("Downloading pose landmarker model to %s", config.model_path)
        config.model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(config.model_url, str(config.model_path))
        logger.info("Model downloaded.")


def _build_landmarker(config: BarPathConfig) -> "vision.PoseLandmarker":
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(config.model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=config.num_poses,
        min_pose_detection_confidence=config.min_detection_confidence,
        min_pose_presence_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    )
    return vision.PoseLandmarker.create_from_options(options)


def _torso_visible(landmarks, visibility_threshold: float) -> bool:
    return all(landmarks[idx].visibility > visibility_threshold for idx in TORSO_INDICES)


def _torso_center(landmarks) -> Tuple[float, float]:
    xs = [landmarks[idx].x for idx in TORSO_INDICES]
    ys = [landmarks[idx].y for idx in TORSO_INDICES]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _bbox_height(landmarks) -> float:
    ys = [lm.y for lm in landmarks]
    return max(ys) - min(ys)


def _hip_y(landmarks) -> float:
    return (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2.0


def select_lifter(
    pose_list,
    w: int,
    h: int,
    prev_center_px: Optional[Tuple[float, float]] = None,
    visibility_threshold: float = 0.5,
):
    """
    Score each detected pose 0-1 and return (best_landmarks, best_score,
    num_candidates).

    Signals (normalised across the detected poses in this frame):
      - CENTRENESS (0.3): torso centre x close to 0.5
      - CLOSENESS  (0.2): pose bounding-box height (bigger = closer)
      - LOWNESS    (0.2): hip y (lower in frame = better)
      - CONTINUITY (0.3): torso centre close to the previous frame's
        selected torso centre (pixel space). Without this, a bystander
        standing tall at the frame edge can outscore a lifter who is
        crouched low at the bar, since a bent-over pose has a smaller
        bounding-box height and briefly loses on CLOSENESS alone.
        Skipped on the first frame (prev_center_px is None).

    num_candidates lets the caller avoid seeding continuity from a frame
    where only one person was detected at all -- e.g. if the lifter is
    briefly undetected (occluded/back turned) and a bystander is the sole
    candidate, that "selection" was never actually contested and should
    not anchor continuity for later frames once the lifter reappears.
    """
    candidates = [
        landmarks
        for landmarks in pose_list
        if landmarks and _torso_visible(landmarks, visibility_threshold)
    ]
    if not candidates:
        return None, 0.0, 0

    centre_x = np.array([_torso_center(lm)[0] for lm in candidates])
    bbox_h = np.array([_bbox_height(lm) for lm in candidates])
    hip_y = np.array([_hip_y(lm) for lm in candidates])

    dist_from_centre = np.abs(centre_x - 0.5)
    max_dist = dist_from_centre.max()
    centreness = (
        1.0 - (dist_from_centre / max_dist) if max_dist > 1e-9 else np.ones_like(dist_from_centre)
    )

    def _normalise(values):
        lo, hi = values.min(), values.max()
        if hi - lo < 1e-9:
            return np.ones_like(values)
        return (values - lo) / (hi - lo)

    closeness = _normalise(bbox_h)
    lowness = _normalise(hip_y)

    weight_centreness, weight_closeness, weight_lowness = (
        WEIGHT_CENTRENESS, WEIGHT_CLOSENESS, WEIGHT_LOWNESS
    )

    if prev_center_px is None:
        continuity = np.zeros(len(candidates))
        weight_continuity = 0.0
        scale = 1.0 / (weight_centreness + weight_closeness + weight_lowness)
        weight_centreness *= scale
        weight_closeness *= scale
        weight_lowness *= scale
    else:
        centre_px = np.array([
            (_torso_center(lm)[0] * w, _torso_center(lm)[1] * h) for lm in candidates
        ])
        dist_px = np.linalg.norm(centre_px - np.array(prev_center_px), axis=1)
        radius = CONTINUITY_RADIUS_FRACTION * w
        continuity = np.clip(1.0 - dist_px / radius, 0.0, 1.0)
        weight_continuity = WEIGHT_CONTINUITY

    scores = (
        weight_centreness * centreness
        + weight_closeness * closeness
        + weight_lowness * lowness
        + weight_continuity * continuity
    )

    best_idx = int(np.argmax(scores))
    return candidates[best_idx], float(scores[best_idx]), len(candidates)


_POINT_NAMES = {
    "left_shoulder": LEFT_SHOULDER,
    "right_shoulder": RIGHT_SHOULDER,
    "left_elbow": LEFT_ELBOW,
    "right_elbow": RIGHT_ELBOW,
    "left_wrist": LEFT_WRIST,
    "right_wrist": RIGHT_WRIST,
    "left_hip": LEFT_HIP,
    "right_hip": RIGHT_HIP,
}


def _empty_row(
    frame_id: int, timestamp_ms: int, frame_width: int, frame_height: int, fps: float
) -> Dict:
    row = {
        "frame_id": frame_id,
        "timestamp_ms": timestamp_ms,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "fps": fps,
        "selection_score": 0.0,
    }
    for name in _POINT_NAMES:
        row[f"{name}_x"] = None
        row[f"{name}_y"] = None
        row[f"{name}_visibility"] = 0.0
    row["bar_x"] = None
    row["bar_y"] = None
    return row


def extract_raw_landmarks(video_path: Path, config: BarPathConfig) -> List[Dict]:
    """Pass 1: read one video and return a list of per-frame raw landmark
    dicts (pixel-space x/y plus visibility for shoulders/elbows/wrists/hips,
    plus the bar position proxy bar_x/bar_y = wrist midpoint). Frames with
    no confidently-selected lifter get an all-None row so frame indexing
    stays contiguous for the cleaner's interpolation step."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_idx = 0
    prev_center_px: Optional[Tuple[float, float]] = None
    rows: List[Dict] = []

    try:
        with _build_landmarker(config) as landmarker:
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

                row = _empty_row(frame_idx, timestamp_ms, w, h, fps)

                if result.pose_landmarks:
                    best_landmarks, best_score, num_candidates = select_lifter(
                        result.pose_landmarks, w, h,
                        prev_center_px=prev_center_px,
                        visibility_threshold=config.visibility_threshold,
                    )
                    if best_landmarks is not None:
                        row["selection_score"] = best_score
                        for name, idx in _POINT_NAMES.items():
                            lm = best_landmarks[idx]
                            row[f"{name}_x"] = lm.x * w
                            row[f"{name}_y"] = lm.y * h
                            row[f"{name}_visibility"] = float(getattr(lm, "visibility", 1.0))

                        left_wrist = best_landmarks[LEFT_WRIST]
                        right_wrist = best_landmarks[RIGHT_WRIST]
                        row["bar_x"] = (left_wrist.x + right_wrist.x) / 2.0 * w
                        row["bar_y"] = (left_wrist.y + right_wrist.y) / 2.0 * h

                        # Only trust this frame's selection to anchor continuity
                        # if it was actually contested by >=2 candidates -- a
                        # lone detection could be a bystander caught while the
                        # real lifter is briefly undetected (occluded/turned
                        # away), and locking onto it would bias every future
                        # frame toward the wrong person via the continuity term.
                        if num_candidates >= 2:
                            cx, cy = _torso_center(best_landmarks)
                            prev_center_px = (cx * w, cy * h)

                rows.append(row)
                frame_idx += 1
    finally:
        cap.release()

    return rows


def find_lift_window(
    df: pd.DataFrame,
    deviation_fraction: float = LIFT_DEVIATION_FRACTION,
    baseline_frames: int = LIFT_BASELINE_FRAMES,
    pad: int = LIFT_WINDOW_PAD,
) -> Tuple[int, int]:
    """Locate the [start, end] row-index range (inclusive) covering the
    actual lift, so walk-in/walk-out frames can be excluded from feature
    extraction.

    Approach: the bar sits near a resting height (on the ground, or held
    static) during walk-in and walk-out, and departs from that height by a
    large margin only while the lift is happening. We take the median
    bar_y over the first few frames as the resting baseline, then keep the
    span from the first to the last frame where bar_y deviates from that
    baseline by more than deviation_fraction * frame_height, padded a few
    frames on each side.

    This does NOT try to bridge pauses within the lift (e.g. the rack
    pause between a clean's catch and the jerk drive) by gap-merging --
    testing showed gap-merging is unreliable because that pause can be
    much longer than incidental noise gaps elsewhere. Instead the window
    is simply the outer envelope of first-to-last deviation from baseline,
    which naturally spans any such pause since bar_y stays elevated
    (away from baseline) throughout it.

    Returns (0, len(df)-1) if bar_y never deviates from baseline (e.g. no
    lift detected, or a lifter selection failure produced flat/empty data).
    """
    bar_y = df["bar_y"].interpolate(limit_direction="both")
    if bar_y.isna().all():
        return 0, len(df) - 1

    frame_height = df["frame_height"].iloc[0]
    baseline = bar_y.iloc[:baseline_frames].median()
    threshold = deviation_fraction * frame_height
    deviation = (bar_y - baseline).abs()
    active = np.where((deviation > threshold).to_numpy())[0]

    if len(active) == 0:
        return 0, len(df) - 1

    start = max(0, int(active[0]) - pad)
    end = min(len(df) - 1, int(active[-1]) + pad)
    return start, end


def process_all_videos() -> None:
    config = BarPathConfig()
    download_model_if_needed(config)

    video_paths = sorted(
        path
        for path in config.raw_video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in config.video_extensions
    )
    logger.info("Found %s front-view video(s) in %s", len(video_paths), config.raw_video_dir)

    if not video_paths:
        logger.warning("No supported video files found in %s", config.raw_video_dir)
        return

    failed_videos: List[str] = []
    for video_path in video_paths:
        video_id = video_path.stem
        try:
            logger.info("Extracting raw landmarks for %s", video_path.name)
            rows = extract_raw_landmarks(video_path, config)
            if not rows:
                logger.warning("No frames processed for %s", video_id)
                continue

            out_df = pd.DataFrame(rows)
            out_df.insert(0, "video_id", video_id)

            start, end = find_lift_window(out_df)
            out_df["lift_phase"] = False
            out_df.loc[start:end, "lift_phase"] = True
            logger.info(
                "%s: lift phase frames %s-%s of %s (%.0f%% kept)",
                video_id,
                out_df["frame_id"].iloc[start],
                out_df["frame_id"].iloc[end],
                len(out_df),
                100 * (end - start + 1) / len(out_df),
            )

            out_path = config.interim_output_dir / f"{video_id}_raw.csv"
            out_df.to_csv(out_path, index=False)
            logger.info("Saved %s raw rows for %s to %s", len(out_df), video_id, out_path)
        except Exception:
            logger.exception("Failed to extract raw landmarks for %s", video_path.name)
            failed_videos.append(video_path.name)

    succeeded = len(video_paths) - len(failed_videos)
    if failed_videos:
        logger.warning(
            "Processed %s/%s video(s); failed: %s",
            succeeded, len(video_paths), ", ".join(failed_videos),
        )
    else:
        logger.info("Processed %s/%s video(s) successfully", succeeded, len(video_paths))


if __name__ == "__main__":
    process_all_videos()
