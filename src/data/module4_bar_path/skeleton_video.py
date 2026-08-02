"""
Module 4 (bar_path) — render a skeleton-overlay video for a single lift.

This is a second, separate pass over the video: extract_step() (used by
the rest of the pipeline) only keeps 8 named landmarks as flat CSV
columns and never retains the frame images, so there's nothing to draw
on and no full 33-point skeleton to draw from that stage's output. This
module re-runs MediaPipe (reusing raw_extractor's select_lifter() so the
"who is the lifter" decision is identical to what the rest of the
pipeline uses) and writes an annotated .mp4 with the selected lifter's
skeleton drawn on every frame, trimmed to the same lift-phase window
find_lift_window() would detect -- so what you see in the video matches
what the charts/features are computed from.

Run:
    python -m src.data.module4_bar_path.skeleton_video --video path/to/lift.mp4 --out path/to/out.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .config import BarPathConfig
from .raw_extractor import (
    LIFT_BASELINE_FRAMES,
    LIFT_DEVIATION_FRACTION,
    LIFT_WINDOW_PAD,
    _build_landmarker,
    _torso_center,
    download_model_if_needed,
    select_lifter,
)
from .utils import setup_logger

logger = setup_logger(__name__)

POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
SKELETON_COLOR = (0, 220, 0)   # BGR -- matches verify_lifter_selection.py's "selected lifter" color
SKELETON_THICKNESS = 3
OUTPUT_WIDTH = 960  # smaller than the verify script's 1280 -- this is for in-app playback, not detailed QA


def _draw_skeleton(frame: np.ndarray, landmarks, w: int, h: int) -> None:
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in POSE_CONNECTIONS:
        if a < len(points) and b < len(points):
            cv2.line(frame, points[a], points[b], SKELETON_COLOR, SKELETON_THICKNESS)
    for x, y in points:
        cv2.circle(frame, (x, y), max(2, SKELETON_THICKNESS), SKELETON_COLOR, -1)


def _resize_landscape(frame: np.ndarray, target_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    return cv2.resize(frame, (target_width, int(h * scale)))


def render_skeleton_video(
    video_path: Path,
    out_path: Path,
    config: Optional[BarPathConfig] = None,
    trim_to_lift_phase: bool = True,
) -> Path:
    """Read video_path, draw the selected lifter's skeleton on every
    frame, and write out_path as a new .mp4. Frame selection/skipping
    matches raw_extractor's FRAME_SKIP=3 so the annotated video's motion
    reads the same as what the rest of the pipeline analysed, not an
    unrelated full-framerate pass.

    Lift-phase trimming here re-derives the window from bar_y (same
    approach as find_lift_window) but on the fly, frame by frame, since
    we don't have a pre-computed dataframe at this point in a fresh
    render -- it's a lighter, single-pass version of the same idea:
    everything before the bar first moves meaningfully off its resting
    position is buffered and only flushed to the writer once the lift is
    confirmed to have started, and writing stops once no frame has shown
    lift-level motion for a while.
    """
    config = config or BarPathConfig()
    download_model_if_needed(config)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out_fps = fps / 3  # every 3rd frame is processed (FRAME_SKIP), so playback speed must match

    writer: Optional[cv2.VideoWriter] = None
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    prev_center_px: Optional[Tuple[float, float]] = None
    baseline_y: Optional[float] = None
    baseline_samples: list[float] = []
    frames_written = 0

    try:
        with _build_landmarker(config) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % 3 != 0:
                    frame_idx += 1
                    continue

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((frame_idx / fps) * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                best_landmarks = None
                if result.pose_landmarks:
                    best_landmarks, _, num_candidates = select_lifter(
                        result.pose_landmarks, w, h,
                        prev_center_px=prev_center_px,
                        visibility_threshold=config.visibility_threshold,
                    )
                    if best_landmarks is not None:
                        _draw_skeleton(frame, best_landmarks, w, h)
                        if num_candidates >= 2:
                            cx, cy = _torso_center(best_landmarks)
                            prev_center_px = (cx * w, cy * h)

                frame = _resize_landscape(frame, OUTPUT_WIDTH)

                if trim_to_lift_phase and best_landmarks is not None:
                    from .raw_extractor import LEFT_WRIST, RIGHT_WRIST
                    bar_y_norm = (best_landmarks[LEFT_WRIST].y + best_landmarks[RIGHT_WRIST].y) / 2.0

                    if len(baseline_samples) < LIFT_BASELINE_FRAMES:
                        baseline_samples.append(bar_y_norm)
                        baseline_y = float(np.median(baseline_samples))
                        frame_idx += 1
                        continue

                    is_lifting = abs(bar_y_norm - baseline_y) > LIFT_DEVIATION_FRACTION
                    if not is_lifting and frames_written == 0:
                        frame_idx += 1
                        continue

                if writer is None:
                    # avc1 (H.264), not mp4v -- mp4v produces MPEG-4 Part 2,
                    # which OpenCV can read back fine but no mainstream
                    # browser's <video> tag supports, so st.video() silently
                    # fails to play it. avc1 is what browsers actually decode.
                    writer = cv2.VideoWriter(
                        str(out_path), cv2.VideoWriter_fourcc(*"avc1"), out_fps,
                        (frame.shape[1], frame.shape[0]),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(
                            f"Could not open video writer for {out_path} with avc1 codec "
                            "(browser-incompatible mp4v fallback was intentionally removed)"
                        )
                writer.write(frame)
                frames_written += 1
                frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if frames_written == 0:
        raise RuntimeError(f"No frames written for {video_path} -- no lifter detected")

    logger.info("Wrote %s frames to %s", frames_written, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a skeleton-overlay video for one lift.")
    parser.add_argument("--video", required=True, help="Path to a front-view lift video.")
    parser.add_argument("--out", required=True, help="Path to write the annotated .mp4.")
    parser.add_argument("--no-trim", action="store_true", help="Keep walk-in/walk-out frames.")
    args = parser.parse_args()

    render_skeleton_video(
        Path(args.video), Path(args.out), trim_to_lift_phase=not args.no_trim
    )


if __name__ == "__main__":
    main()
