from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

import cv2

from .utils import setup_logger

logger = setup_logger(__name__)


def load_video(
    video_path: Path,
) -> Iterator[Tuple[int, "numpy.ndarray", int, int, float]]:
    video_path = Path(video_path)
    if not video_path.exists():
        logger.error("Video file not found: %s", video_path)
        raise FileNotFoundError(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        logger.error("Unable to open video: %s", video_path)
        raise RuntimeError(f"Unable to open video: {video_path}")

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_id = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame is None:
                logger.warning(
                    "Skipping empty frame %s from %s", frame_id, video_path.name
                )
                frame_id += 1
                continue

            yield frame_id, frame, frame_width, frame_height, fps
            frame_id += 1
    finally:
        capture.release()
        logger.info("Released video capture for %s", video_path.name)
