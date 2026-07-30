from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from .cleaner import clean_coordinates, save_cleaned_coordinates
from .config import TrunkConfig
from .feature_extractor import extract_trunk_features, summarize_trunk_features
from .landmark_extractor import build_pose_detector, extract_landmarks
from .storage import save_frame_level_features
from .utils import setup_logger
from .video_loader import load_video

logger = setup_logger(__name__)


def _collect_landmarks(video_path: Path, config: TrunkConfig) -> List[dict]:
    frames: List[dict] = []
    pose = build_pose_detector(config)

    try:
        for frame_id, frame, _frame_width, _frame_height, fps in load_video(video_path):
            timestamp_ms = int((frame_id / fps) * 1000) if fps else int(frame_id * 1000 / 30)
            landmarks = extract_landmarks(pose, frame, frame_id, timestamp_ms, config)
            landmarks["fps"] = float(fps)
            frames.append(landmarks)
    finally:
        pose.close()

    return frames


def process_video(video_path: Path, config: TrunkConfig) -> None:
    video_id = video_path.stem
    logger.info("Starting trunk processing for video %s", video_id)

    raw_records = _collect_landmarks(video_path, config)
    if not raw_records:
        logger.warning("No landmarks extracted for %s", video_id)
        return

    raw_df = pd.DataFrame(raw_records)
    cleaned_df = clean_coordinates(raw_df, config, video_id)

    if cleaned_df.empty:
        logger.warning("Cleaned coordinates are empty for %s", video_id)
        return

    save_cleaned_coordinates(cleaned_df, video_id, config.processed_output_dir)
    logger.info("Cleaned coordinates saved for %s", video_id)

    feature_df = extract_trunk_features(cleaned_df, video_id, config)

    if feature_df.empty:
        logger.warning("No trunk features produced for %s", video_id)
        return

    output_path = config.interim_output_dir / f"{video_id}_trunk.csv"
    save_frame_level_features(feature_df, output_path)
    logger.info("Frame-level trunk features saved for %s to %s", video_id, output_path)


def build_lift_feature_table(config: TrunkConfig) -> pd.DataFrame:
    """Build one-row-per-lift trunk features from the per-video CSVs."""
    feature_rows = []
    for csv_path in sorted(config.interim_output_dir.glob("*_trunk.csv")):
        try:
            frame_df = pd.read_csv(csv_path)
        except Exception as exc:
            logger.warning("Could not read %s: %s", csv_path, exc)
            continue

        if frame_df.empty:
            continue

        video_id = csv_path.stem.replace("_trunk", "")
        feature_rows.append(summarize_trunk_features(frame_df, video_id))

    if feature_rows:
        combined = pd.concat(feature_rows, ignore_index=True)
        combined.to_csv(config.features_output_path, index=False)
        logger.info("Saved trunk lift features to %s", config.features_output_path)
        return combined

    empty_df = pd.DataFrame(
        columns=[
            "video_id",
            "lift_id",
            "label",
            "avg_spine_angle",
            "max_lean_deviation",
            "avg_postural_deviation",
            "asymmetry_ratio",
            "dominant_phase",
        ]
    )
    empty_df.to_csv(config.features_output_path, index=False)
    return empty_df


def process_all_videos() -> None:
    config = TrunkConfig()
    video_paths = sorted(
        path
        for path in config.raw_video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in config.video_extensions
    )
    logger.info("Found %s video(s) in %s", len(video_paths), config.raw_video_dir)

    if not video_paths:
        logger.warning("No supported video files found in %s", config.raw_video_dir)
        return

    failed_videos: List[str] = []
    for video_path in video_paths:
        try:
            logger.info("Processing side-view video: %s", video_path.name)
            process_video(video_path, config)
        except Exception as exc:
            logger.exception("Failed to process video %s", video_path.name)
            failed_videos.append(video_path.name)

    succeeded = len(video_paths) - len(failed_videos)
    if failed_videos:
        logger.warning(
            "Processed %s/%s video(s); failed: %s",
            succeeded,
            len(video_paths),
            ", ".join(failed_videos),
        )
    else:
        logger.info("Processed %s/%s video(s) successfully", succeeded, len(video_paths))

    build_lift_feature_table(config)


if __name__ == "__main__":
    process_all_videos()
