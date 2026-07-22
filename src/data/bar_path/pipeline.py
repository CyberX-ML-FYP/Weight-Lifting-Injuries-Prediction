from __future__ import annotations

from pathlib import Path
from typing import List

import dask.dataframe as dd
import pandas as pd

from .cleaner import clean_coordinates
from .config import BarPathConfig
from .feature_extractor import extract_bar_path_features
from .landmark_extractor import extract_landmarks
from .storage import save_cleaned_coordinates, save_features
from .utils import setup_logger
from .video_loader import load_video

logger = setup_logger(__name__)


def _collect_landmarks(video_path: Path) -> List[dict]:
    frames: List[dict] = []

    for frame_id, frame, frame_width, frame_height in load_video(video_path):
        landmarks = extract_landmarks(frame, frame_id, frame_width, frame_height)
        landmarks["video_id"] = video_path.stem
        frames.append(landmarks)

    return frames


def _prepare_dataframe(records: List[dict]) -> dd.DataFrame:
    pandas_df = pd.DataFrame(records)
    return dd.from_pandas(pandas_df, npartitions=1)


def process_video(video_path: Path, config: BarPathConfig) -> None:
    video_id = video_path.stem
    logger.info("Starting processing for video %s", video_id)

    raw_records = _collect_landmarks(video_path)
    if not raw_records:
        logger.warning("No landmarks extracted for %s", video_id)
        return

    raw_df = _prepare_dataframe(raw_records)
    cleaned_df = clean_coordinates(raw_df, config, video_id)

    if cleaned_df.compute().empty:
        logger.warning("Cleaned coordinates are empty for %s", video_id)
        return

    save_cleaned_coordinates(cleaned_df, video_id, config.processed_output_dir)
    logger.info("Cleaned coordinates saved for %s", video_id)

    feature_row = extract_bar_path_features(cleaned_df, video_id)
    save_features(feature_row, config.features_output_path)
    logger.info("Features saved for %s", video_id)


def process_all_videos() -> None:
    config = BarPathConfig()
    video_paths = sorted(config.raw_video_dir.glob("*.mp4"))
    logger.info("Found %s video(s) in %s", len(video_paths), config.raw_video_dir)

    if not video_paths:
        logger.warning("No MP4 files found in %s", config.raw_video_dir)
        return

    for video_path in video_paths:
        try:
            process_video(video_path, config)
        except Exception as exc:
            logger.exception("Failed to process video %s", video_path.name)
            continue


if __name__ == "__main__":
    process_all_videos()
