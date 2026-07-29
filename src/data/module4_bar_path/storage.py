from __future__ import annotations

from pathlib import Path

import dask.dataframe as dd
import pandas as pd

from .utils import setup_logger

logger = setup_logger(__name__)


def save_cleaned_coordinates(
    cleaned_df: dd.DataFrame, video_id: str, output_dir: Path
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_id}_cleaned.csv"

    logger.info("Saving cleaned coordinates for %s to %s", video_id, output_path)
    try:
        cleaned_df.to_csv(str(output_path), single_file=True, index=False)
    except Exception as exc:
        logger.exception("Unable to save cleaned coordinates for %s", video_id)
        raise

    return output_path


def save_frame_level_features(features_df: pd.DataFrame, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    features_df.to_csv(output_path, index=False)
    logger.info("Saved frame-level features to %s", output_path)
    return output_path
