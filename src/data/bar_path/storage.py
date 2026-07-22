from __future__ import annotations

from pathlib import Path
from typing import Dict

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


def save_features(features: Dict[str, float | str], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_df = pd.DataFrame([features])
    new_ddf = dd.from_pandas(row_df, npartitions=1)

    if output_path.exists():
        existing_ddf = dd.read_csv(str(output_path))
        combined = dd.concat(
            [existing_ddf, new_ddf], axis=0, interleave_partitions=True
        )
    else:
        combined = new_ddf

    temp_path = output_path.with_suffix(".tmp.csv")
    logger.info("Saving features to %s", output_path)
    combined.to_csv(str(temp_path), single_file=True, index=False)

    if temp_path.exists():
        temp_path.replace(output_path)

    return output_path
