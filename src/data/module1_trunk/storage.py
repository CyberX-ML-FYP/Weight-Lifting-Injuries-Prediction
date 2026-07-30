from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import setup_logger

logger = setup_logger(__name__)


def save_frame_level_features(features_df: pd.DataFrame, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    features_df.to_csv(output_path, index=False)
    logger.info("Saved frame-level trunk features to %s", output_path)
    return output_path
