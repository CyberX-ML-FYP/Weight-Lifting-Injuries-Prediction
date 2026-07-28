from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .config import BarPathConfig
from .storage import save_cleaned_coordinates
from .utils import setup_logger

logger = setup_logger(__name__)

COORDINATE_COLUMNS = ["bar_x", "bar_y"]
VISIBILITY_COLUMNS = ["left_visibility", "right_visibility"]


def clean_coordinates(
    raw_data: Union[dd.DataFrame, List[Dict[str, float]]],
    config: BarPathConfig,
    video_id: str,
) -> dd.DataFrame:
    if isinstance(raw_data, list):
        raw_df = pd.DataFrame(raw_data)
        ddf = dd.from_pandas(raw_df, npartitions=1)
    else:
        ddf = raw_data

    ddf = ddf[
        (ddf.left_visibility >= config.visibility_threshold)
        & (ddf.right_visibility >= config.visibility_threshold)
    ]

    pdf = ddf.compute().sort_values("frame_id").reset_index(drop=True)
    if pdf.empty:
        logger.warning(
            "No valid frames left after visibility filtering for %s", video_id
        )
        return dd.from_pandas(pdf, npartitions=1)

    pdf[COORDINATE_COLUMNS] = pdf[COORDINATE_COLUMNS].interpolate(
        method="linear", limit_direction="both"
    )
    pdf[COORDINATE_COLUMNS] = (
        pdf[COORDINATE_COLUMNS].fillna(method="ffill").fillna(method="bfill")
    )

    if len(pdf) >= config.smoothing_window:
        for column in COORDINATE_COLUMNS:
            pdf[column] = savgol_filter(
                pdf[column].to_numpy(dtype=float),
                window_length=config.smoothing_window,
                polyorder=config.smoothing_polyorder,
                mode="interp",
            )

    displacement = np.sqrt(
        np.diff(pdf["bar_x"], prepend=pdf["bar_x"].iloc[0]) ** 2
        + np.diff(pdf["bar_y"], prepend=pdf["bar_y"].iloc[0]) ** 2
    )
    outlier_mask = displacement > config.outlier_threshold

    if outlier_mask.any():
        logger.info(
            "Detected %s outlier frames in %s",
            int(outlier_mask.sum()),
            video_id,
        )
        pdf.loc[outlier_mask, COORDINATE_COLUMNS] = np.nan
        pdf[COORDINATE_COLUMNS] = pdf[COORDINATE_COLUMNS].interpolate(
            method="linear", limit_direction="both"
        )
        pdf[COORDINATE_COLUMNS] = (
            pdf[COORDINATE_COLUMNS].fillna(method="ffill").fillna(method="bfill")
        )

    pdf["x"] = pdf["bar_x"] / pdf["frame_width"]
    pdf["y"] = pdf["bar_y"] / pdf["frame_height"]

    cleaned_ddf = dd.from_pandas(pdf, npartitions=1)
    save_cleaned_coordinates(cleaned_ddf, video_id, config.interim_output_dir)
    logger.info("Saved cleaned coordinates for %s to interim directory", video_id)
    return cleaned_ddf
