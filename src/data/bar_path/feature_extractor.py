from __future__ import annotations

import numpy as np
import dask.dataframe as dd

from .utils import compute_perpendicular_distances, setup_logger

logger = setup_logger(__name__)


def extract_bar_path_features(
    cleaned_df: dd.DataFrame, video_id: str
) -> dict[str, float | str]:
    pdf = cleaned_df.compute()
    if pdf.empty:
        logger.warning("No cleaned coordinates available for %s", video_id)
        return {
            "video_id": video_id,
            "total_path_length": 0.0,
            "net_displacement": 0.0,
            "straightness_ratio": 0.0,
            "x_range": 0.0,
            "y_range": 0.0,
            "path_variance": 0.0,
            "average_step_displacement": 0.0,
            "step_standard_deviation": 0.0,
            "max_line_deviation": 0.0,
            "mean_line_deviation": 0.0,
        }

    x = pdf["x"].to_numpy(dtype=float)
    y = pdf["y"].to_numpy(dtype=float)

    displacements = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    total_path_length = float(np.nansum(displacements))
    net_displacement = float(np.linalg.norm([x[-1] - x[0], y[-1] - y[0]]))
    straightness_ratio = (
        float(net_displacement / total_path_length) if total_path_length > 0 else 0.0
    )
    x_range = float(np.nanmax(x) - np.nanmin(x))
    y_range = float(np.nanmax(y) - np.nanmin(y))
    path_variance = float(np.nanvar(np.column_stack([x, y])))
    average_step_displacement = (
        float(np.nanmean(displacements)) if displacements.size > 0 else 0.0
    )
    step_standard_deviation = (
        float(np.nanstd(displacements)) if displacements.size > 0 else 0.0
    )

    start = np.array([x[0], y[0]], dtype=float)
    end = np.array([x[-1], y[-1]], dtype=float)
    line_deviation = compute_perpendicular_distances(x, y, start, end)
    max_line_deviation = float(np.nanmax(line_deviation))
    mean_line_deviation = float(np.nanmean(line_deviation))

    logger.info("Extracted bar path features for %s", video_id)

    return {
        "video_id": video_id,
        "total_path_length": total_path_length,
        "net_displacement": net_displacement,
        "straightness_ratio": straightness_ratio,
        "x_range": x_range,
        "y_range": y_range,
        "path_variance": path_variance,
        "average_step_displacement": average_step_displacement,
        "step_standard_deviation": step_standard_deviation,
        "max_line_deviation": max_line_deviation,
        "mean_line_deviation": mean_line_deviation,
    }
