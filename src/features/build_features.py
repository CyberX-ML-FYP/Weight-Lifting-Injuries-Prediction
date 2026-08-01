"""Build master feature files — cleaned per-frame CSVs -> one row per video/view.

Aggregates ``data/processed/module<N>/<view>/*.csv`` into
``data/features/module<N>/module<N>_features.csv``: one row per video per
view, summarising the dynamic lift-phase portion of the clip (see
``compute_lift_phase_flags`` in ``src/features/module1_trunk.py``) since
that's the part of the video that represents the actual lift, not
walk-in/walkout or idle standing.

Currently only Module 1 is implemented.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Any, Dict, Optional

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")

VALID_VIEWS = ("side", "front", "angle45")

LABEL_PATTERN = re.compile(r"(good|bad)", re.IGNORECASE)


def _label_from_stem(stem: str) -> Optional[int]:
    """0 = good lift, 1 = bad lift, None if the filename doesn't say."""
    match = LABEL_PATTERN.search(stem)
    if not match:
        return None
    return 0 if match.group(1).lower() == "good" else 1


def _aggregate_module1_video(df: pd.DataFrame, video_id: str, view: str) -> Dict[str, Any]:
    df = df.sort_values("frame_index")

    lift_df = df[df["in_lift_phase"]]
    used_full_video_fallback = lift_df.empty
    if used_full_video_fallback:
        lift_df = df

    spine = lift_df["spine_angle"]
    timestamps = lift_df["timestamp_ms"]

    return {
        "video_id": video_id,
        "view": view,
        "label": _label_from_stem(video_id),
        "n_frames_total": len(df),
        "n_lift_phase_frames": int(df["in_lift_phase"].sum()),
        "lift_phase_duration_ms": float(timestamps.max() - timestamps.min()) if len(timestamps) > 1 else 0.0,
        "used_full_video_fallback": used_full_video_fallback,
        "mean_spine_angle": spine.mean(),
        "max_spine_angle": spine.max(),
        "min_spine_angle": spine.min(),
        "std_spine_angle": spine.std(),
        "range_spine_angle": spine.max() - spine.min(),
        "max_abs_spine_angle": spine.abs().max(),
        "mean_lean_deviation": lift_df["lean_deviation"].mean(),
        "max_lean_deviation": lift_df["lean_deviation"].max(),
        "mean_postural_deviation": lift_df["postural_deviation"].mean(),
        "max_postural_deviation": lift_df["postural_deviation"].max(),
        "shoulder_asymmetry_rate": lift_df["shoulder_asymmetry_flag"].mean(),
        "low_visibility_rate": lift_df["low_visibility"].mean(),
    }


def build_module1_master_features() -> pd.DataFrame:
    rows = []
    for view in VALID_VIEWS:
        view_dir = os.path.join(PROCESSED_DIR, "module1", view)
        for path in sorted(glob.glob(os.path.join(view_dir, "*.csv"))):
            video_id = os.path.splitext(os.path.basename(path))[0]
            df = pd.read_csv(path)
            rows.append(_aggregate_module1_video(df, video_id, view))

    master = pd.DataFrame(rows)

    output_dir = os.path.join(FEATURES_DIR, "module1")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "module1_features.csv")
    master.to_csv(output_path, index=False)
    return master


MODULE_BUILDERS = {
    1: build_module1_master_features,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the master feature file for a module")
    parser.add_argument("--module", type=int, default=1, help="Which module to build (default: 1)")
    args = parser.parse_args()

    if args.module not in MODULE_BUILDERS:
        implemented = ", ".join(str(m) for m in sorted(MODULE_BUILDERS))
        raise NotImplementedError(
            f"Module {args.module} has no feature builder yet. Implemented modules: {implemented}"
        )

    master = MODULE_BUILDERS[args.module]()
    print(f"Wrote {len(master)} rows -> data/features/module{args.module}/module{args.module}_features.csv")


if __name__ == "__main__":
    main()
