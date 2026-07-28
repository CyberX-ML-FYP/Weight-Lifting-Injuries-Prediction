"""
Module 3 — Feature Extractor
Builds one-row-per-lift dataset by merging side/front analysis CSV files.

Instead of looking only at the final phase of the lift, this analyses the
WHOLE lift: every video is split into three equal phase windows
(EARLY 0-33%, MIDDLE 33-66%, FINAL 66-100%), plus whole-lift stability
features and a jerk-moment (lockout) snapshot located from wrist height.

Author: Pasindu (214027H)
"""

import os
import numpy as np
import pandas as pd

from src.module3_arm_analysis.config import DATA_DIR


PHASES = ("early", "middle", "final")


def _extract_label_from_lift_id(lift_id):
    """Map lift id to class label: good -> 0, bad -> 1."""
    name = lift_id.lower()
    if "good" in name:
        return 0
    if "bad" in name:
        return 1
    return None


def _phase_windows(df):
    """Split a frame-level dataframe into 3 equal windows (early/middle/final)."""
    n = len(df)
    i1 = n // 3
    i2 = 2 * n // 3
    return {
        "early": df.iloc[:i1],
        "middle": df.iloc[i1:i2],
        "final": df.iloc[i2:],
    }


def _lockout_index(df):
    """
    Return the positional index of the jerk lockout frame: the frame where
    avg_wrist_y is at its MINIMUM (smallest y = highest position = bar overhead).
    Returns None if the column is missing or empty.
    """
    if "avg_wrist_y" not in df.columns or df.empty:
        return None
    return int(np.argmin(df["avg_wrist_y"].to_numpy()))


def _build_side_features(processed_dir):
    """Build per-lift side-view features (elbow phase windows + lockout snapshot)."""
    rows = []
    for filename in sorted(os.listdir(processed_dir)):
        if not filename.lower().endswith("_side.csv"):
            continue

        csv_path = os.path.join(processed_dir, filename)
        df = pd.read_csv(csv_path).reset_index(drop=True)
        if df.empty:
            continue

        lift_id = filename[:-9].strip()  # remove "_side.csv"
        windows = _phase_windows(df)

        feats = {"lift_id": lift_id}

        # ── Per-phase elbow features ──────────────────────────────────────────
        for phase in PHASES:
            w = windows[phase]
            for side in ("left", "right"):
                col = f"{side}_elbow_angle"
                if w.empty:
                    avg = mx = mn = float("nan")
                else:
                    avg = w[col].mean()
                    mx = w[col].max()
                    mn = w[col].min()
                feats[f"{phase}_avg_{side}_elbow"] = avg
                feats[f"{phase}_max_{side}_elbow"] = mx
                feats[f"{phase}_min_{side}_elbow"] = mn

        # ── Whole-lift stability features ─────────────────────────────────────
        for side in ("left", "right"):
            col = f"{side}_elbow_angle"
            feats[f"std_{side}_elbow"] = df[col].std()
            feats[f"elbow_range_{side}"] = df[col].max() - df[col].min()

        feats["lockout_ratio"] = df["lockout_flag"].mean()

        # ── Jerk-moment (lockout) snapshot ────────────────────────────────────
        lockout_idx = _lockout_index(df)
        if lockout_idx is None:
            feats["lockout_frame_ratio"] = float("nan")
            feats["lockout_left_elbow"] = float("nan")
            feats["lockout_right_elbow"] = float("nan")
        else:
            feats["lockout_frame_ratio"] = lockout_idx / len(df)
            feats["lockout_left_elbow"] = df["left_elbow_angle"].iloc[lockout_idx]
            feats["lockout_right_elbow"] = df["right_elbow_angle"].iloc[lockout_idx]

        rows.append(feats)

    return pd.DataFrame(rows)


def _build_front_features(processed_dir):
    """Build per-lift front-view features (symmetry phase windows + lockout snapshot)."""
    rows = []
    for filename in sorted(os.listdir(processed_dir)):
        if not filename.lower().endswith("_front.csv"):
            continue

        csv_path = os.path.join(processed_dir, filename)
        df = pd.read_csv(csv_path).reset_index(drop=True)
        if df.empty:
            continue

        lift_id = filename[:-10].strip()  # remove "_front.csv"

        # Filter out low-visibility frames when measuring symmetry.
        if "low_visibility_flag" in df.columns:
            valid_mask = df["low_visibility_flag"] == 0
        else:
            valid_mask = pd.Series(True, index=df.index)

        windows = _phase_windows(df)

        feats = {"lift_id": lift_id}

        # ── Per-phase symmetry (front view only) ──────────────────────────────
        for phase in PHASES:
            w = windows[phase]
            w_valid = w[valid_mask.loc[w.index]]
            feats[f"{phase}_avg_symmetry_diff"] = (
                w_valid["symmetry_diff"].mean() if not w_valid.empty else float("nan")
            )

        # ── Whole-lift symmetry features ──────────────────────────────────────
        df_valid = df[valid_mask]
        if df_valid.empty:
            feats["overall_avg_symmetry"] = float("nan")
            feats["overall_max_symmetry"] = float("nan")
            feats["asymmetry_ratio"] = float("nan")
        else:
            feats["overall_avg_symmetry"] = df_valid["symmetry_diff"].mean()
            feats["overall_max_symmetry"] = df_valid["symmetry_diff"].max()
            feats["asymmetry_ratio"] = df_valid["asymmetry_flag"].mean()

        # ── Symmetry at the jerk-moment (lockout) frame ───────────────────────
        lockout_idx = _lockout_index(df)
        if lockout_idx is None:
            feats["lockout_symmetry"] = float("nan")
        else:
            feats["lockout_symmetry"] = df["symmetry_diff"].iloc[lockout_idx]

        rows.append(feats)

    return pd.DataFrame(rows)


def build_master_dataset(output_csv_path=None):
    """
    Build and save merged Module 3 dataset from side + front processed CSVs.

    Returns:
        pandas.DataFrame: One row per lift.
    """
    processed_dir = os.path.join(DATA_DIR, "processed")
    if output_csv_path is None:
        output_csv_path = os.path.join(processed_dir, "module3_master_dataset.csv")

    side_df = _build_side_features(processed_dir)
    front_df = _build_front_features(processed_dir)

    if side_df.empty and front_df.empty:
        raise FileNotFoundError(
            f"No side/front processed CSV files found in {processed_dir}"
        )

    if side_df.empty:
        master_df = front_df
    elif front_df.empty:
        master_df = side_df
    else:
        master_df = pd.merge(side_df, front_df, on="lift_id", how="outer")

    master_df["label"] = master_df["lift_id"].apply(_extract_label_from_lift_id)
    master_df = master_df.sort_values("lift_id").reset_index(drop=True)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    master_df.to_csv(output_csv_path, index=False)

    print("\n===== FEATURE EXTRACTION SUMMARY =====")
    print(f"Total lifts merged    : {len(master_df)}")
    print(f"Total features        : {master_df.shape[1] - 2}")  # minus lift_id, label
    print(f"Output saved          : {output_csv_path}")
    print(f"Good labels (0)       : {(master_df['label'] == 0).sum()}")
    print(f"Bad labels (1)        : {(master_df['label'] == 1).sum()}")

    return master_df


def _print_good_vs_bad(dataset):
    """Print average of each feature for good vs bad lifts."""
    feature_cols = [c for c in dataset.columns if c not in ("lift_id", "label")]
    means = dataset.groupby("label")[feature_cols].mean().T
    means.columns = [{0: "good(0)", 1: "bad(1)"}.get(c, str(c)) for c in means.columns]
    if "good(0)" in means.columns and "bad(1)" in means.columns:
        means["diff(good-bad)"] = means["good(0)"] - means["bad(1)"]

    with pd.option_context(
        "display.max_rows",
        None,
        "display.width",
        120,
        "display.float_format",
        lambda v: f"{v:8.2f}",
    ):
        print("\n===== FEATURE COMPARISON: GOOD vs BAD (averages) =====")
        print(means)


if __name__ == "__main__":
    dataset = build_master_dataset()
    print("\n===== MASTER DATASET =====")
    print(dataset.to_string(index=False))
    _print_good_vs_bad(dataset)
