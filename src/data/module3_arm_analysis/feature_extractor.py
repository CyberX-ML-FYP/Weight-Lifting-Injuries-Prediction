"""
Module 3 — Feature Extractor
Builds one-row-per-lift dataset by merging side/front/angle analysis CSVs.

Analyses the WHOLE lift: every video is split into three equal phase
windows (EARLY 0-33%, MIDDLE 33-66%, FINAL 66-100%), plus whole-lift
stability features and a jerk-moment (lockout) snapshot located from
wrist height.

Side and angle views show elbow flexion (the far arm is often hidden, so
symmetry from these is unreliable). Front view is the only source of
bilateral symmetry, since both arms are visible.

Author: Pasindu (214027H)
"""

import os
import re
import numpy as np
import pandas as pd

from src.data.module3_arm_analysis.config import DATA_DIR


PHASES = ("early", "middle", "final")

# Camera views that show elbow flexion well enough to extract elbow-angle
# features from (side = pure profile, angle = oblique 3/4 view).
ELBOW_VIEWS = ("side", "angle")

_LEADING_NUMBER = re.compile(r"^(\d+)")


def _parse_lift_id_and_label(raw_stem):
    """
    Split a raw filename stem (e.g. '64Good', '16gooda', '9bad') into:
      - lift_id: "<number><good|bad>", used to join side/front/angle views
        of the SAME lift. Built from the number AND the label -- not the
        number alone -- because two different naming conventions are
        mixed in this dataset: an older test batch numbers "good" and
        "bad" lifts with independent counters (its "1bad" and "1good"
        are two unrelated lifts, not two views of "lift 1"), while newer
        uploads use one global counter per physical lift. Combining
        number+label handles both: it keeps "1bad"/"1good" distinct,
        while still merging case/typo variants of the same lift
        ("16good" vs "16gooda", "64Good" vs "64good") since only the
        label *word* differs there, not the label itself.
      - label: 0 (good) / 1 (bad) / None, read from the label text.
    """
    stem = raw_stem.strip()
    match = _LEADING_NUMBER.match(stem)
    number = match.group(1) if match else stem

    lower = stem.lower()
    if "good" in lower:
        label = 0
    elif "bad" in lower:
        label = 1
    else:
        label = None

    label_word = {0: "good", 1: "bad"}.get(label, "unlabeled")
    lift_id = f"{number}{label_word}"

    return lift_id, label


def _collect_labels(processed_dir):
    """Scan every per-frame CSV filename and build a lift_id -> label map."""
    labels = {}
    for filename in sorted(os.listdir(processed_dir)):
        lower = filename.lower()
        for suffix in ("_side.csv", "_front.csv", "_angle.csv"):
            if lower.endswith(suffix):
                raw_stem = filename[: -len(suffix)]
                lift_id, label = _parse_lift_id_and_label(raw_stem)
                if label is not None:
                    labels[lift_id] = label
                break
    return labels


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


def elbow_features_from_df(df, prefix):
    """
    Compute elbow/lockout features from one per-frame dataframe (side or
    angle view). Shared by the batch dataset builder and by predict.py's
    single-lift inference, so both stay consistent with each other.
    """
    df = df.reset_index(drop=True)
    if df.empty:
        return {}

    windows = _phase_windows(df)
    feats = {}

    # ── Per-phase elbow features ────────────────────────────────────────────
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
            feats[f"{prefix}{phase}_avg_{side}_elbow"] = avg
            feats[f"{prefix}{phase}_max_{side}_elbow"] = mx
            feats[f"{prefix}{phase}_min_{side}_elbow"] = mn

    # ── Whole-lift stability features ───────────────────────────────────────
    for side in ("left", "right"):
        col = f"{side}_elbow_angle"
        feats[f"{prefix}std_{side}_elbow"] = df[col].std()
        feats[f"{prefix}elbow_range_{side}"] = df[col].max() - df[col].min()

    feats[f"{prefix}lockout_ratio"] = df["lockout_flag"].mean()

    # ── Jerk-moment (lockout) snapshot ──────────────────────────────────────
    lockout_idx = _lockout_index(df)
    if lockout_idx is None:
        feats[f"{prefix}lockout_frame_ratio"] = float("nan")
        feats[f"{prefix}lockout_left_elbow"] = float("nan")
        feats[f"{prefix}lockout_right_elbow"] = float("nan")
    else:
        feats[f"{prefix}lockout_frame_ratio"] = lockout_idx / len(df)
        feats[f"{prefix}lockout_left_elbow"] = df["left_elbow_angle"].iloc[lockout_idx]
        feats[f"{prefix}lockout_right_elbow"] = df["right_elbow_angle"].iloc[lockout_idx]

    return feats


def front_features_from_df(df):
    """
    Compute symmetry features from one front-view per-frame dataframe.
    Shared by the batch dataset builder and predict.py's single-lift
    inference.
    """
    prefix = "front_"
    df = df.reset_index(drop=True)
    if df.empty:
        return {}

    # Filter out low-visibility frames when measuring symmetry.
    if "low_visibility_flag" in df.columns:
        valid_mask = df["low_visibility_flag"] == 0
    else:
        valid_mask = pd.Series(True, index=df.index)

    windows = _phase_windows(df)
    feats = {}

    # ── Per-phase symmetry (front view only) ────────────────────────────────
    for phase in PHASES:
        w = windows[phase]
        w_valid = w[valid_mask.loc[w.index]]
        feats[f"{prefix}{phase}_avg_symmetry_diff"] = (
            w_valid["symmetry_diff"].mean() if not w_valid.empty else float("nan")
        )

    # ── Whole-lift symmetry features ────────────────────────────────────────
    df_valid = df[valid_mask]
    if df_valid.empty:
        feats[f"{prefix}overall_avg_symmetry"] = float("nan")
        feats[f"{prefix}overall_max_symmetry"] = float("nan")
        feats[f"{prefix}asymmetry_ratio"] = float("nan")
    else:
        feats[f"{prefix}overall_avg_symmetry"] = df_valid["symmetry_diff"].mean()
        feats[f"{prefix}overall_max_symmetry"] = df_valid["symmetry_diff"].max()
        feats[f"{prefix}asymmetry_ratio"] = df_valid["asymmetry_flag"].mean()

    # ── Symmetry at the jerk-moment (lockout) frame ─────────────────────────
    lockout_idx = _lockout_index(df)
    if lockout_idx is None:
        feats[f"{prefix}lockout_symmetry"] = float("nan")
    else:
        feats[f"{prefix}lockout_symmetry"] = df["symmetry_diff"].iloc[lockout_idx]

    return feats


def _build_elbow_features(processed_dir, view):
    """
    Build per-lift elbow/lockout features from *_<view>.csv files
    (side or angle view). Columns are prefixed with the view name so
    side and angle features never collide when merged.
    """
    prefix = f"{view}_"
    suffix = f"_{view}.csv"
    rows = []
    for filename in sorted(os.listdir(processed_dir)):
        if not filename.lower().endswith(suffix):
            continue

        csv_path = os.path.join(processed_dir, filename)
        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        raw_stem = filename[: -len(suffix)]
        lift_id, _ = _parse_lift_id_and_label(raw_stem)

        feats = {"lift_id": lift_id}
        feats.update(elbow_features_from_df(df, prefix))
        rows.append(feats)

    return pd.DataFrame(rows)


def _build_front_features(processed_dir):
    """Build per-lift front-view features (symmetry phase windows + lockout snapshot)."""
    rows = []
    for filename in sorted(os.listdir(processed_dir)):
        if not filename.lower().endswith("_front.csv"):
            continue

        csv_path = os.path.join(processed_dir, filename)
        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        raw_stem = filename[:-len("_front.csv")]
        lift_id, _ = _parse_lift_id_and_label(raw_stem)

        feats = {"lift_id": lift_id}
        feats.update(front_features_from_df(df))
        rows.append(feats)

    return pd.DataFrame(rows)


def build_master_dataset(output_csv_path=None):
    """
    Build and save merged Module 3 dataset from side + front + angle
    processed CSVs. Lifts are joined by their leading lift number, so
    all camera views of the same lift end up on one row.

    Returns:
        pandas.DataFrame: One row per lift.
    """
    processed_dir = os.path.join(DATA_DIR, "processed")
    if output_csv_path is None:
        output_csv_path = os.path.join(processed_dir, "module3_master_dataset.csv")

    view_dfs = [_build_elbow_features(processed_dir, view) for view in ELBOW_VIEWS]
    view_dfs.append(_build_front_features(processed_dir))
    view_dfs = [df for df in view_dfs if not df.empty]

    if not view_dfs:
        raise FileNotFoundError(
            f"No side/front/angle processed CSV files found in {processed_dir}"
        )

    master_df = view_dfs[0]
    for df in view_dfs[1:]:
        master_df = pd.merge(master_df, df, on="lift_id", how="outer")

    labels = _collect_labels(processed_dir)
    master_df["label"] = master_df["lift_id"].map(labels)

    sort_key = master_df["lift_id"].str.extract(r"^(\d+)")[0].astype(int)
    master_df = master_df.loc[sort_key.sort_values().index].reset_index(drop=True)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    master_df.to_csv(output_csv_path, index=False)

    print("\n===== FEATURE EXTRACTION SUMMARY =====")
    print(f"Total lifts merged    : {len(master_df)}")
    print(f"Total features        : {master_df.shape[1] - 2}")  # minus lift_id, label
    print(f"Output saved          : {output_csv_path}")
    print(f"Good labels (0)       : {(master_df['label'] == 0).sum()}")
    print(f"Bad labels (1)        : {(master_df['label'] == 1).sum()}")
    unlabeled = master_df["label"].isna().sum()
    if unlabeled:
        print(f"WARNING: {unlabeled} lift(s) have no 'good'/'bad' in their filename -- unlabeled")

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
