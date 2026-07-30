"""
Hip & Knee Analysis — Dataset build driver (orchestration script).

Runs the full "raw footage -> processed dataset" stage of the Hip & Knee
pipeline end-to-end against the real videos under ``data/raw/hip_knee/``:

1. Loads the YOLO11-pose model.
2. Discovers every video, grouped by camera view (Side/Angle/Front) and
   quality label (Good/Average/Poor/Unknown), via ``discover_videos_by_view``.
3. For every video: extracts frames (ffmpeg), samples ``NUM_SAMPLED_FRAMES``
   evenly, runs YOLO-pose on each sample, quality-checks/holds-last-good
   (Improvements 1-3), and computes the Rule A-D biomechanics
   (``analyze_lift``) exactly once per video (a single YOLO pass is reused
   for both the LSTM feature sequence and the biomechanics metrics, instead
   of re-running pose estimation twice).
4. Saves, per view and combined across views, fixed-length keypoint
   sequences (``X``) and labels (``y``) as ``.npy`` files for LSTM training
   — Good/Poor videos only, matching ``hip_knee_lstm``'s binary classes.
5. Saves a master feature CSV (Rule A-D metrics for every usable video,
   any label) as ``hip_knee_master_dataset.csv`` — deliberately named to
   avoid any collision with the arm-analysis module's
   ``data/processed/module3_master_dataset.csv``.
6. Learns and saves adaptive Rule A-D thresholds (Improvement 4) from the
   Good/Poor master-CSV rows.
7. Logs every skipped/low-confidence/insufficient-frame video and prints a
   final dataset summary.
8. Verifies every generated file exists and has a sane shape/row count.

This script does NOT train the LSTM classifier — it only builds the
dataset. Run ``src.models.hip_knee_lstm.train_hip_knee_classifier`` as a
separate, later step.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.hip_knee_dataset import discover_videos_by_view, extract_lift_keypoint_frames
from src.features.hip_knee_biomechanics import LiftMetrics, analyze_lift
from src.features.hip_knee_config import (
    CLASS_NAMES,
    INTERIM_DIR,
    MIN_KEYPOINT_CONFIDENCE,
    MODELS_DIR,
    NUM_SAMPLED_FRAMES,
    PROCESSED_DIR,
    RAW_VIDEO_DIR,
    VIEW_DIR_NAMES,
    YOLO_POSE_MODEL,
)
from src.features.hip_knee_logger import get_logger, log_execution_time
from src.features.hip_knee_pose_utils import normalize_by_body_proportion
from src.features.hip_knee_scoring import learn_adaptive_thresholds, save_thresholds

logger = get_logger(__name__)

MASTER_CSV_NAME = "hip_knee_master_dataset.csv"
THRESHOLDS_NAME = "hip_knee_thresholds.json"
MIN_USABLE_FRAMES_FOR_METRICS = 2


def _view_slug(view_name: str) -> str:
    """``"Side view"`` -> ``"side_view"`` (safe for filenames)."""
    return view_name.strip().lower().replace(" ", "_")


@dataclass
class VideoProcessingResult:
    """Outcome of processing a single video through the pose pipeline."""

    view: str
    video_name: str
    label: str
    n_usable_frames: int
    metrics: LiftMetrics | None
    sequence: np.ndarray | None  # shape (NUM_SAMPLED_FRAMES, 16) if usable for the LSTM dataset


def _process_video(
    video_path: Path,
    label: str,
    view: str,
    pose_model: Any,
    frames_root: Path,
) -> VideoProcessingResult:
    """Run frame extraction + pose estimation once and derive both the
    LSTM feature sequence and the Rule A-D biomechanics metrics from it."""
    frames = extract_lift_keypoint_frames(
        video_path, frames_root, pose_model, NUM_SAMPLED_FRAMES, MIN_KEYPOINT_CONFIDENCE,
    )
    n_usable = len(frames)

    metrics = None
    if n_usable >= MIN_USABLE_FRAMES_FOR_METRICS:
        metrics = analyze_lift(frames)
    else:
        logger.warning(
            "%s | %s/%s: only %d/%d usable frames — skipped (no biomechanics metrics)",
            view, label, video_path.name, n_usable, NUM_SAMPLED_FRAMES,
        )

    sequence = None
    if n_usable == NUM_SAMPLED_FRAMES:
        sequence = np.array([normalize_by_body_proportion(f) for f in frames])
    elif label in CLASS_NAMES:
        logger.warning(
            "%s | %s/%s: %d/%d usable frames — excluded from LSTM sequence dataset",
            view, label, video_path.name, n_usable, NUM_SAMPLED_FRAMES,
        )

    return VideoProcessingResult(
        view=view, video_name=video_path.name, label=label,
        n_usable_frames=n_usable, metrics=metrics, sequence=sequence,
    )


def _save_view_sequences(
    view_slug: str,
    sequences: list[np.ndarray],
    labels: list[str],
) -> tuple[Path | None, Path | None]:
    """Save one view's (X, y) LSTM arrays as .npy files. Returns the
    written paths, or (None, None) if there was nothing to save."""
    if not sequences:
        logger.warning("No usable Good/Poor sequences for view=%s — skipping .npy save", view_slug)
        return None, None

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    x_path = PROCESSED_DIR / f"{view_slug}_X.npy"
    y_path = PROCESSED_DIR / f"{view_slug}_y.npy"
    np.save(x_path, np.array(sequences))
    np.save(y_path, np.array(labels))
    logger.info("Saved %s (%d sequences) and %s", x_path, len(sequences), y_path)
    return x_path, y_path


def _build_master_dataframe(rows: list[dict]) -> pd.DataFrame:
    columns = [
        "view", "video_name", "label", "n_usable_frames",
        "delay", "hip_dominance_ratio", "synchronization_corr", "rfd_ratio", "mean_confidence",
    ]
    return pd.DataFrame(rows, columns=columns)


def _metrics_row(result: VideoProcessingResult) -> dict:
    metrics = result.metrics
    assert metrics is not None
    return {
        "view": result.view,
        "video_name": result.video_name,
        "label": result.label,
        "n_usable_frames": result.n_usable_frames,
        "delay": metrics.delay,
        "hip_dominance_ratio": metrics.hip_dominance_ratio,
        "synchronization_corr": metrics.synchronization_corr,
        "rfd_ratio": metrics.rfd_ratio,
        "mean_confidence": metrics.mean_confidence,
    }


@dataclass
class ViewCounters:
    """Running totals for one camera view's processing pass."""

    master_rows: list[dict]
    sequences: list[np.ndarray]
    labels: list[str]
    n_metrics_ok: int = 0
    n_skipped: int = 0
    n_unknown_label: int = 0


def _process_view(
    view: str,
    videos_by_label: dict[str, list[Path]],
    pose_model: Any,
    max_videos_per_group: int | None,
) -> tuple[dict, ViewCounters]:
    """Process every video discovered for one camera view. Returns the
    view's output metadata dict plus its running counters."""
    view_slug = _view_slug(view)
    frames_root = INTERIM_DIR / view_slug
    counters = ViewCounters(master_rows=[], sequences=[], labels=[])

    for label, video_paths in videos_by_label.items():
        if label == "Unknown":
            counters.n_unknown_label += len(video_paths)
            logger.warning("%s | skipping %d video(s) with unresolved 'Unknown' label", view, len(video_paths))
            continue
        if max_videos_per_group is not None:
            video_paths = video_paths[:max_videos_per_group]
        for video_path in video_paths:
            _process_and_record(video_path, label, view, pose_model, frames_root, counters)

    x_path, y_path = _save_view_sequences(view_slug, counters.sequences, counters.labels)
    view_output = {
        "n_videos_discovered": sum(len(v) for v in videos_by_label.values()),
        "n_sequences_built": len(counters.sequences),
        "x_path": str(x_path) if x_path else None,
        "y_path": str(y_path) if y_path else None,
    }
    return view_output, counters


def _process_and_record(
    video_path: Path,
    label: str,
    view: str,
    pose_model: Any,
    frames_root: Path,
    counters: ViewCounters,
) -> None:
    """Process one video and fold its outcome into ``counters`` in place."""
    result = _process_video(video_path, label, view, pose_model, frames_root)

    if result.metrics is not None:
        counters.n_metrics_ok += 1
        counters.master_rows.append(_metrics_row(result))
    else:
        counters.n_skipped += 1

    if result.sequence is not None and label in CLASS_NAMES:
        counters.sequences.append(result.sequence)
        counters.labels.append(label)


def _combine_view_sequences(view_outputs: dict[str, dict]) -> tuple[Path | None, Path | None, Path | None]:
    """Concatenate every view's saved (X, y) .npy arrays into one combined
    dataset. Returns ``(None, None, None)`` if no view produced sequences."""
    all_sequences: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_views: list[str] = []

    for view, outputs in view_outputs.items():
        if outputs["x_path"] is None:
            continue
        view_y = np.load(outputs["y_path"])
        all_sequences.append(np.load(outputs["x_path"]))
        all_labels.append(view_y)
        all_views.extend([view] * len(view_y))

    if not all_sequences:
        return None, None, None

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    combined_x_path = PROCESSED_DIR / "combined_X.npy"
    combined_y_path = PROCESSED_DIR / "combined_y.npy"
    combined_view_path = PROCESSED_DIR / "combined_view.npy"
    np.save(combined_x_path, np.concatenate(all_sequences, axis=0))
    np.save(combined_y_path, np.concatenate(all_labels, axis=0))
    np.save(combined_view_path, np.array(all_views))
    logger.info("Saved combined dataset -> %s (%d total sequences)", combined_x_path, len(all_views))
    return combined_x_path, combined_y_path, combined_view_path


def _learn_and_save_thresholds(master_rows: list[dict]) -> Path | None:
    """Learn adaptive Rule A-D thresholds from Good/Poor master-CSV rows
    and persist them. Returns ``None`` if there isn't enough data."""
    good_metrics = [
        LiftMetrics(r["delay"], r["hip_dominance_ratio"], r["synchronization_corr"], r["rfd_ratio"], r["mean_confidence"])
        for r in master_rows if r["label"] == "Good"
    ]
    poor_metrics = [
        LiftMetrics(r["delay"], r["hip_dominance_ratio"], r["synchronization_corr"], r["rfd_ratio"], r["mean_confidence"])
        for r in master_rows if r["label"] == "Poor"
    ]
    if not good_metrics or not poor_metrics:
        logger.warning(
            "Not enough Good (%d) / Poor (%d) samples to learn adaptive thresholds — skipped",
            len(good_metrics), len(poor_metrics),
        )
        return None

    thresholds = learn_adaptive_thresholds(good_metrics, poor_metrics)
    thresholds_path = MODELS_DIR / THRESHOLDS_NAME
    save_thresholds(thresholds, thresholds_path)
    return thresholds_path


def build_dataset(
    raw_video_dir: Path = RAW_VIDEO_DIR,
    view_dir_names: tuple[str, ...] = VIEW_DIR_NAMES,
    max_videos_per_group: int | None = None,
) -> dict:
    """
    Run the full dataset-build pipeline over every discovered video.

    Args:
        raw_video_dir: Root directory containing per-view video sub-folders.
        view_dir_names: Expected view sub-directory names.
        max_videos_per_group: Optional cap on videos processed per
            (view, label) group — useful for a fast smoke test; ``None``
            processes every discovered video.

    Returns:
        A summary dict (also logged/printed) with per-view/per-label
        counts, sequences built, skipped videos and output file paths.
    """
    with log_execution_time(logger, "Load YOLO11-pose model"):
        from ultralytics import YOLO

        pose_model = YOLO(YOLO_POSE_MODEL)

    videos_by_view = discover_videos_by_view(raw_video_dir, view_dir_names)

    master_rows: list[dict] = []
    view_outputs: dict[str, dict] = {}
    total_discovered = 0
    total_metrics_ok = 0
    total_sequences_ok = 0
    total_skipped_frames = 0
    total_unknown_label = 0

    for view, videos_by_label in videos_by_view.items():
        view_output, counters = _process_view(view, videos_by_label, pose_model, max_videos_per_group)
        view_outputs[view] = view_output
        master_rows.extend(counters.master_rows)
        total_discovered += view_output["n_videos_discovered"]
        total_metrics_ok += counters.n_metrics_ok
        total_sequences_ok += len(counters.sequences)
        total_skipped_frames += counters.n_skipped
        total_unknown_label += counters.n_unknown_label

    combined_x_path, combined_y_path, _combined_view_path = _combine_view_sequences(view_outputs)

    master_df = _build_master_dataframe(master_rows)
    master_csv_path = PROCESSED_DIR / MASTER_CSV_NAME
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(master_csv_path, index=False)
    logger.info("Saved master feature CSV -> %s (%d rows)", master_csv_path, len(master_df))

    thresholds_path = _learn_and_save_thresholds(master_rows)

    return {
        "total_videos_discovered": total_discovered,
        "total_metrics_computed": total_metrics_ok,
        "total_sequences_built": total_sequences_ok,
        "total_skipped_insufficient_frames": total_skipped_frames,
        "total_skipped_unknown_label": total_unknown_label,
        "views": view_outputs,
        "master_csv_path": str(master_csv_path),
        "master_csv_rows": len(master_df),
        "combined_x_path": str(combined_x_path) if combined_x_path else None,
        "combined_y_path": str(combined_y_path) if combined_y_path else None,
        "thresholds_path": str(thresholds_path) if thresholds_path else None,
    }


def _verify_master_csv(summary: dict) -> list[str]:
    master_csv_path = Path(summary["master_csv_path"])
    if not master_csv_path.exists():
        return [f"Master CSV missing: {master_csv_path}"]
    if summary["master_csv_rows"] == 0:
        return [f"Master CSV has 0 rows: {master_csv_path}"]
    return []


def _verify_view_output(view: str, outputs: dict) -> list[str]:
    if outputs["x_path"] is None:
        return []

    x_path, y_path = Path(outputs["x_path"]), Path(outputs["y_path"])
    if not x_path.exists() or not y_path.exists():
        return [f"Missing per-view .npy for {view}: {x_path} / {y_path}"]

    problems: list[str] = []
    x_arr, y_arr = np.load(x_path), np.load(y_path)
    if x_arr.shape[0] != y_arr.shape[0]:
        problems.append(f"{view}: X rows ({x_arr.shape[0]}) != y rows ({y_arr.shape[0]})")
    if x_arr.ndim != 3 or x_arr.shape[1:] != (NUM_SAMPLED_FRAMES, 16):
        problems.append(f"{view}: unexpected X shape {x_arr.shape}")
    return problems


def _verify_combined_output(summary: dict) -> list[str]:
    if not summary["combined_x_path"]:
        return []
    combined_x = np.load(summary["combined_x_path"])
    combined_y = np.load(summary["combined_y_path"])
    if combined_x.shape[0] != combined_y.shape[0]:
        return ["combined_X/combined_y row count mismatch"]
    return []


def _verify_outputs(summary: dict) -> list[str]:
    """Sanity-check every file the build reported as written. Returns a
    list of human-readable problems (empty list means everything checked out)."""
    problems = _verify_master_csv(summary)
    for view, outputs in summary["views"].items():
        problems.extend(_verify_view_output(view, outputs))
    problems.extend(_verify_combined_output(summary))

    if summary["thresholds_path"] and not Path(summary["thresholds_path"]).exists():
        problems.append(f"Thresholds file missing: {summary['thresholds_path']}")

    return problems


def main(max_videos_per_group: int | None = None) -> None:
    """Entry point: build the dataset, print a summary, verify outputs."""
    with log_execution_time(logger, "Hip & Knee dataset build (full pipeline)"):
        summary = build_dataset(max_videos_per_group=max_videos_per_group)

    print("\n=== Hip & Knee Dataset Build Summary ===")
    print(f"Total videos discovered:              {summary['total_videos_discovered']}")
    print(f"Videos with Rule A-D metrics computed: {summary['total_metrics_computed']}")
    print(f"Videos usable for LSTM sequences:      {summary['total_sequences_built']}")
    print(f"Videos skipped (insufficient frames):  {summary['total_skipped_insufficient_frames']}")
    print(f"Videos skipped (unresolved label):     {summary['total_skipped_unknown_label']}")
    print(f"Master CSV: {summary['master_csv_path']} ({summary['master_csv_rows']} rows)")
    for view, outputs in summary["views"].items():
        print(f"  [{view}] discovered={outputs['n_videos_discovered']} sequences={outputs['n_sequences_built']}")
    print(f"Combined X/y: {summary['combined_x_path']} / {summary['combined_y_path']}")
    print(f"Adaptive thresholds: {summary['thresholds_path']}")

    problems = _verify_outputs(summary)
    if problems:
        print("\n⚠ Verification found issues:")
        for problem in problems:
            print(f"  - {problem}")
        logger.error("Dataset build verification found %d issue(s)", len(problems))
    else:
        print("\n✔ All generated files verified successfully.")
        logger.info("Dataset build verification passed")


if __name__ == "__main__":
    main()
