"""
Module 3 — Batch Video Processor
Runs arm analysis for side/front videos and exports per-video CSV files.

Author: Pasindu (214027H)
"""
import os

from src.module3_arm_analysis.analyzer import analyze_video
from src.module3_arm_analysis.config import DATA_DIR


def _extract_label(filename):
    """Map filename to class label: good -> 0, bad -> 1."""
    name = filename.lower()
    if "good" in name:
        return 0
    if "bad" in name:
        return 1
    return None


def _process_view(view):
    """Process videos for one view type and print summary."""
    input_dir = os.path.join(DATA_DIR, "raw_videos", view)
    output_dir = os.path.join(DATA_DIR, "processed")
    os.makedirs(output_dir, exist_ok=True)

    valid_exts = (".mp4", ".mov", ".avi", ".mkv")
    videos = sorted(
        filename for filename in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, filename))
        and filename.lower().endswith(valid_exts)
    )

    total_videos = len(videos)
    processed_count = 0
    skipped_count = 0
    good_count = 0
    bad_count = 0

    if total_videos == 0:
        print(f"\n===== {view.upper()} VIEW SUMMARY =====")
        print(f"No videos found in {input_dir}")
        return

    for idx, video_name in enumerate(videos, start=1):
        video_path = os.path.join(input_dir, video_name)
        stem, _ = os.path.splitext(video_name)
        stem = stem.strip()
        output_csv = os.path.join(output_dir, f"{stem}_{view}.csv")

        print(f"[{view}] Processing {idx}/{total_videos}: {video_name}")

        if os.path.exists(output_csv):
            skipped_count += 1
            print(f"  Skipping (already processed): {output_csv}")
            continue

        label = _extract_label(video_name)
        if label == 0:
            good_count += 1
        elif label == 1:
            bad_count += 1

        analyze_video(video_path, output_csv, show_display=False)
        processed_count += 1

    print(f"\n===== {view.upper()} VIEW SUMMARY =====")
    print(f"Total videos found : {total_videos}")
    print(f"Processed now      : {processed_count}")
    print(f"Skipped existing   : {skipped_count}")
    print(f"Good videos        : {good_count}")
    print(f"Bad videos         : {bad_count}")


def run_batch(view=None):
    """
    Run batch processing by view.

    Args:
        view: "side", "front", "angle", or None (process all).
    """
    valid_views = ("side", "front", "angle")
    if view is None:
        views = valid_views
    else:
        normalized_view = view.lower()
        if normalized_view not in valid_views:
            raise ValueError(f"Invalid view '{view}'. Expected one of: {valid_views}")
        views = (normalized_view,)

    for current_view in views:
        _process_view(current_view)


if __name__ == "__main__":
    VIEW = None  # Set to "side" or "front" to process only one view.
    run_batch(view=VIEW)
