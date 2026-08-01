"""Batch runner — videos -> module CSVs.

Reads every video in ``data/raw/videos/<view>/``, runs the requested
analysis module on it, and writes one CSV per video into
``data/interim/module<N>/<view>/``.

Currently only Module 1 (Trunk / Spine, ``src/features/module1_trunk.py``)
is implemented; modules 2-4 will plug in here once their extractors land.
"""

from __future__ import annotations

import argparse
import os

from src.features.module1_trunk import analyze_video

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_VIDEOS_DIR = os.path.join(BASE_DIR, "data", "raw", "videos")
INTERIM_DIR = os.path.join(BASE_DIR, "data", "interim")

VALID_VIEWS = ("side", "front", "angle45")
VALID_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")

MODULE_RUNNERS = {
    1: ("module1", analyze_video),
}


def _process_view(view: str, module: int, *, force: bool, show: bool) -> None:
    module_name, runner = MODULE_RUNNERS[module]
    input_dir = os.path.join(RAW_VIDEOS_DIR, view)
    output_dir = os.path.join(INTERIM_DIR, module_name, view)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(input_dir):
        print(f"\n===== {view.upper()} VIEW SUMMARY =====")
        print(f"No such folder: {input_dir}")
        return

    videos = sorted(
        name for name in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, name))
        and name.lower().endswith(VALID_EXTENSIONS)
    )

    processed_count = 0
    skipped_count = 0

    if not videos:
        print(f"\n===== {view.upper()} VIEW SUMMARY =====")
        print(f"No videos found in {input_dir}")
        return

    for idx, video_name in enumerate(videos, start=1):
        video_path = os.path.join(input_dir, video_name)
        stem, _ = os.path.splitext(video_name)
        output_csv = os.path.join(output_dir, f"{stem}.csv")

        print(f"[{view}] Processing {idx}/{len(videos)}: {video_name}")

        if os.path.exists(output_csv) and not force:
            skipped_count += 1
            print(f"  Skipping (already processed): {output_csv}")
            continue

        runner(video_path, output_csv, show_display=show)
        processed_count += 1

    print(f"\n===== {view.upper()} VIEW SUMMARY =====")
    print(f"Total videos found : {len(videos)}")
    print(f"Processed now      : {processed_count}")
    print(f"Skipped existing   : {skipped_count}")


def run_batch(view: str | None = None, module: int = 1, *, force: bool = False, show: bool = False) -> None:
    if module not in MODULE_RUNNERS:
        implemented = ", ".join(str(m) for m in sorted(MODULE_RUNNERS))
        raise NotImplementedError(
            f"Module {module} has no extractor wired into make_dataset.py yet. "
            f"Implemented modules: {implemented}"
        )

    views = VALID_VIEWS if view is None else (view,)
    for v in views:
        if v not in VALID_VIEWS:
            raise ValueError(f"Invalid view '{v}'. Expected one of: {VALID_VIEWS}")
        _process_view(v, module, force=force, show=show)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-extract module features from raw videos")
    parser.add_argument("--view", choices=VALID_VIEWS, default=None, help="Which camera folder to process (default: all)")
    parser.add_argument("--module", type=int, default=1, help="Which module to run (default: 1)")
    parser.add_argument("--force", action="store_true", help="Re-process videos that already have a CSV")
    parser.add_argument("--show", action="store_true", help="Display the annotated video while processing")
    args = parser.parse_args()

    run_batch(view=args.view, module=args.module, force=args.force, show=args.show)


if __name__ == "__main__":
    main()
