"""Backfill ``in_lift_phase`` onto Module 1 CSVs already extracted from video.

New extractions (``src/features/module1_trunk.py``) add this column
automatically. This script updates CSVs written before that column existed,
without re-running MediaPipe over the source videos.
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

from src.features.module1_trunk import compute_lift_phase_flags

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INTERIM_MODULE1_DIR = os.path.join(BASE_DIR, "data", "interim", "module1")

VALID_VIEWS = ("side", "front", "angle45")


def backfill_view(view: str) -> None:
    view_dir = os.path.join(INTERIM_MODULE1_DIR, view)
    csv_paths = sorted(glob.glob(os.path.join(view_dir, "*.csv")))

    updated = 0
    for path in csv_paths:
        df = pd.read_csv(path)
        df["in_lift_phase"] = compute_lift_phase_flags(df)
        df.to_csv(path, index=False)
        updated += 1

    print(f"[{view}] updated {updated} CSV(s) in {view_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill in_lift_phase onto existing Module 1 CSVs")
    parser.add_argument("--view", choices=VALID_VIEWS, default=None, help="Which view folder to update (default: all)")
    args = parser.parse_args()

    views = VALID_VIEWS if args.view is None else (args.view,)
    for view in views:
        backfill_view(view)


if __name__ == "__main__":
    main()
