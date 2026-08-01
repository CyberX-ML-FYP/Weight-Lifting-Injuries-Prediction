"""Batch runner — interim module CSVs -> cleaned processed CSVs.

Reads every per-video CSV in ``data/interim/module<N>/<view>/``, applies the
module's cleaning pipeline (interpolate -> smooth -> remove outliers ->
interpolate), and writes the result to ``data/processed/module<N>/<view>/``.

Currently only Module 1 is implemented.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from src.features.module1_trunk import clean_module1_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INTERIM_DIR = os.path.join(BASE_DIR, "data", "interim")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

VALID_VIEWS = ("side", "front", "angle45")

MODULE_CLEANERS = {
    1: ("module1", clean_module1_features),
}


def _clean_view(view: str, module: int, *, force: bool) -> None:
    module_name, cleaner = MODULE_CLEANERS[module]
    input_dir = os.path.join(INTERIM_DIR, module_name, view)
    output_dir = os.path.join(PROCESSED_DIR, module_name, view)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(input_dir):
        print(f"\n===== {view.upper()} VIEW SUMMARY =====")
        print(f"No such folder: {input_dir}")
        return

    csv_names = sorted(name for name in os.listdir(input_dir) if name.lower().endswith(".csv"))

    processed_count = 0
    skipped_count = 0

    if not csv_names:
        print(f"\n===== {view.upper()} VIEW SUMMARY =====")
        print(f"No CSVs found in {input_dir}")
        return

    for idx, csv_name in enumerate(csv_names, start=1):
        input_path = os.path.join(input_dir, csv_name)
        output_path = os.path.join(output_dir, csv_name)

        print(f"[{view}] Cleaning {idx}/{len(csv_names)}: {csv_name}")

        if os.path.exists(output_path) and not force:
            skipped_count += 1
            print(f"  Skipping (already cleaned): {output_path}")
            continue

        df = pd.read_csv(input_path)
        cleaned = cleaner(df)
        cleaned.to_csv(output_path, index=False)
        processed_count += 1

    print(f"\n===== {view.upper()} VIEW SUMMARY =====")
    print(f"Total CSVs found   : {len(csv_names)}")
    print(f"Cleaned now        : {processed_count}")
    print(f"Skipped existing   : {skipped_count}")


def run_batch(view: str | None = None, module: int = 1, *, force: bool = False) -> None:
    if module not in MODULE_CLEANERS:
        implemented = ", ".join(str(m) for m in sorted(MODULE_CLEANERS))
        raise NotImplementedError(
            f"Module {module} has no cleaner wired into clean_dataset.py yet. "
            f"Implemented modules: {implemented}"
        )

    views = VALID_VIEWS if view is None else (view,)
    for v in views:
        if v not in VALID_VIEWS:
            raise ValueError(f"Invalid view '{v}'. Expected one of: {VALID_VIEWS}")
        _clean_view(v, module, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean interim module features into processed CSVs")
    parser.add_argument("--view", choices=VALID_VIEWS, default=None, help="Which camera folder to process (default: all)")
    parser.add_argument("--module", type=int, default=1, help="Which module to run (default: 1)")
    parser.add_argument("--force", action="store_true", help="Re-clean CSVs that already have processed output")
    args = parser.parse_args()

    run_batch(view=args.view, module=args.module, force=args.force)


if __name__ == "__main__":
    main()
