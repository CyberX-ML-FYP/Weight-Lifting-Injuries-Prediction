"""
Module 3 — Pipeline Runner
Runs batch processing, feature extraction, and model training in sequence.

Usage:
    python -m src.data.module3_arm_analysis.run_pipeline
    python -m src.data.module3_arm_analysis.run_pipeline --clean
    python -m src.data.module3_arm_analysis.run_pipeline --skip-videos
"""
import argparse
import os

from src.data.module3_arm_analysis.batch_processor import run_batch
from src.data.module3_arm_analysis.feature_extractor import build_master_dataset
from src.data.module3_arm_analysis.train_model import train_module3_model
from src.data.module3_arm_analysis.config import DATA_DIR


def _print_step_banner(step_no, total_steps, title):
    print(f"\n========== STEP {step_no}/{total_steps}: {title} ==========")


def _clean_processed_csvs():
    """Delete only CSV files in data/processed (never touch raw videos)."""
    processed_dir = os.path.join(DATA_DIR, "processed")
    if not os.path.isdir(processed_dir):
        print(f"No processed directory found: {processed_dir}")
        return

    removed = 0
    for filename in os.listdir(processed_dir):
        if filename.lower().endswith(".csv"):
            file_path = os.path.join(processed_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                removed += 1

    print(f"Cleaned processed CSV files: {removed}")


def run_pipeline(clean=False, skip_videos=False):
    """Run Module 3 pipeline end-to-end."""
    if clean:
        _print_step_banner(0, 3, "Cleaning Processed CSVs")
        _clean_processed_csvs()

    if not skip_videos:
        _print_step_banner(1, 3, "Batch Processing")
        run_batch(view=None)
    else:
        _print_step_banner(1, 3, "Batch Processing (Skipped)")
        print("Skipping video processing as requested (--skip-videos).")

    _print_step_banner(2, 3, "Feature Extraction")
    build_master_dataset()

    _print_step_banner(3, 3, "Model Training")
    train_module3_model()

    print("\n========== PIPELINE COMPLETE ==========")


def _parse_args():
    parser = argparse.ArgumentParser(description="Run Module 3 full pipeline.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete all CSV files in data/processed before running.",
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Skip video processing and run only feature extraction + training.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(clean=args.clean, skip_videos=args.skip_videos)
