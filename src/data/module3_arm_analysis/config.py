"""
Module 3 — Configuration constants
Author: Pasindu (214027H)
"""
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR    = os.path.join(BASE_DIR, "data")
OUTPUT_DIR  = os.path.join(BASE_DIR, "reports", "figures", "module3")
MODEL_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_full.task")

# Raw videos live under data/raw/videos/<view>/, same convention as the
# other modules (bar_path, module1_trunk).
RAW_VIDEO_DIR = os.path.join(DATA_DIR, "raw", "videos")

# Per-video cleaned landmark coordinates (visibility-filtered, interpolated,
# smoothed) -- computed before any angle math runs on them.
PROCESSED_DIR = os.path.join(DATA_DIR, "processed", "module3")

# Per-video per-frame feature CSVs (one row per processed frame).
INTERIM_DIR = os.path.join(DATA_DIR, "interim", "module3")

# One-row-per-lift merged dataset (module3_master_dataset.csv) and any
# other final feature tables for this module.
FEATURES_DIR = os.path.join(DATA_DIR, "features", "module3")

MASTER_DATASET_PATH = os.path.join(FEATURES_DIR, "module3_master_dataset.csv")

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

# ── MediaPipe Landmark Indices ────────────────────────────────────────────────
LEFT_SHOULDER  = 11
LEFT_ELBOW     = 13
LEFT_WRIST     = 15
RIGHT_SHOULDER = 12
RIGHT_ELBOW    = 14
RIGHT_WRIST    = 16
LEFT_HIP       = 23
RIGHT_HIP      = 24

# ── Deviation Thresholds ──────────────────────────────────────────────────────
SYMMETRY_THRESHOLD = 15.0   # degrees — bilateral arm symmetry tolerance
LOCKOUT_THRESHOLD  = 160.0  # degrees — minimum elbow extension at jerk lockout
RACK_THRESHOLD     = 100.0  # degrees — maximum elbow angle at catch/rack position

# ── Video Processing ──────────────────────────────────────────────────────────
FRAME_SKIP         = 3      # process every Nth frame for speed
MIN_DETECTION_CONF = 0.5
MIN_TRACKING_CONF  = 0.5