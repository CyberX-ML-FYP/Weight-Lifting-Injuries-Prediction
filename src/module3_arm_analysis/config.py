"""
Module 3 — Configuration constants
Author: Pasindu (214027H)
"""
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR    = os.path.join(BASE_DIR, "data")
OUTPUT_DIR  = os.path.join(BASE_DIR, "reports", "figures", "module3")
ANNOTATED_VIDEO_DIR = os.path.join(BASE_DIR, "reports", "annotated_videos")
MODEL_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_full.task")

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