"""
Hip & Knee Analysis — Configuration constants.

Centralises paths, YOLO11-pose keypoint indices and default thresholds used
across the hip/knee analysis pipeline (feature extraction, model training,
evaluation and inference). Keeping these values in one place means the
hardcoded numbers that used to live in the original research notebook can now
be tuned/overridden in a single location instead of being duplicated across
scripts.

This module belongs to the Hip & Knee (lower-body) analysis pipeline and does
not modify or depend on any other teammate's module (e.g. ``module3_arm_analysis``).
"""
from __future__ import annotations

from pathlib import Path

# ── Project paths (pathlib, no hardcoded absolute paths) ─────────────────────
BASE_DIR: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = BASE_DIR / "data"
# Videos are organised the same way as the source Google Drive folder
# ("FYP research" -> "Side view" / "Angle view" / "Front View"), copied
# locally under data/raw/hip_knee/<view>/ so this module never touches the
# "data/raw_videos" tree used by the arm-analysis module.
RAW_VIDEO_DIR: Path = DATA_DIR / "raw" / "hip_knee"
INTERIM_DIR: Path = DATA_DIR / "interim" / "hip_knee"
PROCESSED_DIR: Path = DATA_DIR / "processed" / "hip_knee"

# ── Camera view folders (mirrors the Drive "FYP research" layout) ────────────
VIEW_DIR_NAMES: tuple[str, ...] = ("Side view", "Angle view", "Front View")

# ── Accepted raw video file extensions (footage is a mix of .mp4 and .MOV) ────
VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".mov")

MODELS_DIR: Path = BASE_DIR / "models" / "hip_knee_analysis"
REPORTS_DIR: Path = BASE_DIR / "reports" / "figures" / "hip_knee_analysis"
LOGS_DIR: Path = BASE_DIR / "reports" / "logs"

# ── YOLO11-Pose (COCO-17) keypoint indices ────────────────────────────────────
# https://docs.ultralytics.com/tasks/pose/ — used instead of MediaPipe's
# 33-point layout because the existing hip/knee pipeline was built on
# Ultralytics YOLO11-pose.
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

REQUIRED_KEYPOINTS = (
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
)

# ── Model file names ───────────────────────────────────────────────────────────
YOLO_POSE_MODEL = "yolo11n-pose.pt"
YOLO_SEG_MODEL = "yolo11n-seg.pt"

# ── Frame quality thresholds (Improvement 2) ──────────────────────────────────
MIN_KEYPOINT_CONFIDENCE: float = 0.5
CENTER_DISTANCE_THRESHOLD_PX: float = 100.0

# ── Temporal smoothing defaults (Improvement 3) ───────────────────────────────
SAVGOL_WINDOW_LENGTH: int = 7
SAVGOL_POLYORDER: int = 2

# ── Sampling / video processing ───────────────────────────────────────────────
DEFAULT_FPS: int = 30
NUM_SAMPLED_FRAMES: int = 20

# ── Rule A-D scoring weights (kept identical to the original research notebook) ─
RULE_WEIGHTS: tuple[float, float, float, float] = (0.35, 0.30, 0.20, 0.15)

# ── Confidence-weighted scoring (reports/confidence_weighted_rules.md) ───────
# Below this combined-confidence threshold, a prediction is flagged as
# low-confidence instead of being reported as a fully-trusted decision.
LOW_CONFIDENCE_THRESHOLD: float = 0.6

# ── Label mapping ─────────────────────────────────────────────────────────────
LABEL_MAP = {"good": "Good", "average": "Average", "bad": "Poor"}
CLASS_NAMES = ("Good", "Poor")

# ── Gemini feedback (Improvement — never hardcode secrets) ───────────────────
GEMINI_API_KEY_ENV_VAR: str = "GEMINI_API_KEY"
GEMINI_MODEL_NAME: str = "gemini-2.5-flash"
