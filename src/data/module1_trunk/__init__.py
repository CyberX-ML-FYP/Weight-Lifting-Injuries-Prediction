"""Trunk and spine analysis module for the weight lifting injury prediction project."""

from .config import TrunkConfig
from .video_loader import load_video
from .landmark_extractor import build_pose_detector, extract_landmarks
from .feature_extractor import extract_trunk_features, summarize_trunk_features
from .storage import save_frame_level_features
from .pipeline import process_all_videos

__all__ = [
    "TrunkConfig",
    "load_video",
    "build_pose_detector",
    "extract_landmarks",
    "extract_trunk_features",
    "summarize_trunk_features",
    "save_frame_level_features",
    "process_all_videos",
]
