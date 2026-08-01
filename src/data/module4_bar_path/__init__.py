"""Bar path analysis module for the weight lifting injury prediction project."""

from .config import BarPathConfig
from .video_loader import load_video
from .landmark_extractor import build_pose_detector, extract_landmarks
from .cleaner import clean_coordinates
from .feature_extractor import extract_bar_path_features
from .storage import save_cleaned_coordinates, save_frame_level_features
from .pipeline import process_all_videos
from .raw_extractor import extract_raw_landmarks, select_lifter, find_lift_window
from .raw_extractor import process_all_videos as extract_all_raw_landmarks
from .raw_cleaner import clean_raw_landmarks, build_lift_feature_table
from .raw_cleaner import process_all_videos as clean_and_extract_features

__all__ = [
    "BarPathConfig",
    "load_video",
    "build_pose_detector",
    "extract_landmarks",
    "clean_coordinates",
    "extract_bar_path_features",
    "save_cleaned_coordinates",
    "save_frame_level_features",
    "process_all_videos",
    "extract_raw_landmarks",
    "select_lifter",
    "find_lift_window",
    "extract_all_raw_landmarks",
    "clean_raw_landmarks",
    "build_lift_feature_table",
    "clean_and_extract_features",
]
