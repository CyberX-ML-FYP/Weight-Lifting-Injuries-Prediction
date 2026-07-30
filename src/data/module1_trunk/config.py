from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TrunkConfig:
    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3])
    raw_video_dir: Path = field(init=False)
    processed_output_dir: Path = field(init=False)
    interim_output_dir: Path = field(init=False)
    features_output_path: Path = field(init=False)
    video_extensions: tuple[str, ...] = (".mp4", ".mov")

    left_shoulder: int = 11
    right_shoulder: int = 12
    left_hip: int = 23
    right_hip: int = 24

    shoulder_asym_threshold: float = 0.03
    frame_skip: int = 3
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    num_poses: int = 5

    visibility_threshold: float = 0.5
    outlier_threshold: float = 0.15
    smoothing_window: int = 7
    smoothing_polyorder: int = 2

    model_path: Path = field(init=False)
    model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    )

    def __post_init__(self) -> None:
        # Shared with module3_arm_analysis so the ~9MB model is only
        # downloaded once for the whole project.
        object.__setattr__(
            self,
            "model_path",
            self.root_dir / "src" / "data" / "module3_arm_analysis" / "pose_landmarker_full.task",
        )
        object.__setattr__(
            self,
            "raw_video_dir",
            self.root_dir / "data" / "raw" / "videos" / "side",
        )
        object.__setattr__(
            self,
            "processed_output_dir",
            self.root_dir / "data" / "processed" / "module1",
        )
        object.__setattr__(
            self,
            "interim_output_dir",
            self.root_dir / "data" / "interim" / "module1",
        )
        object.__setattr__(
            self,
            "features_output_path",
            self.root_dir / "data" / "features" / "module1" / "trunk_features.csv",
        )

        self.raw_video_dir.mkdir(parents=True, exist_ok=True)
        self.processed_output_dir.mkdir(parents=True, exist_ok=True)
        self.interim_output_dir.mkdir(parents=True, exist_ok=True)
        self.features_output_path.parent.mkdir(parents=True, exist_ok=True)
