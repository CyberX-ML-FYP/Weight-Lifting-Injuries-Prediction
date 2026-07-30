from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TrunkConfig:
    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3])
    raw_video_dir: Path = field(init=False)
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

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "raw_video_dir",
            self.root_dir / "data" / "raw" / "videos" / "side",
        )
        object.__setattr__(
            self,
            "interim_output_dir",
            self.root_dir / "data" / "interim" / "module1",
        )
        object.__setattr__(
            self,
            "features_output_path",
            self.root_dir / "data" / "features" / "trunk_features.csv",
        )

        self.raw_video_dir.mkdir(parents=True, exist_ok=True)
        self.interim_output_dir.mkdir(parents=True, exist_ok=True)
        self.features_output_path.parent.mkdir(parents=True, exist_ok=True)
