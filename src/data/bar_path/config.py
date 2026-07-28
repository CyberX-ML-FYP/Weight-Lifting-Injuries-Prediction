from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BarPathConfig:
    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3])
    raw_video_dir: Path = field(init=False)
    interim_output_dir: Path = field(init=False)
    processed_output_dir: Path = field(init=False)
    features_output_path: Path = field(init=False)
    video_extensions: tuple[str, ...] = (".mp4", ".mov")
    visibility_threshold: float = 0.5
    outlier_threshold: float = 50.0
    smoothing_window: int = 7
    smoothing_polyorder: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "raw_video_dir",
            self.root_dir / "data" / "raw" / "videos" / "front",
        )
        object.__setattr__(
            self, "interim_output_dir", self.root_dir / "data" / "interim" / "bar_path"
        )
        object.__setattr__(
            self,
            "processed_output_dir",
            self.root_dir / "data" / "processed" / "bar_path",
        )
        object.__setattr__(
            self,
            "features_output_path",
            self.root_dir / "data" / "features" / "bar_path_features.csv",
        )

        self.raw_video_dir.mkdir(parents=True, exist_ok=True)
        self.interim_output_dir.mkdir(parents=True, exist_ok=True)
        self.processed_output_dir.mkdir(parents=True, exist_ok=True)
        self.features_output_path.parent.mkdir(parents=True, exist_ok=True)
