"""
Module 4 (bar_path) — one-time batch: pre-render skeleton-overlay videos
for the app's demo lifts, so the Streamlit demo tab can play them back
instantly instead of processing a video live (same pattern as the
pre-computed *_cleaned.csv files it already uses for the charts).

Run whenever DEMO_LIFTS in app_module4_bar_path.py changes:
    python -m src.data.module4_bar_path.render_demo_skeletons
"""
from __future__ import annotations

from .config import BarPathConfig
from .skeleton_video import render_skeleton_video
from .utils import setup_logger

logger = setup_logger(__name__)

DEMO_VIDEO_IDS = ["2good", "10good", "46good", "13bad", "19bad", "62bad"]
DEMO_OUTPUT_DIRNAME = "skeleton_demo"


def main() -> None:
    config = BarPathConfig()
    out_dir = config.root_dir / "reports" / "figures" / "module4" / DEMO_OUTPUT_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for video_id in DEMO_VIDEO_IDS:
        video_path = None
        for ext in config.video_extensions:
            for candidate in (
                config.raw_video_dir / f"{video_id}{ext}",
                config.raw_video_dir / f"{video_id}{ext.upper()}",
            ):
                if candidate.exists():
                    video_path = candidate
                    break
            if video_path:
                break

        if video_path is None:
            logger.warning("No source video found for demo lift %s", video_id)
            failed.append(video_id)
            continue

        out_path = out_dir / f"{video_id}_skeleton.mp4"
        try:
            logger.info("Rendering skeleton video for %s", video_id)
            render_skeleton_video(video_path, out_path, config)
        except Exception:
            logger.exception("Failed to render skeleton video for %s", video_id)
            failed.append(video_id)

    succeeded = len(DEMO_VIDEO_IDS) - len(failed)
    if failed:
        logger.warning("Rendered %s/%s demo videos; failed: %s", succeeded, len(DEMO_VIDEO_IDS), failed)
    else:
        logger.info("Rendered %s/%s demo videos successfully", succeeded, len(DEMO_VIDEO_IDS))


if __name__ == "__main__":
    main()
