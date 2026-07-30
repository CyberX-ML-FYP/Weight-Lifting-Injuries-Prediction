"""
Hip & Knee Analysis — Video ingestion and keypoint-sequence dataset building.

WHY these improvements help Hip & Knee analysis
------------------------------------------------
- Frames are extracted with explicit error handling (missing ``ffmpeg``,
  missing/corrupt video files) instead of letting ``subprocess`` fail
  silently, and every step is logged with timing information (Improvement 7).
- ``build_keypoint_sequence_dataset`` applies the frame-quality checks and
  body-proportion normalization from ``hip_knee_pose_utils`` to every sampled
  frame, and — unlike the notebook's crude ``try/except`` that re-ran
  detection on the raw frame — falls back to the last known-good frame when
  a sample is rejected, so a single bad detection can no longer leave a
  frame slot in the sequence unset.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from src.features.hip_knee_config import (
    LABEL_MAP,
    MIN_KEYPOINT_CONFIDENCE,
    NUM_SAMPLED_FRAMES,
    REQUIRED_KEYPOINTS,
    VIDEO_EXTENSIONS,
    VIEW_DIR_NAMES,
)
from src.features.hip_knee_logger import get_logger, log_execution_time
from src.features.hip_knee_pose_utils import (
    BodyKeypoints,
    extract_body_keypoints,
    normalize_by_body_proportion,
    select_person_by_center,
)

logger = get_logger(__name__)


def _safe_dirname(name: str) -> str:
    """
    Sanitize a video stem for use as a directory name.

    Windows silently strips trailing spaces/dots from path components
    (e.g. creating ``"1 Average & 2 good "`` actually creates
    ``"1 Average & 2 good"``), which then makes later lookups against the
    original (unstripped) string fail with ``FileNotFoundError``. Stripping
    here keeps the directory name consistent with what Windows actually
    creates.
    """
    return name.strip().rstrip(".") or "_unnamed"


def get_label(video_path: Path | str) -> str:
    """
    Map a video filename to its quality label (Good/Average/Poor).

    Extends the original notebook's convention (strip a leading numeric id
    and any non-alphabetic characters, e.g. ``"8good.mp4" -> "Good"``) with
    substring matching so real-world filename typos/variants — extra
    trailing letters (``"26gooda.mp4"``), stray spaces (``"3 new.mp4"``),
    mixed case (``"64Good.mp4"``) — still resolve correctly instead of
    silently falling back to "Unknown". Files that don't contain any known
    keyword (e.g. re-shoot markers like "New"/"changed" in a "Missed"
    folder, or ambiguous combined names) still correctly return "Unknown"
    so they can be excluded rather than mislabeled.
    """
    filename = Path(video_path).stem
    cleaned = re.sub(r"^\d+", "", filename).lower()
    cleaned = re.sub(r"[^a-zA-Z]", "", cleaned)

    matches = [label for keyword, label in LABEL_MAP.items() if keyword in cleaned]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        logger.warning("Ambiguous filename (matches %s): %s — labelling as Unknown", matches, filename)
    return "Unknown"


def _resolve_ffmpeg_binary() -> str:
    """
    Locate an ffmpeg executable, preferring the system PATH.

    Falls back to the static binary bundled with the optional
    ``imageio-ffmpeg`` package when ``ffmpeg`` is not installed system-wide
    (e.g. this dev machine), so frame extraction works without requiring a
    separate system-level install.

    Raises:
        FileNotFoundError: If neither a system ``ffmpeg`` nor the
            ``imageio-ffmpeg`` fallback is available.
    """
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg is not None:
        return system_ffmpeg

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise FileNotFoundError(
            "ffmpeg was not found on PATH and the 'imageio-ffmpeg' fallback "
            "package is not installed; install ffmpeg or run "
            "`pip install imageio-ffmpeg` to extract frames."
        ) from exc


def _cleanup_partial_frames(output_dir: Path) -> None:
    """
    Remove any partially-written ``frame_*.jpg`` files after a failed ffmpeg
    run, so a later retry does not silently treat an incomplete/corrupt
    extraction as already-cached (see the ``existing`` skip-check below).
    """
    for partial_frame in output_dir.glob("frame_*.jpg"):
        partial_frame.unlink(missing_ok=True)


def extract_frames(video_path: Path, output_dir: Path, overwrite: bool = False) -> list[Path]:
    """
    Extract every frame of ``video_path`` as JPEGs into ``output_dir`` via ffmpeg.

    Args:
        video_path: Source video file.
        output_dir: Directory frames are written to (created if missing).
        overwrite: If ``False`` and frames already exist, skip re-extraction.

    Returns:
        Sorted list of extracted frame paths. Empty if the video decodes
        successfully but genuinely contains no frames (e.g. zero-duration).

    Raises:
        FileNotFoundError: If ffmpeg is unavailable (see
            ``_resolve_ffmpeg_binary``) or the video file does not exist.
        ValueError: If the video file exists but is empty (0 bytes).
        RuntimeError: If ffmpeg fails to decode the video (corrupted file or
            unsupported/unreadable format).
    """
    ffmpeg_binary = _resolve_ffmpeg_binary()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if video_path.stat().st_size == 0:
        raise ValueError(f"Video file is empty (0 bytes): {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("frame_*.jpg"))
    if existing and not overwrite:
        logger.info("Frames already extracted for %s (%d frames) — skipping", video_path.name, len(existing))
        return existing

    with log_execution_time(logger, f"ffmpeg frame extraction: {video_path.name}"):
        try:
            subprocess.run(
                [ffmpeg_binary, "-nostdin", "-y", "-i", str(video_path), str(output_dir / "frame_%05d.jpg")],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            _cleanup_partial_frames(output_dir)
            stderr_tail = exc.stderr.decode(errors="replace")[-500:] if exc.stderr else "(no stderr captured)"
            raise RuntimeError(
                f"ffmpeg failed to decode '{video_path.name}' — the file may be corrupted or in an "
                f"unsupported/unreadable format. ffmpeg stderr (tail): {stderr_tail}"
            ) from exc

    return sorted(output_dir.glob("frame_*.jpg"))


def discover_videos(directory: Path) -> dict[str, list[Path]]:
    """
    Group every video file in ``directory`` by its quality label.

    Matches any extension in ``VIDEO_EXTENSIONS`` case-insensitively (the
    real footage is a mix of ``.mp4`` and ``.MOV`` files, as exported by
    different phones/cameras).

    Returns:
        Mapping of ``{"Good": [...], "Average": [...], "Poor": [...]}``
        (labels with no matching videos are omitted).
    """
    grouped: dict[str, list[Path]] = {}
    for video_path in sorted(directory.iterdir()):
        if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        label = get_label(video_path)
        grouped.setdefault(label, []).append(video_path)

    for label, videos in grouped.items():
        logger.info("discover_videos | dir=%s label=%s count=%d", directory, label, len(videos))
    return grouped


def discover_videos_by_view(
    root_dir: Path,
    view_dir_names: tuple[str, ...] = VIEW_DIR_NAMES,
) -> dict[str, dict[str, list[Path]]]:
    """
    Group videos by camera view then by quality label.

    Mirrors the source footage layout (``root_dir`` = the local copy of the
    "FYP research" Drive folder, containing "Side view"/"Angle view"/"Front
    View" sub-folders).

    Args:
        root_dir: Directory containing one sub-directory per camera view.
        view_dir_names: Expected view sub-directory names.

    Returns:
        ``{"Side view": {"Good": [...], "Poor": [...]}, "Angle view": {...}, ...}``.
        A view is omitted if its folder does not exist under ``root_dir``.
    """
    videos_by_view: dict[str, dict[str, list[Path]]] = {}
    for view_name in view_dir_names:
        view_dir = root_dir / view_name
        if not view_dir.is_dir():
            logger.warning("View folder not found, skipping: %s", view_dir)
            continue
        videos_by_view[view_name] = discover_videos(view_dir)

    return videos_by_view


def sample_frame_indices(n_frames: int, n_samples: int = NUM_SAMPLED_FRAMES) -> np.ndarray:
    """Evenly sample ``n_samples`` frame indices across ``n_frames``."""
    if n_frames <= 0:
        return np.array([], dtype=int)
    return np.linspace(0, n_frames - 1, min(n_samples, n_frames), dtype=int)


def run_pose_model(
    pose_model: Any,
    frame_path: Path,
    min_confidence: float = MIN_KEYPOINT_CONFIDENCE,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Run a YOLO-pose model on one frame and return ``(keypoints_xy, confidences)``
    for the athlete (as opposed to a bystander/spotter/mirror reflection).

    YOLO can emit multiple "person" detections on a single-athlete frame
    (e.g. gym mirror reflections, or low-confidence phantom boxes on
    equipment), sometimes at high confidence. To pick the right one:

    1. Candidates are filtered to those whose mean keypoint confidence over
       the required joints meets ``min_confidence`` (same bar used by
       ``extract_body_keypoints``/``check_frame_quality``) \u2014 this alone
       resolves most single-athlete frames, since spurious detections are
       usually low-confidence.
    2. If more than one plausible candidate remains, ``select_person_by_center``
       (previously written but never wired in) is used to prefer whichever
       is unambiguously closest to the frame center \u2014 the real athlete is
       typically framed centrally, while bystanders/reflections are not.
    3. If center-distance is inconclusive (nobody, or more than one,
       qualifies), fall back to the highest-confidence plausible candidate
       rather than discarding the frame outright \u2014 rejecting an entire
       frame whenever disambiguation is inconclusive proved too aggressive
       in testing (real footage with mirrors could fail every sampled
       frame), so a best-effort selection keeps the pipeline usable while
       still being an improvement over blindly trusting detection order.

    Returns ``(None, None)`` when no usable person is detected, instead of
    raising, so the caller can apply the frame-quality fallback logic.
    """
    results = pose_model(str(frame_path), verbose=False)
    keypoints = results[0].keypoints
    if keypoints is None or keypoints.xy.shape[0] == 0:
        return None, None

    num_people = keypoints.xy.shape[0]
    all_xy = [keypoints.xy[i].cpu().numpy() for i in range(num_people)]
    all_conf = (
        [keypoints.conf[i].cpu().numpy() for i in range(num_people)]
        if keypoints.conf is not None
        else [None] * num_people
    )

    def _mean_conf(i: int) -> float:
        return 1.0 if all_conf[i] is None else float(all_conf[i][list(REQUIRED_KEYPOINTS)].mean())

    plausible = [i for i in range(num_people) if _mean_conf(i) >= min_confidence]

    if len(plausible) == 0:
        return None, None
    if len(plausible) == 1:
        person_idx = plausible[0]
    else:
        candidate_xy = [all_xy[i] for i in plausible]
        selected = select_person_by_center(candidate_xy, results[0].orig_shape)
        if selected is not None:
            person_idx = plausible[selected]
        else:
            person_idx = max(plausible, key=_mean_conf)
            logger.info(
                "%d plausible people detected in %s with no single candidate unambiguously closest "
                "to frame center \u2014 falling back to the highest-confidence candidate",
                len(plausible), frame_path,
            )

    return all_xy[person_idx], all_conf[person_idx]


def _collect_body_keypoints_with_paths(
    frames: list[Path],
    indices: np.ndarray,
    pose_model: Any,
    min_confidence: float,
) -> list[tuple[Path, BodyKeypoints]]:
    """
    Run the pose model over the sampled frame indices of one video and
    return ``(frame_path, BodyKeypoints)`` pairs for each accepted sample,
    holding the last known-good frame whenever a sample fails the quality
    check instead of leaving a gap (extracted so both the LSTM feature
    pipeline and the Rule A-D biomechanics pipeline share identical
    frame-selection logic). Keeping the frame path alongside the keypoints
    lets callers — e.g. the prediction pipeline's annotated-video output —
    draw overlays directly onto the exact image each keypoint set came from.
    """
    pairs: list[tuple[Path, BodyKeypoints]] = []
    last_good: BodyKeypoints | None = None

    for idx in indices:
        keypoints_xy, confidences = run_pose_model(pose_model, frames[idx], min_confidence)
        body_points = (
            extract_body_keypoints(keypoints_xy, confidences, min_confidence)
            if keypoints_xy is not None
            else None
        )

        if body_points is None:
            body_points = last_good
            if body_points is None:
                logger.warning("Frame %s rejected with no prior good frame — skipping video", frames[idx])
                continue
        else:
            last_good = body_points

        pairs.append((frames[idx], body_points))

    return pairs


def _collect_body_keypoints(
    frames: list[Path],
    indices: np.ndarray,
    pose_model: Any,
    min_confidence: float,
) -> list[BodyKeypoints]:
    """Same as ``_collect_body_keypoints_with_paths`` but without frame paths."""
    return [body_points for _, body_points in _collect_body_keypoints_with_paths(frames, indices, pose_model, min_confidence)]


def _build_frame_sequence(
    frames: list[Path],
    indices: np.ndarray,
    pose_model: Any,
    min_confidence: float,
    normalize: bool,
) -> list[np.ndarray]:
    """
    Build one video's per-frame feature sequence for the LSTM dataset
    (extracted from ``build_keypoint_sequence_dataset`` to keep its
    cognitive complexity low).
    """
    keypoints_sequence = _collect_body_keypoints(frames, indices, pose_model, min_confidence)
    return [
        normalize_by_body_proportion(body_points) if normalize else body_points.as_feature_vector()
        for body_points in keypoints_sequence
    ]


def extract_lift_keypoint_frames(
    video_path: Path,
    frames_root: Path,
    pose_model: Any,
    n_samples: int = NUM_SAMPLED_FRAMES,
    min_confidence: float = 0.5,
) -> list[BodyKeypoints]:
    """
    Extract the raw (unnormalized) per-frame ``BodyKeypoints`` sequence for
    one video, for use with ``hip_knee_biomechanics.analyze_lift`` (which
    needs named joint keypoints rather than a flattened/normalized feature
    vector).

    Args:
        video_path: Source video file.
        frames_root: Root directory frames are extracted into.
        pose_model: A loaded Ultralytics YOLO-pose model instance.
        n_samples: Number of frames sampled across the video.
        min_confidence: Minimum mean keypoint confidence to accept a frame.

    Returns:
        List of accepted ``BodyKeypoints`` (may be shorter than
        ``n_samples`` if some samples had no prior good frame to fall back on).
    """
    frame_dir = frames_root / _safe_dirname(video_path.stem)
    frames = extract_frames(video_path, frame_dir)
    if not frames:
        return []

    indices = sample_frame_indices(len(frames), n_samples)
    return _collect_body_keypoints(frames, indices, pose_model, min_confidence)


def extract_lift_frames_with_keypoints(
    video_path: Path,
    frames_root: Path,
    pose_model: Any,
    n_samples: int = NUM_SAMPLED_FRAMES,
    min_confidence: float = 0.5,
) -> list[tuple[Path, BodyKeypoints]]:
    """
    Like ``extract_lift_keypoint_frames``, but also returns the source frame
    image path each ``BodyKeypoints`` was extracted from.

    Used by the prediction pipeline (``src.models.hip_knee_predict``) to
    draw pose/angle/rule-score overlays directly onto the exact sampled
    frames used for classification and biomechanical analysis, producing an
    annotated output video.

    Returns:
        List of ``(frame_path, BodyKeypoints)`` pairs (may be shorter than
        ``n_samples`` if some samples had no prior good frame to fall back on).
    """
    frame_dir = frames_root / _safe_dirname(video_path.stem)
    frames = extract_frames(video_path, frame_dir)
    if not frames:
        return []

    indices = sample_frame_indices(len(frames), n_samples)
    return _collect_body_keypoints_with_paths(frames, indices, pose_model, min_confidence)


def build_keypoint_sequence_dataset(
    video_label_pairs: list[tuple[Path, str]],
    frames_root: Path,
    pose_model: Any,
    n_samples: int = NUM_SAMPLED_FRAMES,
    min_confidence: float = 0.5,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build an ``(X, y)`` keypoint-sequence dataset for LSTM training.

    For every video, ``n_samples`` frames are evenly sampled, run through
    ``pose_model`` and quality-checked; rejected frames reuse the last
    accepted frame's keypoints (holding the pose steady) rather than being
    left blank, keeping every sequence a fixed length.

    Args:
        video_label_pairs: ``(video_path, label)`` tuples, e.g. from
            ``discover_videos``.
        frames_root: Root directory frames are extracted into (one
            sub-directory per video).
        pose_model: A loaded Ultralytics YOLO-pose model instance.
        n_samples: Number of frames sampled per video.
        min_confidence: Minimum mean keypoint confidence to accept a frame.
        normalize: Apply body-proportion normalization to each frame's
            keypoints (Improvement 1).

    Returns:
        ``X`` of shape ``(n_videos, n_samples, 16)`` and ``y`` of shape
        ``(n_videos,)`` string labels.
    """
    sequences: list[np.ndarray] = []
    labels: list[str] = []

    for video_path, label in video_label_pairs:
        frame_dir = frames_root / _safe_dirname(video_path.stem)
        frames = extract_frames(video_path, frame_dir)
        if not frames:
            logger.warning("No frames extracted for %s — skipping", video_path.name)
            continue

        indices = sample_frame_indices(len(frames), n_samples)
        sequence = _build_frame_sequence(frames, indices, pose_model, min_confidence, normalize)

        if len(sequence) == n_samples:
            sequences.append(np.array(sequence))
            labels.append(label)
        else:
            logger.warning("Video %s produced %d/%d usable frames — excluded from dataset", video_path.name, len(sequence), n_samples)

    return np.array(sequences), np.array(labels)
