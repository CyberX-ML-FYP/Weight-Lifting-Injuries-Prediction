"""
Hip & Knee Analysis — End-to-end single-video prediction pipeline.

Runs the full inference pipeline on one lift video:

    Video -> Frame Extraction -> YOLO11 Pose -> Keypoint Sequence
          -> Feature Extraction -> Biomechanics (Rule A-D)
          -> LSTM Prediction -> Final Report

Reuses existing Hip & Knee module components rather than reimplementing
them: frame extraction/pose sampling (``src.data.hip_knee_dataset``),
Rule A-D biomechanics (``src.features.hip_knee_biomechanics``), adaptive
thresholds/rule scoring (``src.features.hip_knee_scoring``), the trained
LSTM architecture (``src.models.hip_knee_lstm``) and the model/label-encoder
loading helpers already written for evaluation (``src.models.hip_knee_evaluate``).

This module does NOT retrain the model, does NOT regenerate the training
dataset, and does NOT modify the saved model weights — it only loads
existing artifacts and produces a prediction report for one new video.

Outputs (per video):
    reports/prediction.json                     — full structured report
    reports/annotated_videos/<stem>_annotated.mp4 — pose/angle/score overlay video
    Console report (printed)
    ``PredictionReport`` Python object (returned)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.data.hip_knee_dataset import extract_lift_frames_with_keypoints
from src.features.hip_knee_biomechanics import LiftMetrics, analyze_lift, extract_hip_knee_angles
from src.features.hip_knee_config import (
    INTERIM_DIR,
    LOW_CONFIDENCE_THRESHOLD,
    MIN_KEYPOINT_CONFIDENCE,
    MODELS_DIR,
    NUM_SAMPLED_FRAMES,
    YOLO_POSE_MODEL,
)
from src.features.hip_knee_logger import get_logger, log_execution_time
from src.features.hip_knee_pose_utils import BodyKeypoints, normalize_by_body_proportion, smooth_angle_sequence
from src.features.hip_knee_scoring import (
    AdaptiveThresholds,
    compute_confidence_weighted_score,
    load_thresholds,
    normalize_score,
)
from src.models.hip_knee_evaluate import REPORTS_DIR, load_label_encoder, load_trained_model
from src.models.hip_knee_lstm import HipKneeLSTMClassifier
from src.models.hip_knee_train import LABEL_ENCODER_PATH, MODEL_PATH, TrainingConfig

logger = get_logger(__name__)

# ── Default paths ──────────────────────────────────────────────────────────────
DEFAULT_FRAMES_ROOT: Path = INTERIM_DIR / "predict"
DEFAULT_THRESHOLDS_PATH: Path = MODELS_DIR / "hip_knee_thresholds.json"
DEFAULT_PREDICTION_JSON_PATH: Path = REPORTS_DIR / "prediction.json"
DEFAULT_ANNOTATED_VIDEO_DIR: Path = REPORTS_DIR / "annotated_videos"


@dataclass(frozen=True)
class PredictionReport:
    """Full result of running the Hip & Knee pipeline on one video."""

    video_name: str
    predicted_class: str
    prediction_confidence: float
    rule_a_score: float
    rule_b_score: float
    rule_c_score: float
    rule_d_score: float
    hip_rom: float
    knee_rom: float
    hip_peak: float
    knee_peak: float
    synchronization_delay: float
    correlation: float
    rate_of_force_development: float
    confidence_score: float
    n_frames_used: int

    # ── Anthropometric normalization (reports/anthropometric_normalization.md) ──
    leg_length_reference_px: float
    hip_linear_rom_normalized: float
    knee_linear_rom_normalized: float
    hip_peak_velocity_normalized: float
    knee_peak_velocity_normalized: float

    # ── Confidence-weighted Rule A-D scoring (reports/confidence_weighted_rules.md) ──
    rule_a_confidence: float
    rule_b_confidence: float
    rule_c_confidence: float
    rule_d_confidence: float
    overall_confidence: float
    confidence_weighted_final_score: float
    is_low_confidence: bool

    def to_dict(self) -> dict:
        return asdict(self)


def load_pose_model(model_name: str = YOLO_POSE_MODEL) -> Any:
    """Load the Ultralytics YOLO11-pose model (auto-downloads weights on first use)."""
    with log_execution_time(logger, "Load YOLO11-pose model"):
        from ultralytics import YOLO

        return YOLO(model_name)


def predict_class(model: HipKneeLSTMClassifier, sequence: np.ndarray) -> tuple[int, float]:
    """Run the LSTM on one keypoint sequence and return ``(predicted_class_index, confidence)``."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(sequence[np.newaxis, ...], dtype=torch.float32))
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        classification = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[classification].item())
    return classification, confidence


def compute_rule_scores(classification: int, metrics: LiftMetrics, thresholds: AdaptiveThresholds) -> dict[str, float]:
    """
    Per-rule 0-100 normalized scores.

    Reuses the same min-max normalization ``compute_performance_score``
    blends into one final score, but keeps Rule A/B/C/D visible
    individually, as required by the prediction report.
    """
    ranges = (
        (thresholds.good_a, thresholds.good_b, thresholds.good_c, thresholds.good_d)
        if classification == 0
        else (thresholds.poor_a, thresholds.poor_b, thresholds.poor_c, thresholds.poor_d)
    )
    raw_values = (metrics.delay, metrics.hip_dominance_ratio, metrics.synchronization_corr, metrics.rfd_ratio)
    keys = ("rule_a_score", "rule_b_score", "rule_c_score", "rule_d_score")
    return {
        key: normalize_score(value, rng.min_value, rng.max_value)
        for key, value, rng in zip(keys, raw_values, ranges)
    }


def compute_rom_and_peaks(keypoints_sequence: list[BodyKeypoints]) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Compute hip/knee range-of-motion and peak angle from the extracted keypoint sequence."""
    hip_angles, knee_angles = extract_hip_knee_angles(keypoints_sequence)
    hip_angles = smooth_angle_sequence(hip_angles)
    knee_angles = smooth_angle_sequence(knee_angles)

    rom_peaks = {
        "hip_rom": float(np.max(hip_angles) - np.min(hip_angles)) if len(hip_angles) else 0.0,
        "knee_rom": float(np.max(knee_angles) - np.min(knee_angles)) if len(knee_angles) else 0.0,
        "hip_peak": float(np.max(hip_angles)) if len(hip_angles) else 0.0,
        "knee_peak": float(np.max(knee_angles)) if len(knee_angles) else 0.0,
    }
    return rom_peaks, hip_angles, knee_angles


def extract_rule_confidences(metrics: LiftMetrics) -> dict[str, float]:
    """Pull the per-rule confidence values already computed by ``analyze_lift`` into the dict shape ``compute_confidence_weighted_score`` expects."""
    return {
        "rule_a_confidence": metrics.rule_a_confidence,
        "rule_b_confidence": metrics.rule_b_confidence,
        "rule_c_confidence": metrics.rule_c_confidence,
        "rule_d_confidence": metrics.rule_d_confidence,
    }


def is_low_confidence_prediction(
    prediction_confidence: float,
    overall_confidence: float,
    threshold: float = LOW_CONFIDENCE_THRESHOLD,
) -> bool:
    """
    Flag a prediction as low-confidence instead of silently reporting it as a
    fully-trusted decision, if EITHER the LSTM's own softmax confidence OR
    the pose-derived overall Rule A-D confidence falls below ``threshold``.
    """
    return prediction_confidence < threshold or overall_confidence < threshold


def build_prediction_report(
    video_path: Path,
    classification: int,
    predicted_label: str,
    prediction_confidence: float,
    rule_scores: dict[str, float],
    rom_peaks: dict[str, float],
    metrics: LiftMetrics,
    n_frames_used: int,
) -> PredictionReport:
    """Assemble every computed value into one ``PredictionReport``."""
    rule_confidences = extract_rule_confidences(metrics)
    blended_rule_score = compute_confidence_weighted_score(rule_scores, rule_confidences)
    confidence_weighted_final_score = 50.0 + blended_rule_score * 0.5 if classification == 0 else blended_rule_score * 0.5
    overall_confidence = metrics.overall_confidence
    low_confidence = is_low_confidence_prediction(prediction_confidence, overall_confidence)

    return PredictionReport(
        video_name=video_path.name,
        predicted_class=predicted_label,
        prediction_confidence=prediction_confidence,
        rule_a_score=rule_scores["rule_a_score"],
        rule_b_score=rule_scores["rule_b_score"],
        rule_c_score=rule_scores["rule_c_score"],
        rule_d_score=rule_scores["rule_d_score"],
        hip_rom=rom_peaks["hip_rom"],
        knee_rom=rom_peaks["knee_rom"],
        hip_peak=rom_peaks["hip_peak"],
        knee_peak=rom_peaks["knee_peak"],
        synchronization_delay=metrics.delay,
        correlation=metrics.synchronization_corr,
        rate_of_force_development=metrics.rfd_ratio,
        confidence_score=metrics.mean_confidence,
        n_frames_used=n_frames_used,
        leg_length_reference_px=metrics.reference.primary_reference,
        hip_linear_rom_normalized=metrics.hip_linear_rom_normalized,
        knee_linear_rom_normalized=metrics.knee_linear_rom_normalized,
        hip_peak_velocity_normalized=metrics.hip_peak_linear_velocity_normalized,
        knee_peak_velocity_normalized=metrics.knee_peak_linear_velocity_normalized,
        rule_a_confidence=rule_confidences["rule_a_confidence"],
        rule_b_confidence=rule_confidences["rule_b_confidence"],
        rule_c_confidence=rule_confidences["rule_c_confidence"],
        rule_d_confidence=rule_confidences["rule_d_confidence"],
        overall_confidence=overall_confidence,
        confidence_weighted_final_score=confidence_weighted_final_score,
        is_low_confidence=low_confidence,
    )


def save_prediction_json(report: PredictionReport, path: Path = DEFAULT_PREDICTION_JSON_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    logger.info("Saved prediction JSON -> %s", path)


def _pt(point: np.ndarray) -> tuple[int, int]:
    return int(point[0]), int(point[1])


def draw_pose_overlay(image: np.ndarray, body: BodyKeypoints, hip_angle: float, knee_angle: float) -> np.ndarray:
    """Draw hip/knee/thigh lines, joints and the current joint angles onto one frame."""
    canvas = image.copy()
    sides = (
        (body.left_shoulder, body.left_hip, body.left_knee, body.left_ankle),
        (body.right_shoulder, body.right_hip, body.right_knee, body.right_ankle),
    )
    for shoulder, hip, knee, ankle in sides:
        cv2.line(canvas, _pt(shoulder), _pt(hip), (0, 255, 0), 2)      # torso -> hip line
        cv2.line(canvas, _pt(hip), _pt(knee), (0, 165, 255), 2)        # hip -> knee (thigh)
        cv2.line(canvas, _pt(knee), _pt(ankle), (255, 0, 0), 2)        # knee -> ankle line
        for joint in (shoulder, hip, knee, ankle):
            cv2.circle(canvas, _pt(joint), 4, (0, 0, 255), -1)

    cv2.putText(canvas, f"Hip: {hip_angle:.1f} deg", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(canvas, f"Knee: {knee_angle:.1f} deg", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return canvas


def draw_report_overlay(image: np.ndarray, report: PredictionReport) -> np.ndarray:
    """Overlay the predicted class and Rule A-D scores onto a frame."""
    canvas = image.copy()
    lines = [
        f"Prediction: {report.predicted_class} ({report.prediction_confidence * 100:.1f}%)",
        f"Rule A: {report.rule_a_score:.1f}  Rule B: {report.rule_b_score:.1f}",
        f"Rule C: {report.rule_c_score:.1f}  Rule D: {report.rule_d_score:.1f}",
    ]
    y = canvas.shape[0] - 15 - (len(lines) - 1) * 22
    for line in lines:
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        y += 22
    return canvas


def save_annotated_video(
    frame_keypoint_pairs: list[tuple[Path, BodyKeypoints]],
    hip_angles: np.ndarray,
    knee_angles: np.ndarray,
    report: PredictionReport,
    output_path: Path,
    fps: int = 4,
) -> Path:
    """Render the sampled/analysed frames with pose + report overlays into an .mp4 video."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_image = cv2.imread(str(frame_keypoint_pairs[0][0]))
    if first_image is None:
        raise ValueError(f"Could not read frame image: {frame_keypoint_pairs[0][0]}")
    height, width = first_image.shape[:2]

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(
            f"Could not open video writer for {output_path} (missing/unsupported mp4v codec on this system)."
        )
    try:
        for (frame_path, body), hip_angle, knee_angle in zip(frame_keypoint_pairs, hip_angles, knee_angles):
            image = cv2.imread(str(frame_path))
            if image is None:
                logger.warning("Could not read frame %s — skipping in annotated video", frame_path)
                continue
            annotated = draw_pose_overlay(image, body, float(hip_angle), float(knee_angle))
            annotated = draw_report_overlay(annotated, report)
            writer.write(annotated)
    finally:
        writer.release()

    logger.info("Saved annotated video -> %s", output_path)
    return output_path


def print_prediction_report(report: PredictionReport) -> None:
    print("=" * 60)
    print("HIP & KNEE — LIFT PREDICTION REPORT")
    print("=" * 60)
    print(f"Video                     : {report.video_name}")
    print(f"Predicted Class           : {report.predicted_class}")
    print(f"Prediction Confidence     : {report.prediction_confidence * 100:.2f}%")
    print()
    print(f"Rule A Score (Sequence)   : {report.rule_a_score:.2f}")
    print(f"Rule B Score (Hip Dom.)   : {report.rule_b_score:.2f}")
    print(f"Rule C Score (Sync.)      : {report.rule_c_score:.2f}")
    print(f"Rule D Score (RFD)        : {report.rule_d_score:.2f}")
    print(f"Rule A Confidence         : {report.rule_a_confidence:.4f}")
    print(f"Rule B Confidence         : {report.rule_b_confidence:.4f}")
    print(f"Rule C Confidence         : {report.rule_c_confidence:.4f}")
    print(f"Rule D Confidence         : {report.rule_d_confidence:.4f}")
    print(f"Overall Confidence        : {report.overall_confidence:.4f}")
    print(f"Confidence-Weighted Score : {report.confidence_weighted_final_score:.2f}")
    print(f"Low-Confidence Flag       : {report.is_low_confidence}")
    print()
    print(f"Hip ROM (deg)             : {report.hip_rom:.2f}")
    print(f"Knee ROM (deg)            : {report.knee_rom:.2f}")
    print(f"Hip Peak (deg)            : {report.hip_peak:.2f}")
    print(f"Knee Peak (deg)           : {report.knee_peak:.2f}")
    print()
    print(f"Synchronization Delay (s) : {report.synchronization_delay:.4f}")
    print(f"Correlation               : {report.correlation:.4f}")
    print(f"Rate of Force Development : {report.rate_of_force_development:.4f}")
    print(f"Confidence Score (pose)   : {report.confidence_score:.4f}")
    print(f"Frames used               : {report.n_frames_used}")
    print()
    print(f"Leg Length Reference (px) : {report.leg_length_reference_px:.1f}")
    print(f"Hip Linear ROM (norm.)    : {report.hip_linear_rom_normalized:.4f} leg-lengths")
    print(f"Knee Linear ROM (norm.)   : {report.knee_linear_rom_normalized:.4f} leg-lengths")
    print(f"Hip Peak Velocity (norm.) : {report.hip_peak_velocity_normalized:.4f} leg-lengths/s")
    print(f"Knee Peak Velocity (norm.): {report.knee_peak_velocity_normalized:.4f} leg-lengths/s")
    if report.is_low_confidence:
        print(f"WARNING: low-confidence prediction (< {LOW_CONFIDENCE_THRESHOLD:.2f}); interpret with caution.")
    print("=" * 60)


def _build_sequence(keypoints_sequence: list[BodyKeypoints]) -> np.ndarray:
    return np.array([normalize_by_body_proportion(bp) for bp in keypoints_sequence], dtype=np.float32)


def predict_video(
    video_path: Path,
    frames_root: Path = DEFAULT_FRAMES_ROOT,
    pose_model: Any | None = None,
    model_path: Path = MODEL_PATH,
    label_encoder_path: Path = LABEL_ENCODER_PATH,
    thresholds_path: Path = DEFAULT_THRESHOLDS_PATH,
    output_json_path: Path | None = None,
    output_video_path: Path | None = None,
    n_samples: int = NUM_SAMPLED_FRAMES,
    min_confidence: float = MIN_KEYPOINT_CONFIDENCE,
    config: TrainingConfig | None = None,
    save_video: bool = True,
    video_fps: int = 4,
) -> PredictionReport:
    """
    Run the full Hip & Knee prediction pipeline on one video.

    Video -> Frame Extraction -> YOLO11 Pose -> Keypoint Sequence ->
    Feature Extraction -> Biomechanics (Rule A-D) -> LSTM Prediction ->
    Final Report (JSON + console + annotated video + returned object).

    Does NOT retrain the model or regenerate the training dataset — it only
    loads the already-trained artifacts and analyses ``video_path``.
    """
    config = config or TrainingConfig()
    video_path = Path(video_path)
    output_json_path = output_json_path or DEFAULT_PREDICTION_JSON_PATH
    output_video_path = output_video_path or (DEFAULT_ANNOTATED_VIDEO_DIR / f"{video_path.stem}_annotated.mp4")

    with log_execution_time(logger, f"predict_video: {video_path.name}"):
        pose_model = pose_model or load_pose_model()

        frame_keypoint_pairs = extract_lift_frames_with_keypoints(video_path, frames_root, pose_model, n_samples, min_confidence)
        if not frame_keypoint_pairs:
            raise ValueError(f"No usable frames extracted from {video_path}")
        keypoints_sequence = [body for _, body in frame_keypoint_pairs]

        label_encoder = load_label_encoder(label_encoder_path)
        thresholds = load_thresholds(thresholds_path)
        sequence = _build_sequence(keypoints_sequence)
        model = load_trained_model(model_path, input_size=sequence.shape[-1], num_classes=len(label_encoder.classes_), config=config)

        classification, prediction_confidence = predict_class(model, sequence)
        predicted_label = str(label_encoder.inverse_transform([classification])[0])

        metrics = analyze_lift(keypoints_sequence)
        rule_scores = compute_rule_scores(classification, metrics, thresholds)
        rom_peaks, hip_angles, knee_angles = compute_rom_and_peaks(keypoints_sequence)

        report = build_prediction_report(
            video_path,
            classification,
            predicted_label,
            prediction_confidence,
            rule_scores,
            rom_peaks,
            metrics,
            len(keypoints_sequence),
        )

        save_prediction_json(report, output_json_path)
        if save_video:
            # Annotated-video rendering is a secondary output — a codec/writer
            # failure here should not discard the JSON/console prediction
            # report, which is the pipeline's core deliverable.
            try:
                save_annotated_video(frame_keypoint_pairs, hip_angles, knee_angles, report, output_video_path, fps=video_fps)
            except Exception:
                logger.exception("Annotated video generation failed for %s — continuing without it", video_path.name)

    print_prediction_report(report)
    logger.info(
        "predict_video complete | video=%s predicted_class=%s confidence=%.4f",
        video_path.name, report.predicted_class, report.prediction_confidence,
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Hip & Knee lift-quality prediction pipeline on one video.")
    parser.add_argument("--video", type=Path, required=True, help="Path to the input lift video.")
    parser.add_argument("--frames-root", type=Path, default=DEFAULT_FRAMES_ROOT)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--label-encoder-path", type=Path, default=LABEL_ENCODER_PATH)
    parser.add_argument("--thresholds-path", type=Path, default=DEFAULT_THRESHOLDS_PATH)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-video", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=NUM_SAMPLED_FRAMES)
    parser.add_argument("--min-confidence", type=float, default=MIN_KEYPOINT_CONFIDENCE)
    parser.add_argument("--fps", type=int, default=4, help="Playback FPS for the annotated output video.")
    parser.add_argument("--skip-video", action="store_true", help="Skip generating the annotated output video.")
    return parser.parse_args()


def main() -> PredictionReport:
    args = _parse_args()
    try:
        return predict_video(
            video_path=args.video,
            frames_root=args.frames_root,
            model_path=args.model_path,
            label_encoder_path=args.label_encoder_path,
            thresholds_path=args.thresholds_path,
            output_json_path=args.output_json,
            output_video_path=args.output_video,
            n_samples=args.n_samples,
            min_confidence=args.min_confidence,
            save_video=not args.skip_video,
            video_fps=args.fps,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        # Known/expected failure modes (missing file, empty video, corrupted/
        # unsupported format, no usable frames, etc.) — report a clean,
        # actionable message instead of a raw traceback; full details still
        # go to the log file via logger.exception.
        logger.exception("Hip & Knee prediction failed for %s", args.video)
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
