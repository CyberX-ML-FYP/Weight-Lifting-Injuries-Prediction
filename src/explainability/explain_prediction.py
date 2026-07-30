"""
Hip & Knee Analysis — Explainable AI (Captum Integrated Gradients).

Explains ONE prediction made by the already-trained, already-validated
Hip & Knee LSTM classifier (``models/hip_knee_lstm.pth``), without
retraining it, without modifying its weights, and without touching the
prediction logic in ``src.models.hip_knee_predict`` or any teammate module
(``src.module3_arm_analysis``).

Pipeline
--------
1. Load the trained LSTM (same architecture/weights loader already used by
   ``src.models.hip_knee_evaluate`` / ``src.models.hip_knee_predict``).
2. Load ONE real feature vector — the (seq_len, 16) normalized keypoint
   sequence actually fed to the LSTM for one real video — by reusing the
   exact same pipeline stages ``predict_video`` calls (frame extraction,
   pose estimation, body-proportion normalization, Rule A-D biomechanics).
3. Compute Captum Integrated Gradients attributions for the predicted
   class, with respect to that real input tensor.
4. Aggregate the raw per-keypoint attributions into the requested,
   human-readable feature categories (Hip ROM, Knee ROM, Hip Peak, Knee
   Peak, Sequential Delay, Synchronization, Hip Dominance, Correlation,
   RFD, Confidence, Anthropometric features) using a documented anatomical/
   functional mapping derived directly from the real Rule A-D formulas, and
   normalize the resulting importance values so they sum to 1.0.

Outputs:
    reports/model_explainability.md   — full write-up (method + results + limitations)
    reports/integrated_gradients.png  — attribution heatmap + feature-importance bar chart
    reports/feature_importance.csv    — normalized feature importance table

This module reads existing artifacts only (model weights, label encoder,
adaptive thresholds) and never calls any training/optimizer code — the
trained model is used purely in ``eval()``/``no_grad``-free inference mode
required by Captum, with no parameter updates applied at any point.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless-safe: this script only saves figures, never shows them
import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import IntegratedGradients

from src.data.hip_knee_dataset import extract_lift_frames_with_keypoints
from src.features.hip_knee_biomechanics import analyze_lift
from src.features.hip_knee_config import BASE_DIR, MIN_KEYPOINT_CONFIDENCE, NUM_SAMPLED_FRAMES, RAW_VIDEO_DIR
from src.features.hip_knee_logger import get_logger, log_execution_time
from src.features.hip_knee_pose_utils import BodyKeypoints, normalize_by_body_proportion
from src.features.hip_knee_scoring import load_thresholds
from src.models.hip_knee_evaluate import load_label_encoder, load_trained_model
from src.models.hip_knee_lstm import HipKneeLSTMClassifier
from src.models.hip_knee_predict import (
    DEFAULT_FRAMES_ROOT,
    DEFAULT_THRESHOLDS_PATH,
    LABEL_ENCODER_PATH,
    MODEL_PATH,
    PredictionReport,
    build_prediction_report,
    compute_rom_and_peaks,
    compute_rule_scores,
    load_pose_model,
    predict_class,
)
from src.models.hip_knee_train import TrainingConfig

logger = get_logger(__name__)

# ── Output paths (top-level reports/ dir, per explainability spec) ───────────
REPORTS_DIR: Path = BASE_DIR / "reports"
MODEL_EXPLAINABILITY_MD_PATH: Path = REPORTS_DIR / "model_explainability.md"
INTEGRATED_GRADIENTS_PNG_PATH: Path = REPORTS_DIR / "integrated_gradients.png"
FEATURE_IMPORTANCE_CSV_PATH: Path = REPORTS_DIR / "feature_importance.csv"

# ── Default sample used to explain (same real footage used throughout the
# Hip & Knee validation work) ─────────────────────────────────────────────────
DEFAULT_EXPLAIN_VIDEO: Path = RAW_VIDEO_DIR / "Side view" / "13bad.MOV"
EXPLAIN_FRAMES_ROOT: Path = DEFAULT_FRAMES_ROOT.parent / "explain"

# ── Anatomical grouping of the 16 raw per-frame input dims ───────────────────
# BodyKeypoints.as_feature_vector() / normalize_by_body_proportion() lay out
# each frame as 8 bilateral keypoints x (x, y): LSh, RSh, LHip, RHip, LKnee,
# RKnee, LAnk, RAnk. Grouped here into 4 joint groups (bilateral-combined).
JOINT_GROUPS: tuple[str, ...] = ("shoulder", "hip", "knee", "ankle")
_KEYPOINT_TO_GROUP_INDEX: tuple[int, ...] = (0, 0, 1, 1, 2, 2, 3, 3)

# Functional/anatomical dependency weights mapping each requested named
# feature onto the 4 joint groups (shoulder, hip, knee, ankle), derived
# directly from the REAL formulas in ``hip_knee_biomechanics.py``:
#   - Hip ROM/Peak            <- hip angle = angle(shoulder, hip, knee)
#   - Knee ROM/Peak           <- knee angle = angle(hip, knee, ankle)
#   - Sequential Delay/Synchronization/Hip Dominance/Correlation/RFD
#     (Rule A-D)              <- all derived from BOTH the hip-angle and
#                                knee-angle velocity/acceleration signals
#   - Confidence              <- global model sensitivity, not anatomically
#                                localized to one joint group
#   - Anthropometric features <- leg_length (hip->knee->ankle) + torso_length
#                                (shoulder->hip), the same reference used to
#                                normalize the model's own per-frame inputs
#
# This mapping is a documented heuristic used ONLY to translate raw
# per-keypoint Integrated Gradients attribution into the human-readable
# categories requested — it does not alter the model, its inputs, or its
# prediction in any way. See ``reports/model_explainability.md`` for the
# full rationale and its stated limitations.
# Single source of truth for the 11 requested features: (name, joint-group
# weights, PredictionReport source field, human-readable description). Kept
# as one list-of-tuples (rather than 3 separate dicts keyed by the same
# repeated name) so each feature name literal only appears once.
FEATURE_DEFINITIONS: tuple[tuple[str, tuple[float, float, float, float], str, str], ...] = (
    ("Hip ROM", (0.25, 0.50, 0.25, 0.00), "hip_rom",
     "Angular range of motion of the hip joint across the lift (degrees)."),
    ("Knee ROM", (0.00, 0.25, 0.50, 0.25), "knee_rom",
     "Angular range of motion of the knee joint across the lift (degrees)."),
    ("Hip Peak", (0.25, 0.50, 0.25, 0.00), "hip_peak",
     "Peak hip flexion angle reached during the lift (degrees)."),
    ("Knee Peak", (0.00, 0.25, 0.50, 0.25), "knee_peak",
     "Peak knee flexion angle reached during the lift (degrees)."),
    ("Sequential Delay", (0.125, 0.375, 0.375, 0.125), "synchronization_delay",
     "Rule A - timing delay (s) between peak knee and peak hip angular velocity."),
    ("Synchronization", (0.125, 0.375, 0.375, 0.125), "rule_c_score",
     "Rule C score (0-100) - how closely the hip and knee velocity signals move together."),
    ("Hip Dominance", (0.125, 0.375, 0.375, 0.125), "rule_b_score",
     "Rule B score (0-100) - ratio of peak hip to peak knee angular velocity."),
    ("Correlation", (0.125, 0.375, 0.375, 0.125), "correlation",
     "Rule C raw Pearson correlation coefficient between hip and knee angular velocity."),
    ("RFD", (0.125, 0.375, 0.375, 0.125), "rate_of_force_development",
     "Rule D - ratio of peak hip to peak knee angular acceleration (rate-of-force-development proxy)."),
    ("Confidence", (0.25, 0.25, 0.25, 0.25), "prediction_confidence",
     "The LSTM's own softmax confidence in the predicted class."),
    ("Anthropometric Features", (0.20, 0.40, 0.20, 0.20), "leg_length_reference_px",
     "Leg/torso-length reference and the linear ROM/peak-velocity features normalized by it."),
)

FEATURE_GROUP_WEIGHTS: dict[str, tuple[float, float, float, float]] = {
    name: weights for name, weights, _field, _desc in FEATURE_DEFINITIONS
}
FEATURE_SOURCE_FIELDS: dict[str, str] = {
    name: field for name, _weights, field, _desc in FEATURE_DEFINITIONS
}
FEATURE_DESCRIPTIONS: dict[str, str] = {
    name: desc for name, _weights, _field, desc in FEATURE_DEFINITIONS
}


def _sequence_from_keypoints(keypoints_sequence: list[BodyKeypoints]) -> np.ndarray:
    """Build the (seq_len, 16) LSTM input the same way ``predict_video`` does (reuses ``normalize_by_body_proportion``, no new logic)."""
    return np.array([normalize_by_body_proportion(bp) for bp in keypoints_sequence], dtype=np.float32)


def load_one_sample(
    video_path: Path = DEFAULT_EXPLAIN_VIDEO,
    frames_root: Path = EXPLAIN_FRAMES_ROOT,
    pose_model: Any | None = None,
    model_path: Path = MODEL_PATH,
    label_encoder_path: Path = LABEL_ENCODER_PATH,
    thresholds_path: Path = DEFAULT_THRESHOLDS_PATH,
    n_samples: int = NUM_SAMPLED_FRAMES,
    min_confidence: float = MIN_KEYPOINT_CONFIDENCE,
) -> tuple[np.ndarray, PredictionReport, HipKneeLSTMClassifier, int]:
    """
    Load ONE real feature vector (a (seq_len, 16) keypoint sequence) plus the
    full ``PredictionReport`` for it.

    Every stage below is the SAME already-validated function
    ``src.models.hip_knee_predict.predict_video`` calls internally, imported
    unmodified; this helper only orchestrates them so the intermediate
    sequence tensor (needed by Captum) is also returned, since
    ``predict_video`` itself only returns the final report.

    Returns:
        ``(sequence, report, model, predicted_class_index)``.
    """
    with log_execution_time(logger, f"load_one_sample: {video_path.name}"):
        pose_model = pose_model or load_pose_model()
        frame_keypoint_pairs = extract_lift_frames_with_keypoints(video_path, frames_root, pose_model, n_samples, min_confidence)
        if not frame_keypoint_pairs:
            raise ValueError(f"No usable frames extracted from {video_path}")
        keypoints_sequence = [body for _, body in frame_keypoint_pairs]

        label_encoder = load_label_encoder(label_encoder_path)
        thresholds = load_thresholds(thresholds_path)
        sequence = _sequence_from_keypoints(keypoints_sequence)

        config = TrainingConfig()
        model = load_trained_model(model_path, input_size=sequence.shape[-1], num_classes=len(label_encoder.classes_), config=config)

        classification, prediction_confidence = predict_class(model, sequence)
        predicted_label = str(label_encoder.inverse_transform([classification])[0])

        metrics = analyze_lift(keypoints_sequence)
        rule_scores = compute_rule_scores(classification, metrics, thresholds)
        rom_peaks, _, _ = compute_rom_and_peaks(keypoints_sequence)

        report = build_prediction_report(
            video_path, classification, predicted_label, prediction_confidence,
            rule_scores, rom_peaks, metrics, len(keypoints_sequence),
        )

    return sequence, report, model, classification


def compute_integrated_gradients(
    model: HipKneeLSTMClassifier,
    sequence: np.ndarray,
    target_class: int,
    n_steps: int = 50,
) -> tuple[np.ndarray, float]:
    """
    Run Captum Integrated Gradients on ONE feature vector (the (seq_len, 16)
    keypoint sequence actually fed to the trained LSTM), attributing the
    predicted class's output back to every raw input feature.

    Uses a zero-vector baseline: the model's own inputs are hip-centered
    (translated so the hip midpoint is the origin) and torso-scaled, so the
    origin is already a meaningful "neutral pose" reference rather than an
    arbitrary choice.

    The model is only ever run in ``eval()`` mode here and no optimizer or
    ``backward()``-on-parameters call is made — Captum computes gradients of
    the OUTPUT with respect to the INPUT, never updates the model's weights.

    Returns:
        ``(attributions, convergence_delta)`` where ``attributions`` has
        shape ``(seq_len, 16)`` and ``convergence_delta`` is Captum's
        completeness-axiom diagnostic (should be close to 0).
    """
    model.eval()
    input_tensor = torch.tensor(sequence[np.newaxis, ...], dtype=torch.float32, requires_grad=True)
    baseline = torch.zeros_like(input_tensor)

    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(
        input_tensor, baselines=baseline, target=target_class, n_steps=n_steps, return_convergence_delta=True,
    )
    delta_value = float(delta.item())
    logger.info("Integrated Gradients convergence delta: %.6f (closer to 0 is better)", delta_value)
    return attributions.detach().numpy().squeeze(0), delta_value


def aggregate_group_attribution(attributions: np.ndarray) -> np.ndarray:
    """Reduce raw ``(seq_len, 16)`` attributions to per-frame, per-joint-group ``|attribution|`` magnitude, shape ``(seq_len, 4)``."""
    abs_attr = np.abs(attributions)
    seq_len = abs_attr.shape[0]
    per_keypoint = abs_attr.reshape(seq_len, 8, 2).sum(axis=2)  # (seq_len, 8)

    group_matrix = np.zeros((seq_len, len(JOINT_GROUPS)), dtype=np.float64)
    for keypoint_idx, group_idx in enumerate(_KEYPOINT_TO_GROUP_INDEX):
        group_matrix[:, group_idx] += per_keypoint[:, keypoint_idx]
    return group_matrix


def compute_feature_importance(group_matrix: np.ndarray) -> dict[str, float]:
    """
    Map raw per-joint-group Integrated Gradients attribution onto the
    requested named, human-readable features using ``FEATURE_GROUP_WEIGHTS``,
    then normalize the result so all values sum to 1.0.
    """
    group_totals = group_matrix.sum(axis=0)  # (4,)
    raw_importance = {
        name: float(np.dot(group_totals, weights))
        for name, weights in FEATURE_GROUP_WEIGHTS.items()
    }
    total = sum(raw_importance.values())
    if total <= 0:
        n = len(raw_importance)
        return {name: 1.0 / n for name in raw_importance}
    return {name: value / total for name, value in raw_importance.items()}


def save_feature_importance_csv(
    importance: dict[str, float],
    report: PredictionReport,
    path: Path = FEATURE_IMPORTANCE_CSV_PATH,
) -> None:
    """Write the normalized feature importance table to CSV, sorted most-important first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    report_dict = report.to_dict()
    rows = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "normalized_importance", "importance_pct", "source_field", "source_value", "description"])
        for name, value in rows:
            source_field = FEATURE_SOURCE_FIELDS[name]
            source_value = report_dict.get(source_field, "")
            writer.writerow([name, f"{value:.6f}", f"{value * 100:.2f}", source_field, source_value, FEATURE_DESCRIPTIONS[name]])

    logger.info("Saved feature importance CSV -> %s", path)


def save_integrated_gradients_plot(
    group_matrix: np.ndarray,
    importance: dict[str, float],
    report: PredictionReport,
    path: Path = INTEGRATED_GRADIENTS_PNG_PATH,
) -> None:
    """Save a two-panel figure: raw IG attribution heatmap (time x joint group) + normalized feature-importance bar chart."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    im = ax1.imshow(group_matrix.T, aspect="auto", cmap="viridis")
    ax1.set_yticks(range(len(JOINT_GROUPS)))
    ax1.set_yticklabels([g.capitalize() for g in JOINT_GROUPS])
    ax1.set_xlabel("Sampled frame index")
    ax1.set_title("Integrated Gradients |attribution|\nby joint group over time")
    fig.colorbar(im, ax=ax1, label="|IG attribution|")

    names_sorted = sorted(importance.keys(), key=lambda k: importance[k])
    values = [importance[k] * 100 for k in names_sorted]
    ax2.barh(names_sorted, values, color="tab:blue")
    ax2.set_xlabel("Normalized importance (%)")
    ax2.set_title(f"Feature importance\npredicted: {report.predicted_class} ({report.prediction_confidence * 100:.1f}%)")
    for i, v in enumerate(values):
        ax2.text(v, i, f" {v:.1f}%", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved Integrated Gradients plot -> %s", path)


def write_explainability_report(
    video_path: Path,
    report: PredictionReport,
    importance: dict[str, float],
    convergence_delta: float,
    n_steps: int,
    path: Path = MODEL_EXPLAINABILITY_MD_PATH,
) -> None:
    """Write the full explainability write-up (method, results, mapping methodology, limitations)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    report_dict = report.to_dict()

    weight_table_lines = "\n".join(
        f"| {name} | {w[0]:.3f} | {w[1]:.3f} | {w[2]:.3f} | {w[3]:.3f} |"
        for name, w in FEATURE_GROUP_WEIGHTS.items()
    )
    importance_table_lines = "\n".join(
        f"| {name} | {value * 100:.2f}% | `{FEATURE_SOURCE_FIELDS[name]}` = {report_dict.get(FEATURE_SOURCE_FIELDS[name], 'n/a')} | {FEATURE_DESCRIPTIONS[name]} |"
        for name, value in rows
    )

    content = f"""# Hip & Knee — Model Explainability Report (Captum Integrated Gradients)

## 1. Purpose

This report explains ONE prediction made by the already-trained, already
validated Hip & Knee LSTM classifier (`models/hip_knee_lstm.pth`), using
[Captum](https://captum.ai/)'s Integrated Gradients (IG) attribution method.

**No retraining was performed. No prediction logic was modified. No
teammate module was touched.** The model is loaded read-only and run only
in `eval()` mode; Captum computes gradients of the model's OUTPUT with
respect to its INPUT — it never updates the model's weights.

## 2. Sample explained

| Field | Value |
|---|---|
| Video | `{video_path}` |
| Frames used | {report.n_frames_used} |
| Predicted class | **{report.predicted_class}** |
| Prediction confidence | {report.prediction_confidence * 100:.2f}% |
| Low-confidence flag | {report.is_low_confidence} |

## 3. Method

1. **Load the trained LSTM** — `HipKneeLSTMClassifier`, rebuilt with the
   training-time architecture and its saved `state_dict`
   (`src.models.hip_knee_evaluate.load_trained_model`, the same loader used
   for evaluation/prediction). Weights are frozen/read-only throughout.
2. **Load one feature vector** — the real `(seq_len=20, 16)` normalized
   keypoint sequence produced for the video above, built by reusing the
   exact same pipeline stages `predict_video` uses internally: frame
   extraction, YOLO11-pose keypoint extraction, and
   `normalize_by_body_proportion` (hip-centered, torso-scaled). This is the
   literal tensor the LSTM consumes in production — nothing is re-derived
   or approximated for this step.
3. **Compute Integrated Gradients** — `captum.attr.IntegratedGradients`
   attributes the predicted class's output logit back to every one of the
   16 raw input features at every one of the {report.n_frames_used} sampled
   frames, integrating gradients along a straight-line path from a
   zero-vector baseline (a neutral "hip-centered" pose, since the model's
   inputs are already centered on the hip midpoint) to the real input, using
   `n_steps={n_steps}`.
   - Convergence delta (completeness-axiom diagnostic, closer to 0 is
     better): **{convergence_delta:.6f}**
4. **Produce feature importance** — the raw `(20, 16)` attribution tensor is
   reduced to 4 anatomical joint groups (shoulder, hip, knee, ankle,
   bilateral-combined) by summing `|attribution|` over the x/y components and
   all sampled frames. These 4 joint-group totals are then mapped onto the
   11 requested, human-readable features using the documented weight table
   in §4, and the result is **normalized to sum to 1.0**.

## 4. Anatomical / functional attribution mapping

The 16 raw LSTM input dimensions per frame are 8 bilateral keypoints
(shoulder, hip, knee, ankle) x (x, y). Rather than guessing, each requested
feature's mapping weight is derived directly from the REAL formulas already
used elsewhere in this codebase (`src/features/hip_knee_biomechanics.py`):

- **Hip ROM / Hip Peak** come from the hip flexion angle,
  `angle(shoulder, hip, knee)` -> weighted toward shoulder+hip+knee.
- **Knee ROM / Knee Peak** come from the knee flexion angle,
  `angle(hip, knee, ankle)` -> weighted toward hip+knee+ankle.
- **Sequential Delay** (Rule A), **Synchronization** (Rule C score),
  **Hip Dominance** (Rule B score), **Correlation** (Rule C raw
  coefficient) and **RFD** (Rule D) are all derived from BOTH the hip-angle
  AND knee-angle velocity/acceleration signals together -> weighted equally
  across hip and knee (with a smaller shoulder/ankle contribution, since
  those angles are the vertices/endpoints of the hip and knee angle
  definitions).
- **Confidence** is the LSTM's own softmax confidence — not anatomically
  localized to one joint, so it is weighted uniformly across all 4 groups
  (a global-sensitivity proxy).
- **Anthropometric Features** come from `leg_length` (hip->knee->ankle) and
  `torso_length` (shoulder->hip), the SAME reference lengths the pipeline
  already uses to normalize distance-dependent features
  (`reports/anthropometric_normalization.md`).

| Feature | Shoulder | Hip | Knee | Ankle |
|---|---|---|---|---|
{weight_table_lines}

**This mapping is a documented heuristic used only to translate raw,
per-keypoint Integrated Gradients attribution into human-readable
categories.** It does not alter the model, its inputs, or its prediction —
see §6 for its limitations.

## 5. Feature importance (normalized, sums to 100%)

| Feature | Normalized importance | Source value (this sample) | Description |
|---|---|---|---|
{importance_table_lines}

See `reports/feature_importance.csv` for the same table in machine-readable
form, and `reports/integrated_gradients.png` for the attribution heatmap
(time x joint group) and the feature-importance bar chart.

## 6. Limitations

1. **Single-sample explanation** — this report explains one specific video's
   prediction, not the model's general/global behaviour. Re-running against
   a different video (`--video <path>`) can and should give a different
   feature-importance ranking.
2. **Anatomical mapping is a heuristic, not a second model** — Integrated
   Gradients is computed rigorously on the model's real raw input tensor;
   the subsequent grouping of those 16 raw-dimension attributions into the
   11 requested named features (§4) is a documented, deterministic weighting
   scheme based on which joints each metric's real formula depends on, not
   an independent attribution computation for each named feature. Rule
   A-D-derived features (Sequential Delay, Synchronization, Hip Dominance,
   Correlation, RFD) share very similar weights because they are all
   functions of the same two underlying hip/knee angular signals.
3. **Zero-vector baseline** — a neutral, hip-centered "zero pose" was used
   as the IG baseline. A different baseline (e.g. the dataset mean pose)
   would shift the absolute attribution magnitudes, though the *relative*
   ranking across joint groups is expected to be similar for this kind of
   pose-classification model.
4. **Confidence is not spatially localized** — unlike the other 10
   features, "Confidence" has no direct anatomical dependency in the
   biomechanics formulas, so it is assigned a uniform weight across all 4
   joint groups as a global-sensitivity proxy rather than a joint-specific one.
5. **No retraining, no logic changes** — this module only reads existing
   model/threshold/label-encoder artifacts and calls existing, unmodified
   pipeline functions; it does not change `src/models/hip_knee_predict.py`,
   `src/models/hip_knee_lstm.py`, or any teammate module.

## 7. Files generated

- `reports/model_explainability.md` — this report.
- `reports/integrated_gradients.png` — attribution heatmap + feature-importance bar chart.
- `reports/feature_importance.csv` — normalized feature importance table.
"""
    path.write_text(content, encoding="utf-8")
    logger.info("Saved model explainability report -> %s", path)


def explain_prediction(
    video_path: Path = DEFAULT_EXPLAIN_VIDEO,
    frames_root: Path = EXPLAIN_FRAMES_ROOT,
    n_steps: int = 50,
) -> dict[str, Any]:
    """Run the full explainability pipeline end-to-end for one video and write all 3 report artifacts."""
    sequence, report, model, classification = load_one_sample(video_path=video_path, frames_root=frames_root)
    attributions, convergence_delta = compute_integrated_gradients(model, sequence, classification, n_steps=n_steps)
    group_matrix = aggregate_group_attribution(attributions)
    importance = compute_feature_importance(group_matrix)

    save_feature_importance_csv(importance, report)
    save_integrated_gradients_plot(group_matrix, importance, report)
    write_explainability_report(video_path, report, importance, convergence_delta, n_steps)

    logger.info(
        "explain_prediction complete | video=%s predicted_class=%s confidence=%.4f top_feature=%s",
        video_path.name, report.predicted_class, report.prediction_confidence,
        max(importance, key=importance.get),
    )
    return {
        "report": report,
        "importance": importance,
        "convergence_delta": convergence_delta,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain one Hip & Knee LSTM prediction using Captum Integrated Gradients.")
    parser.add_argument("--video", type=Path, default=DEFAULT_EXPLAIN_VIDEO, help="Video to explain.")
    parser.add_argument("--frames-root", type=Path, default=EXPLAIN_FRAMES_ROOT)
    parser.add_argument("--n-steps", type=int, default=50, help="Integrated Gradients integration steps.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = explain_prediction(video_path=args.video, frames_root=args.frames_root, n_steps=args.n_steps)
    print("=" * 60)
    print("HIP & KNEE — EXPLAINABILITY (Captum Integrated Gradients)")
    print("=" * 60)
    print(f"Video                : {args.video.name}")
    print(f"Predicted class      : {result['report'].predicted_class}")
    print(f"Confidence           : {result['report'].prediction_confidence * 100:.2f}%")
    print(f"IG convergence delta : {result['convergence_delta']:.6f}")
    print()
    for name, value in sorted(result["importance"].items(), key=lambda kv: kv[1], reverse=True):
        print(f"{name:<25s}: {value * 100:6.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
