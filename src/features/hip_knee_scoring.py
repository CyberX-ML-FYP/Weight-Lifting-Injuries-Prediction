"""
Hip & Knee Analysis — Adaptive thresholds and confidence-weighted scoring.

WHY these improvements help Hip & Knee analysis
------------------------------------------------
- Adaptive thresholds: the original notebook hardcoded the good/poor Rule
  A-D min/max ranges as a literal tuple pasted from one training run.
  ``learn_adaptive_thresholds`` recomputes those ranges from the current
  training dataset and persists them to JSON, so thresholds automatically
  stay in sync whenever the model is retrained on new footage instead of
  silently drifting out of date.
- Confidence-weighted scoring: a lift analysed from mostly low-confidence
  keypoints (e.g. partial occlusion, motion blur) produces noisier Rule A-D
  values. Multiplying the raw score by the clip's mean keypoint confidence
  discounts the influence of unreliable detections on the final performance
  score, rather than trusting every prediction equally.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.features.hip_knee_biomechanics import LiftMetrics
from src.features.hip_knee_config import RULE_WEIGHTS
from src.features.hip_knee_logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RuleRange:
    """Min/max range observed for one Rule (A/B/C/D) over a set of lifts."""

    min_value: float
    max_value: float


@dataclass(frozen=True)
class AdaptiveThresholds:
    """
    Learned good/poor Rule A-D ranges, replacing the notebook's hardcoded
    ``ranges`` tuple. Persisted to JSON as part of the saved model artifacts.
    """

    good_a: RuleRange
    good_b: RuleRange
    good_c: RuleRange
    good_d: RuleRange
    poor_a: RuleRange
    poor_b: RuleRange
    poor_c: RuleRange
    poor_d: RuleRange
    n_good_samples: int
    n_poor_samples: int
    learned_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AdaptiveThresholds":
        def _range(key: str) -> RuleRange:
            return RuleRange(**data[key])

        return cls(
            good_a=_range("good_a"), good_b=_range("good_b"),
            good_c=_range("good_c"), good_d=_range("good_d"),
            poor_a=_range("poor_a"), poor_b=_range("poor_b"),
            poor_c=_range("poor_c"), poor_d=_range("poor_d"),
            n_good_samples=data["n_good_samples"],
            n_poor_samples=data["n_poor_samples"],
            learned_at=data.get("learned_at", ""),
        )


def _range_of(values: list[float]) -> RuleRange:
    if not values:
        return RuleRange(min_value=0.0, max_value=1.0)
    return RuleRange(min_value=float(np.min(values)), max_value=float(np.max(values)))


def learn_adaptive_thresholds(
    good_metrics: list[LiftMetrics],
    poor_metrics: list[LiftMetrics],
) -> AdaptiveThresholds:
    """
    Learn Rule A-D good/poor ranges from labelled training lifts.

    Args:
        good_metrics: ``LiftMetrics`` computed for every "Good" training lift.
        poor_metrics: ``LiftMetrics`` computed for every "Poor" training lift.

    Returns:
        An ``AdaptiveThresholds`` instance capturing the observed min/max
        range for each rule, for both classes.
    """
    thresholds = AdaptiveThresholds(
        good_a=_range_of([m.delay for m in good_metrics]),
        good_b=_range_of([m.hip_dominance_ratio for m in good_metrics]),
        good_c=_range_of([m.synchronization_corr for m in good_metrics]),
        good_d=_range_of([m.rfd_ratio for m in good_metrics]),
        poor_a=_range_of([m.delay for m in poor_metrics]),
        poor_b=_range_of([m.hip_dominance_ratio for m in poor_metrics]),
        poor_c=_range_of([m.synchronization_corr for m in poor_metrics]),
        poor_d=_range_of([m.rfd_ratio for m in poor_metrics]),
        n_good_samples=len(good_metrics),
        n_poor_samples=len(poor_metrics),
    )
    logger.info(
        "learn_adaptive_thresholds | good_n=%d poor_n=%d",
        thresholds.n_good_samples, thresholds.n_poor_samples,
    )
    return thresholds


def save_thresholds(thresholds: AdaptiveThresholds, path: Path) -> None:
    """Persist learned thresholds (normalization parameters) to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(thresholds.to_dict(), indent=2), encoding="utf-8")
    logger.info("Saved adaptive thresholds -> %s", path)


def load_thresholds(path: Path) -> AdaptiveThresholds:
    """Load previously learned thresholds from JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return AdaptiveThresholds.from_dict(data)


def normalize_score(value: float, min_val: float, max_val: float) -> float:
    """Min-max normalize ``value`` into a 0-100 score (same as the notebook)."""
    if max_val == min_val:
        return 0.0
    score = (value - min_val) / (max_val - min_val)
    score = float(np.clip(score, 0.0, 1.0))
    return score * 100.0


def compute_performance_score(
    classification: int,
    metrics: LiftMetrics,
    thresholds: AdaptiveThresholds,
    weights: tuple[float, float, float, float] = RULE_WEIGHTS,
    confidence_weight: float | None = None,
) -> float:
    """
    Compute the final 0-100 performance score for a lift.

    Same weighting/blending logic as the original notebook's
    ``get_performance_score`` (0.35/0.30/0.20/0.15 weighted blend of the
    normalized Rule A-D scores, mapped to 0-50 for "Poor" classifications
    and 50-100 for "Good" classifications) with one addition: the raw
    weighted score is scaled by ``confidence_weight`` (defaults to the
    clip's mean keypoint confidence) so low-confidence detections pull the
    score toward a neutral midpoint rather than being trusted fully.

    Args:
        classification: ``0`` = Good, ``1`` = Poor (matches the notebook's
            ``LabelEncoder`` convention).
        metrics: Rule A-D metrics for the lift being scored.
        thresholds: Learned good/poor ranges used for normalization.
        weights: Rule A/B/C/D blend weights.
        confidence_weight: Optional override in ``[0, 1]``; defaults to
            ``metrics.mean_confidence``.

    Returns:
        Final performance score in ``[0, 100]``.
    """
    weight_a, weight_b, weight_c, weight_d = weights
    conf = confidence_weight if confidence_weight is not None else metrics.mean_confidence
    conf = float(np.clip(conf, 0.0, 1.0))

    if classification == 0:
        a_score = normalize_score(metrics.delay, thresholds.good_a.min_value, thresholds.good_a.max_value)
        b_score = normalize_score(metrics.hip_dominance_ratio, thresholds.good_b.min_value, thresholds.good_b.max_value)
        c_score = normalize_score(metrics.synchronization_corr, thresholds.good_c.min_value, thresholds.good_c.max_value)
        d_score = normalize_score(metrics.rfd_ratio, thresholds.good_d.min_value, thresholds.good_d.max_value)
        raw_score = weight_a * a_score + weight_b * b_score + weight_c * c_score + weight_d * d_score
        raw_score *= conf
        final_score = 50.0 + raw_score * 0.5
    else:
        a_score = normalize_score(metrics.delay, thresholds.poor_a.min_value, thresholds.poor_a.max_value)
        b_score = normalize_score(metrics.hip_dominance_ratio, thresholds.poor_b.min_value, thresholds.poor_b.max_value)
        c_score = normalize_score(metrics.synchronization_corr, thresholds.poor_c.min_value, thresholds.poor_c.max_value)
        d_score = normalize_score(metrics.rfd_ratio, thresholds.poor_d.min_value, thresholds.poor_d.max_value)
        raw_score = weight_a * a_score + weight_b * b_score + weight_c * c_score + weight_d * d_score
        raw_score *= conf
        final_score = raw_score * 0.5

    logger.info(
        "compute_performance_score | classification=%d confidence=%.3f final_score=%.2f",
        classification, conf, final_score,
    )
    return final_score


def compute_confidence_weighted_score(
    rule_scores: dict[str, float],
    rule_confidences: dict[str, float],
    weights: tuple[float, float, float, float] = RULE_WEIGHTS,
) -> float:
    """
    Blend already-normalized (0-100) Rule A-D scores into one final score,
    weighting each rule's contribution by BOTH its fixed importance weight
    (``weights``) and its own confidence (``rule_confidences``) — see
    ``reports/confidence_weighted_rules.md`` for the full algorithm. A
    low-confidence rule contributes less to the final score; a
    high-confidence rule contributes (up to) its full nominal weight.

    This is additive to, and does not replace, ``compute_performance_score``
    (kept unchanged for backward compatibility).

    Args:
        rule_scores: ``{"rule_a_score": ..., "rule_b_score": ..., ...}``,
            each already in ``[0, 100]``.
        rule_confidences: ``{"rule_a_confidence": ..., ...}``, each in
            ``[0, 1]`` (see ``LiftMetrics``/``compute_rule_confidences``).
        weights: Rule A/B/C/D importance weights (same as
            ``compute_performance_score``).

    Returns:
        Confidence-weighted final score in ``[0, 100]``. Falls back to an
        unweighted blend if every rule confidence is (near) zero, to avoid
        dividing by zero.
    """
    score_keys = ("rule_a_score", "rule_b_score", "rule_c_score", "rule_d_score")
    conf_keys = ("rule_a_confidence", "rule_b_confidence", "rule_c_confidence", "rule_d_confidence")

    effective_weights = [w * float(np.clip(rule_confidences[ck], 0.0, 1.0)) for w, ck in zip(weights, conf_keys)]
    total_effective_weight = sum(effective_weights)

    if total_effective_weight < 1e-9:
        total_weight = sum(weights)
        return sum(w * rule_scores[k] for w, k in zip(weights, score_keys)) / total_weight

    weighted_sum = sum(ew * rule_scores[k] for ew, k in zip(effective_weights, score_keys))
    return weighted_sum / total_effective_weight
