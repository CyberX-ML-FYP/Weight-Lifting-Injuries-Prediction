"""
Hip & Knee Analysis — Biomechanical angle/velocity/acceleration + Rule A-D.

WHY these improvements help Hip & Knee analysis
------------------------------------------------
- Rule A (Sequential Extension), Rule B (Hip Dominance), Rule C
  (Synchronization) and Rule D (Rate-of-Force-Development approximation)
  keep the EXACT same definitions as the original research notebook — only
  the surrounding code is hardened:
    * bilateral averaging (left + right side) instead of left-side-only,
      which reduces noise when one side is briefly occluded;
    * guards against divide-by-zero / constant-signal edge cases that used
      to throw inside ``pearsonr``/ratio calculations on short clips;
    * optional temporal smoothing before differentiating, since raw pose
      jitter is amplified by ``np.gradient`` (a first derivative) and
      amplified again by acceleration (a second derivative).
- A ``LiftMetrics`` dataclass replaces the notebook's bare ``(A, B, C, D)``
  tuple so downstream code (scoring, evaluation, logging) is self-documenting
  and type-checked instead of relying on positional order.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import pearsonr

from src.features.hip_knee_config import DEFAULT_FPS, RULE_WEIGHTS
from src.features.hip_knee_logger import get_logger
from src.features.hip_knee_pose_utils import BodyKeypoints, smooth_angle_sequence

logger = get_logger(__name__)


@dataclass(frozen=True)
class AnthropometricReference:
    """
    Clip-level reference lengths (pixels) used to normalize distance-based
    features so they are comparable across athletes of different heights
    and camera distances. See ``reports/anthropometric_normalization.md``
    for the full derivation.
    """

    leg_length: float       # thigh + shank segment length (hip->knee->ankle)
    torso_length: float     # shoulder-midpoint to hip-midpoint distance
    shoulder_width: float   # left-shoulder to right-shoulder distance
    primary_reference: float  # the reference length used for normalization (= leg_length)


@dataclass(frozen=True)
class LiftMetrics:
    """Biomechanical metrics for one lift, computed by ``analyze_lift``."""

    delay: float             # Rule A — sequential extension delay (seconds)
    hip_dominance_ratio: float  # Rule B
    synchronization_corr: float  # Rule C
    rfd_ratio: float          # Rule D
    mean_confidence: float    # Average keypoint confidence across the clip

    # ── Anthropometric normalization (see reports/anthropometric_normalization.md) ──
    # Defaulted (rather than required) so older call sites that only know
    # about the original 5 Rule A-D fields (e.g. thresholds learned from the
    # flat master CSV in hip_knee_build_dataset.py, which has no keypoints
    # to derive reference lengths from) keep working unmodified.
    reference: AnthropometricReference = field(default_factory=lambda: AnthropometricReference(0.0, 0.0, 0.0, 0.0))
    hip_linear_rom_px: float = 0.0          # raw vertical hip-joint travel (pixels)
    hip_linear_rom_normalized: float = 0.0  # same, divided by reference.primary_reference
    knee_linear_rom_px: float = 0.0         # raw vertical knee-joint travel (pixels)
    knee_linear_rom_normalized: float = 0.0  # same, divided by reference.primary_reference
    hip_peak_linear_velocity_px_s: float = 0.0          # raw peak vertical hip speed (px/s)
    hip_peak_linear_velocity_normalized: float = 0.0    # same, in reference-lengths/s
    knee_peak_linear_velocity_px_s: float = 0.0         # raw peak vertical knee speed (px/s)
    knee_peak_linear_velocity_normalized: float = 0.0   # same, in reference-lengths/s

    # ── Confidence-weighted Rule A-D scoring (see reports/confidence_weighted_rules.md) ──
    # Defaulted for the same backward-compatibility reason as the block above.
    rule_a_confidence: float = 0.0   # confidence of the frame(s) Rule A's peak-timing depends on
    rule_b_confidence: float = 0.0   # confidence of the frame(s) Rule B's peak-magnitude depends on
    rule_c_confidence: float = 0.0   # confidence across the WHOLE clip (Rule C uses the full signal)
    rule_d_confidence: float = 0.0   # confidence of the frame(s) Rule D's peak-acceleration depends on
    overall_confidence: float = 0.0  # RULE_WEIGHTS-weighted blend of the 4 rule confidences


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Return the angle ABC (in degrees) at vertex ``b``.

    Identical formula to the original notebook/arm-module implementation:
    θ = cos⁻¹((BA · BC) / (|BA| · |BC|)).
    """
    a, b, c = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), np.asarray(c, dtype=np.float64)
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return 0.0
    cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def extract_hip_knee_angles(
    frames: Sequence[BodyKeypoints],
    bilateral: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute hip and knee flexion angles across a sequence of frames.

    Args:
        frames: Per-frame extracted lower-body keypoints.
        bilateral: If ``True``, average the left and right side angle for
            each frame (more robust to single-side occlusion). If
            ``False``, use the left side only, matching the original
            notebook exactly.

    Returns:
        ``(hip_angles, knee_angles)`` as 1-D arrays, one value per frame.
    """
    hip_angles: list[float] = []
    knee_angles: list[float] = []

    for kp in frames:
        left_hip_angle = calculate_angle(kp.left_shoulder, kp.left_hip, kp.left_knee)
        left_knee_angle = calculate_angle(kp.left_hip, kp.left_knee, kp.left_ankle)

        if bilateral:
            right_hip_angle = calculate_angle(kp.right_shoulder, kp.right_hip, kp.right_knee)
            right_knee_angle = calculate_angle(kp.right_hip, kp.right_knee, kp.right_ankle)
            hip_angles.append((left_hip_angle + right_hip_angle) / 2.0)
            knee_angles.append((left_knee_angle + right_knee_angle) / 2.0)
        else:
            hip_angles.append(left_hip_angle)
            knee_angles.append(left_knee_angle)

    return np.array(hip_angles), np.array(knee_angles)


def _dist(p: np.ndarray, q: np.ndarray) -> float:
    """Euclidean distance between two 2-D keypoints (pixels)."""
    return float(np.linalg.norm(np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64)))


def compute_reference_lengths(frames: Sequence[BodyKeypoints]) -> AnthropometricReference:
    """
    Compute clip-level anthropometric reference lengths (in pixels).

    For each frame, computes:
      - leg_length    = |hip->knee| + |knee->ankle| (bilateral average), i.e.
                        the thigh + shank segment lengths, which track true
                        limb length better than a straight hip-ankle line
                        (which shortens whenever the knee is bent).
      - torso_length  = |shoulder_midpoint -> hip_midpoint|
      - shoulder_width = |left_shoulder -> right_shoulder|

    The per-frame values are then reduced with the MEDIAN across the clip
    (robust to single-frame pose jitter/occlusion, unlike the mean). The
    median leg_length is used as ``primary_reference`` for normalizing
    distance-dependent features — see
    ``reports/anthropometric_normalization.md`` for the rationale.

    Returns an all-zero ``AnthropometricReference`` if ``frames`` is empty.
    """
    if not frames:
        return AnthropometricReference(0.0, 0.0, 0.0, 0.0)

    leg_lengths: list[float] = []
    torso_lengths: list[float] = []
    shoulder_widths: list[float] = []

    for kp in frames:
        left_leg = _dist(kp.left_hip, kp.left_knee) + _dist(kp.left_knee, kp.left_ankle)
        right_leg = _dist(kp.right_hip, kp.right_knee) + _dist(kp.right_knee, kp.right_ankle)
        leg_lengths.append((left_leg + right_leg) / 2.0)

        hip_mid = (kp.left_hip + kp.right_hip) / 2.0
        shoulder_mid = (kp.left_shoulder + kp.right_shoulder) / 2.0
        torso_lengths.append(_dist(shoulder_mid, hip_mid))

        shoulder_widths.append(_dist(kp.left_shoulder, kp.right_shoulder))

    leg_length = float(np.median(leg_lengths))
    torso_length = float(np.median(torso_lengths))
    shoulder_width = float(np.median(shoulder_widths))

    return AnthropometricReference(
        leg_length=leg_length,
        torso_length=torso_length,
        shoulder_width=shoulder_width,
        primary_reference=leg_length,
    )


def normalize_distance(value: float, reference_length: float) -> float:
    """
    Normalize a raw pixel-space distance/velocity by an anthropometric
    reference length, guarding against a zero/degenerate reference.

    Returns ``value`` unchanged (a safe no-op) if ``reference_length`` is
    too small to divide by, instead of raising or returning ``inf``/``nan``.
    """
    if reference_length < 1e-6:
        return float(value)
    return float(value) / reference_length


def compute_linear_rom(frames: Sequence[BodyKeypoints], joint: str, reference_length: float) -> tuple[float, float]:
    """
    Linear (pixel-space) range of motion of a joint's vertical trajectory.

    Unlike the angular Hip/Knee ROM (``extract_hip_knee_angles``, already
    scale-invariant by construction), the joint's raw vertical travel in
    pixels IS distance-dependent — it grows with camera zoom/proximity and
    with athlete height. This function returns both the raw pixel ROM and
    the anthropometrically normalized ROM (in reference-lengths), per
    ``reports/anthropometric_normalization.md``.

    Args:
        frames: Per-frame extracted keypoints for one lift.
        joint: ``"hip"`` or ``"knee"``.
        reference_length: Anthropometric reference length (pixels) to
            normalize by, e.g. ``AnthropometricReference.primary_reference``.

    Returns:
        ``(raw_rom_px, normalized_rom)``.
    """
    if not frames:
        return 0.0, 0.0

    if joint == "hip":
        positions = np.array([((kp.left_hip[1] + kp.right_hip[1]) / 2.0) for kp in frames])
    elif joint == "knee":
        positions = np.array([((kp.left_knee[1] + kp.right_knee[1]) / 2.0) for kp in frames])
    else:
        raise ValueError(f"Unknown joint '{joint}' (expected 'hip' or 'knee')")

    raw_rom = float(np.max(positions) - np.min(positions))
    return raw_rom, normalize_distance(raw_rom, reference_length)


def compute_linear_velocity(
    frames: Sequence[BodyKeypoints],
    joint: str,
    reference_length: float,
    fps: int = DEFAULT_FPS,
) -> tuple[float, float]:
    """
    Peak linear (pixel-space) vertical velocity of a joint, raw and
    anthropometrically normalized.

    Like ``compute_linear_rom``, this is genuinely distance-dependent
    (pixels/second) and benefits from normalization, unlike the existing
    angular velocity used by Rule A-D (degrees/second, already
    scale-invariant).

    Returns:
        ``(peak_raw_velocity_px_s, peak_normalized_velocity)``.
    """
    if len(frames) < 2:
        return 0.0, 0.0

    if joint == "hip":
        positions = np.array([((kp.left_hip[1] + kp.right_hip[1]) / 2.0) for kp in frames])
    elif joint == "knee":
        positions = np.array([((kp.left_knee[1] + kp.right_knee[1]) / 2.0) for kp in frames])
    else:
        raise ValueError(f"Unknown joint '{joint}' (expected 'hip' or 'knee')")

    velocity, _ = compute_derivatives(positions, fps)
    peak_raw = float(np.max(np.abs(velocity))) if len(velocity) else 0.0
    return peak_raw, normalize_distance(peak_raw, reference_length)


def _confidence_at(frame_confidences: np.ndarray, indices: Sequence[int]) -> float:
    """Mean pose confidence at a specific set of frame indices (clipped to valid range)."""
    if frame_confidences.size == 0:
        return 0.0
    valid = [i for i in indices if 0 <= i < frame_confidences.size]
    if not valid:
        return float(np.mean(frame_confidences))
    return float(np.mean(frame_confidences[valid]))


def compute_rule_confidences(
    frames: Sequence[BodyKeypoints],
    hip_velocity: np.ndarray,
    knee_velocity: np.ndarray,
    hip_acceleration: np.ndarray,
    knee_acceleration: np.ndarray,
) -> dict[str, float]:
    """
    Per-rule confidence values, grounded in the specific frame(s) each rule
    actually depends on (rather than one blanket clip-wide confidence for
    every rule). See ``reports/confidence_weighted_rules.md`` for the full
    algorithm/rationale.

    - Rule A (sequential extension delay) depends only on the *timing* of
      the knee-velocity and hip-velocity peaks -> confidence is the mean
      pose confidence at those two specific frames.
    - Rule B (hip dominance ratio) depends only on the *peak magnitude*
      frames of hip/knee velocity -> confidence at those two frames.
    - Rule C (synchronization correlation) uses the *entire* velocity
      signal -> confidence is the mean pose confidence across ALL frames.
    - Rule D (RFD ratio) depends only on the *peak magnitude* frames of
      hip/knee acceleration -> confidence at those two frames.

    Returns a dict with keys ``rule_a_confidence`` .. ``rule_d_confidence``,
    each in ``[0, 1]``. Returns all-zero if ``frames`` is empty.
    """
    frame_confidences = np.array([kp.mean_confidence for kp in frames]) if frames else np.array([])
    if frame_confidences.size == 0:
        return {"rule_a_confidence": 0.0, "rule_b_confidence": 0.0, "rule_c_confidence": 0.0, "rule_d_confidence": 0.0}

    knee_peak_idx = int(np.argmax(knee_velocity)) if len(knee_velocity) else 0
    hip_peak_idx = int(np.argmax(hip_velocity)) if len(hip_velocity) else 0
    hip_abs_peak_idx = int(np.argmax(np.abs(hip_velocity))) if len(hip_velocity) else 0
    knee_abs_peak_idx = int(np.argmax(np.abs(knee_velocity))) if len(knee_velocity) else 0
    hip_accel_peak_idx = int(np.argmax(np.abs(hip_acceleration))) if len(hip_acceleration) else 0
    knee_accel_peak_idx = int(np.argmax(np.abs(knee_acceleration))) if len(knee_acceleration) else 0

    return {
        "rule_a_confidence": _confidence_at(frame_confidences, [knee_peak_idx, hip_peak_idx]),
        "rule_b_confidence": _confidence_at(frame_confidences, [hip_abs_peak_idx, knee_abs_peak_idx]),
        "rule_c_confidence": float(np.mean(frame_confidences)),
        "rule_d_confidence": _confidence_at(frame_confidences, [hip_accel_peak_idx, knee_accel_peak_idx]),
    }


def compute_overall_confidence(
    rule_confidences: dict[str, float],
    weights: tuple[float, float, float, float] = RULE_WEIGHTS,
) -> float:
    """
    Blend the 4 per-rule confidences into one overall confidence value,
    using the SAME ``RULE_WEIGHTS`` importance weights already used to blend
    the Rule A-D scores themselves (0.35/0.30/0.20/0.15), so a rule that
    matters more to the final decision also matters more to the overall
    confidence.
    """
    weight_a, weight_b, weight_c, weight_d = weights
    return (
        weight_a * rule_confidences["rule_a_confidence"]
        + weight_b * rule_confidences["rule_b_confidence"]
        + weight_c * rule_confidences["rule_c_confidence"]
        + weight_d * rule_confidences["rule_d_confidence"]
    )


def compute_derivatives(signal: np.ndarray, fps: int = DEFAULT_FPS) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(velocity, acceleration)`` of a signal sampled at ``fps``."""
    if len(signal) < 2:
        zeros = np.zeros_like(signal)
        return zeros, zeros
    dt = 1.0 / fps
    velocity = np.gradient(signal, dt)
    acceleration = np.gradient(velocity, dt)
    return velocity, acceleration


def rule_a_sequential_extension(knee_velocity: np.ndarray, hip_velocity: np.ndarray, fps: int = DEFAULT_FPS) -> float:
    """
    Rule A — timing delay (seconds) between peak knee and peak hip velocity.

    Same definition as the original notebook (positive delay = hip peaks
    after knee peaks, the expected "hip after knee" extension sequence).
    """
    if len(knee_velocity) == 0 or len(hip_velocity) == 0:
        return 0.0
    knee_peak = int(np.argmax(knee_velocity))
    hip_peak = int(np.argmax(hip_velocity))
    return (hip_peak - knee_peak) / fps


def rule_b_hip_dominance(knee_velocity: np.ndarray, hip_velocity: np.ndarray) -> float:
    """Rule B — ratio of peak hip angular velocity to peak knee angular velocity."""
    hip_power = float(np.max(np.abs(hip_velocity))) if len(hip_velocity) else 0.0
    knee_power = float(np.max(np.abs(knee_velocity))) if len(knee_velocity) else 0.0
    return hip_power / (knee_power + 1e-6)


def rule_c_synchronization(knee_velocity: np.ndarray, hip_velocity: np.ndarray) -> float:
    """
    Rule C — Pearson correlation between knee and hip angular velocity.

    Guards against ``pearsonr`` raising on constant (zero-variance) signals,
    which the original notebook did not handle.
    """
    if len(knee_velocity) < 2 or len(hip_velocity) < 2:
        return 0.0
    if np.std(knee_velocity) < 1e-9 or np.std(hip_velocity) < 1e-9:
        return 0.0
    corr, _ = pearsonr(knee_velocity, hip_velocity)
    return float(corr)


def rule_d_rfd(knee_acceleration: np.ndarray, hip_acceleration: np.ndarray) -> float:
    """Rule D — ratio of peak hip to peak knee angular acceleration (RFD proxy)."""
    hip_rfd = float(np.max(np.abs(hip_acceleration))) if len(hip_acceleration) else 0.0
    knee_rfd = float(np.max(np.abs(knee_acceleration))) if len(knee_acceleration) else 0.0
    return hip_rfd / (knee_rfd + 1e-6)


def analyze_lift(
    frames: Sequence[BodyKeypoints],
    fps: int = DEFAULT_FPS,
    apply_smoothing: bool = True,
    bilateral: bool = True,
) -> LiftMetrics:
    """
    Run the full Rule A-D biomechanical analysis over a sequence of frames.

    Args:
        frames: Per-frame extracted (and quality-checked) lower-body
            keypoints for one lift.
        fps: Video frame rate used for the velocity/acceleration derivatives.
        apply_smoothing: Apply Savitzky-Golay smoothing to the angle
            sequences before differentiating (reduces pose-jitter noise).
        bilateral: Average left/right side angles (see
            ``extract_hip_knee_angles``).

    Returns:
        A ``LiftMetrics`` instance with Rule A-D scores plus the mean
        keypoint confidence across the clip (used later for confidence
        weighted scoring).
    """
    if not frames:
        logger.warning("analyze_lift called with an empty frame sequence")
        return LiftMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

    hip_angles, knee_angles = extract_hip_knee_angles(frames, bilateral=bilateral)

    if apply_smoothing:
        hip_angles = smooth_angle_sequence(hip_angles)
        knee_angles = smooth_angle_sequence(knee_angles)

    hip_velocity, hip_acceleration = compute_derivatives(hip_angles, fps)
    knee_velocity, knee_acceleration = compute_derivatives(knee_angles, fps)

    delay = rule_a_sequential_extension(knee_velocity, hip_velocity, fps)
    hip_dominance_ratio = rule_b_hip_dominance(knee_velocity, hip_velocity)
    synchronization_corr = rule_c_synchronization(knee_velocity, hip_velocity)
    rfd_ratio = rule_d_rfd(knee_acceleration, hip_acceleration)
    mean_confidence = float(np.mean([kp.mean_confidence for kp in frames]))

    # Anthropometric normalization (raw, unnormalized keypoints — normalization
    # is applied by dividing by the clip's own reference length, not by
    # pre-scaling the keypoints themselves). See
    # reports/anthropometric_normalization.md for the full derivation.
    reference = compute_reference_lengths(frames)
    hip_linear_rom_px, hip_linear_rom_normalized = compute_linear_rom(frames, "hip", reference.primary_reference)
    knee_linear_rom_px, knee_linear_rom_normalized = compute_linear_rom(frames, "knee", reference.primary_reference)
    hip_peak_linear_velocity_px_s, hip_peak_linear_velocity_normalized = compute_linear_velocity(
        frames, "hip", reference.primary_reference, fps,
    )
    knee_peak_linear_velocity_px_s, knee_peak_linear_velocity_normalized = compute_linear_velocity(
        frames, "knee", reference.primary_reference, fps,
    )

    # Confidence-weighted Rule A-D scoring (reports/confidence_weighted_rules.md):
    # each rule's confidence is grounded in the specific frame(s) it actually
    # depends on, rather than one blanket clip-wide value for every rule.
    rule_confidences = compute_rule_confidences(frames, hip_velocity, knee_velocity, hip_acceleration, knee_acceleration)
    overall_confidence = compute_overall_confidence(rule_confidences)

    logger.info(
        "analyze_lift | delay=%.4f hip_dom=%.4f corr=%.4f rfd=%.4f conf=%.3f leg_ref_px=%.1f overall_conf=%.3f",
        delay, hip_dominance_ratio, synchronization_corr, rfd_ratio, mean_confidence, reference.primary_reference, overall_confidence,
    )

    return LiftMetrics(
        delay=delay,
        hip_dominance_ratio=hip_dominance_ratio,
        synchronization_corr=synchronization_corr,
        rfd_ratio=rfd_ratio,
        mean_confidence=mean_confidence,
        reference=reference,
        hip_linear_rom_px=hip_linear_rom_px,
        hip_linear_rom_normalized=hip_linear_rom_normalized,
        knee_linear_rom_px=knee_linear_rom_px,
        knee_linear_rom_normalized=knee_linear_rom_normalized,
        hip_peak_linear_velocity_px_s=hip_peak_linear_velocity_px_s,
        hip_peak_linear_velocity_normalized=hip_peak_linear_velocity_normalized,
        knee_peak_linear_velocity_px_s=knee_peak_linear_velocity_px_s,
        knee_peak_linear_velocity_normalized=knee_peak_linear_velocity_normalized,
        rule_a_confidence=rule_confidences["rule_a_confidence"],
        rule_b_confidence=rule_confidences["rule_b_confidence"],
        rule_c_confidence=rule_confidences["rule_c_confidence"],
        rule_d_confidence=rule_confidences["rule_d_confidence"],
        overall_confidence=overall_confidence,
    )
