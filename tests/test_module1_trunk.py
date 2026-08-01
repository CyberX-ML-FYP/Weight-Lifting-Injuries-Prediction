import pandas as pd
import pytest

from src.features.module1_trunk import (
    calculate_angle,
    classify_lift_phase,
    compute_lift_phase_flags,
    compute_trunk_features_from_landmarks,
)


def test_compute_trunk_features_from_landmarks():
    landmarks = {
        11: {"x": 0.40, "y": 0.20},
        12: {"x": 0.60, "y": 0.20},
        23: {"x": 0.45, "y": 0.70},
        24: {"x": 0.55, "y": 0.70},
    }

    features = compute_trunk_features_from_landmarks(landmarks)

    assert features["shoulder_mid_x"] == pytest.approx(0.5)
    assert features["shoulder_mid_y"] == pytest.approx(0.2)
    assert features["hip_mid_x"] == pytest.approx(0.5)
    assert features["hip_mid_y"] == pytest.approx(0.7)
    assert features["spine_angle"] > 0
    assert features["shoulder_asymmetry_flag"] == 0


def test_asymmetry_and_phase_classification():
    landmarks = {
        11: {"x": 0.60, "y": 0.10},
        12: {"x": 0.80, "y": 0.18},
        23: {"x": 0.12, "y": 0.70},
        24: {"x": 0.28, "y": 0.70},
    }

    features = compute_trunk_features_from_landmarks(landmarks)

    assert features["shoulder_asymmetry_flag"] == 1
    assert classify_lift_phase(features["spine_angle"]) == "first_pull"


def test_calculate_angle_uses_triangle_geometry():
    angle = calculate_angle((1, 0), (0, 0), (1, 1))
    assert angle == pytest.approx(45.0)


def test_compute_lift_phase_flags_detects_burst_between_static_periods():
    n_static = 20
    n_burst = 10
    n_rows = n_static + n_burst + n_static

    timestamps = [50 * i for i in range(n_rows)]
    shoulder_y = [0.5] * n_static + [0.5 - 0.03 * i for i in range(n_burst)] + [0.5 - 0.03 * (n_burst - 1)] * n_static
    hip_y = [0.7] * n_static + [0.7 - 0.03 * i for i in range(n_burst)] + [0.7 - 0.03 * (n_burst - 1)] * n_static

    df = pd.DataFrame({"timestamp_ms": timestamps, "shoulder_mid_y": shoulder_y, "hip_mid_y": hip_y})
    flags = compute_lift_phase_flags(df)

    assert not flags.iloc[:10].any()
    assert flags.iloc[n_static:n_static + n_burst].any()
    assert not flags.iloc[n_static + n_burst + 8:].any()
