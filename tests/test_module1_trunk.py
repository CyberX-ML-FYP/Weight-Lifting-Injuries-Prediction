import pytest

from src.features.module1_trunk import (
    calculate_angle,
    classify_lift_phase,
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
