import pandas as pd
import pytest

from src.data.module1_trunk.config import TrunkConfig
from src.data.module1_trunk.feature_extractor import (
    calculate_angle,
    classify_lift_phase,
    extract_trunk_features,
)


def _landmark_row(frame_id, left_shoulder, right_shoulder, left_hip, right_hip, fps=30.0):
    return {
        "frame_id": frame_id,
        "left_shoulder_x": left_shoulder[0],
        "left_shoulder_y": left_shoulder[1],
        "right_shoulder_x": right_shoulder[0],
        "right_shoulder_y": right_shoulder[1],
        "left_hip_x": left_hip[0],
        "left_hip_y": left_hip[1],
        "right_hip_x": right_hip[0],
        "right_hip_y": right_hip[1],
        "fps": fps,
    }


def test_extract_trunk_features_from_landmarks():
    config = TrunkConfig()
    raw_df = pd.DataFrame(
        [_landmark_row(0, (0.40, 0.20), (0.60, 0.20), (0.45, 0.70), (0.55, 0.70))]
    )

    features = extract_trunk_features(raw_df, "1good", config)

    row = features.iloc[0]
    assert row["shoulder_mid_x"] == pytest.approx(0.5)
    assert row["shoulder_mid_y"] == pytest.approx(0.2)
    assert row["hip_mid_x"] == pytest.approx(0.5)
    assert row["hip_mid_y"] == pytest.approx(0.7)
    assert row["spine_angle"] > 0
    assert row["shoulder_asymmetry_flag"] == 0
    assert row["low_visibility"] == 0


def test_asymmetry_and_phase_classification():
    config = TrunkConfig()
    raw_df = pd.DataFrame(
        [_landmark_row(0, (0.60, 0.10), (0.80, 0.18), (0.12, 0.70), (0.28, 0.70))]
    )

    features = extract_trunk_features(raw_df, "2bad", config)

    row = features.iloc[0]
    assert row["shoulder_asymmetry_flag"] == 1
    assert classify_lift_phase(row["spine_angle"]) == "first_pull"


def test_calculate_angle_uses_triangle_geometry():
    angle = calculate_angle((1, 0), (0, 0), (1, 1))
    assert angle == pytest.approx(45.0)
