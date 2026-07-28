import pandas as pd

from src.module3_arm_analysis.feature_extractor import build_front_view_combined_dataset


def test_build_front_view_combined_dataset(tmp_path):
    pd.DataFrame(
        {
            "frame": [1, 2],
            "symmetry_diff": [5.0, 10.0],
            "asymmetry_flag": [0, 1],
            "low_visibility_flag": [0, 0],
        }
    ).to_csv(tmp_path / "1good_front.csv", index=False)

    pd.DataFrame(
        {
            "frame": [1, 2, 3],
            "symmetry_diff": [3.0, 8.0, 4.0],
            "asymmetry_flag": [0, 0, 1],
            "low_visibility_flag": [0, 0, 0],
        }
    ).to_csv(tmp_path / "2bad_front.csv", index=False)

    output_path = tmp_path / "combined_front_dataset.csv"
    dataset = build_front_view_combined_dataset(
        tmp_path, output_csv_path=str(output_path)
    )

    assert output_path.exists()
    assert {"lift_id", "label", "phase", "frame_ratio"}.issubset(dataset.columns)
    assert dataset["lift_id"].nunique() == 2
    assert set(dataset["label"].dropna().unique()) == {0, 1}
    assert set(dataset["phase"].unique()) <= {"early", "middle", "final"}
