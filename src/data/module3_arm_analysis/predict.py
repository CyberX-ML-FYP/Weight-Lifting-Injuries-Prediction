"""
Module 3 — Single-Lift Prediction
Runs the trained model on ONE new lift (one or more camera views) instead
of a whole batch. This is what a frontend calls.

Author: Pasindu (214027H)
"""
import os
import tempfile

import joblib
import pandas as pd

from src.module3_arm_analysis.analyzer import analyze_video
from src.module3_arm_analysis.config import BASE_DIR
from src.module3_arm_analysis.feature_extractor import (
    ELBOW_VIEWS,
    elbow_features_from_df,
    front_features_from_df,
)

MODEL_PATH = os.path.join(BASE_DIR, "models", "module3_model.pkl")


def load_model():
    """Load the trained model bundle (model + selected features + fill-in means)."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}. Run train_model.py first."
        )
    return joblib.load(MODEL_PATH)


def analyze_lift_videos(video_paths, show_display=False):
    """
    Run pose analysis on one or more camera views of a single lift and
    compute Module 3's engineered features for it.

    Args:
        video_paths: dict mapping view name ("side", "front", "angle") to
            a video file path. At least one of "side"/"angle" is needed
            for a meaningful prediction -- the trained model's top
            features all come from those two views, not front.
        show_display: passed through to analyze_video (live overlay window).

    Returns:
        (features, frame_dfs): features is a flat dict of every computed
        feature across the provided views; frame_dfs maps view -> the
        raw per-frame dataframe (frame, elbow angles, wrist_y, ...), for
        rendering charts in a UI.
    """
    unknown = set(video_paths) - {"side", "front", "angle"}
    if unknown:
        raise ValueError(f"Unknown view(s): {unknown}. Expected side/front/angle.")
    if not video_paths:
        raise ValueError("Provide at least one video (side and/or angle recommended).")

    features = {}
    frame_dfs = {}

    with tempfile.TemporaryDirectory(prefix="module3_predict_") as tmp_dir:
        for view, path in video_paths.items():
            tmp_csv = os.path.join(tmp_dir, f"{view}.csv")
            df = analyze_video(path, tmp_csv, show_display=show_display)
            frame_dfs[view] = df

            if view in ELBOW_VIEWS:
                features.update(elbow_features_from_df(df, f"{view}_"))
            elif view == "front":
                features.update(front_features_from_df(df))

    return features, frame_dfs


def predict_from_features(features):
    """
    Score a single lift's feature dict against the trained model.

    Any of the model's selected features that couldn't be computed
    (e.g. that camera view wasn't provided) falls back to the training
    set's mean for that feature -- the same principled gap-filling used
    during training, not something guessed from this one lift.

    Returns a dict: label (0/1), prediction ("good"/"bad"),
    probability_bad, and which selected features were actually measured
    vs. imputed.
    """
    bundle = load_model()
    model = bundle["model"]
    selected_features = bundle["features"]
    feature_means = bundle["feature_means"]

    row = {}
    imputed = []
    for name in selected_features:
        value = features.get(name)
        if value is None or pd.isna(value):
            row[name] = feature_means[name]
            imputed.append(name)
        else:
            row[name] = value

    X = pd.DataFrame([row], columns=selected_features)
    label = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    # proba is ordered by the model's classes_ (0=good, 1=bad in this dataset)
    probability_bad = float(proba[list(model.classes_).index(1)])

    return {
        "label": label,
        "prediction": "bad" if label == 1 else "good",
        "probability_bad": probability_bad,
        "model_inputs": row,
        "imputed_features": imputed,
    }


def predict_lift(video_paths, show_display=False):
    """
    End-to-end: video file(s) for one lift -> prediction.

    Args:
        video_paths: dict mapping "side"/"front"/"angle" -> video file path.
        show_display: passed through to the pose analyzer.

    Returns:
        dict with the prediction (see predict_from_features), plus
        "features" (every computed feature) and "frame_dfs" (per-frame
        data per view, for plotting).
    """
    features, frame_dfs = analyze_lift_videos(video_paths, show_display=show_display)
    result = predict_from_features(features)
    result["features"] = features
    result["frame_dfs"] = frame_dfs
    return result


if __name__ == "__main__":
    # Example usage
    result = predict_lift({
        "side": "data/raw_videos/side/1good.MOV",
        "front": "data/raw_videos/front/1good .mp4",
    })
    print(f"Prediction: {result['prediction']} (P(bad)={result['probability_bad']:.2f})")
    print(f"Imputed (missing view) features: {result['imputed_features']}")
