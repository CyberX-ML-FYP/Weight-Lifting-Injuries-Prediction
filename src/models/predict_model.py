"""Predict lift quality for a new video using a trained module model.

Runs the full pipeline end-to-end on a single video: pose extraction ->
cleaning -> per-video feature aggregation -> classification, reusing the
exact same aggregation/encoding logic as ``src/features/build_features.py``
and ``src/models/train_model.py`` so prediction-time features can't drift
out of sync with what the model was trained on.

Currently only Module 1 is implemented, so this predicts lift quality from
trunk/spine features alone. Per src/models/train_model.py, that signal is
weak in isolation (~66% CV accuracy, barely above the majority-class
baseline) -- treat this as one component's prediction, not the full
system's performance-score/injury-risk output described in the README,
which needs the other three modules merged in.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from src.features.build_features import VALID_VIEWS, aggregate_module1_video
from src.features.module1_trunk import analyze_video, clean_module1_features
from src.models.train_model import CATEGORICAL_COLUMNS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INTERIM_DIR = os.path.join(BASE_DIR, "data", "interim")
MODELS_DIR = os.path.join(BASE_DIR, "models")

LABEL_NAMES = {0: "good", 1: "bad"}


def _infer_view(video_path: str) -> str:
    parent = os.path.basename(os.path.dirname(os.path.abspath(video_path)))
    if parent not in VALID_VIEWS:
        raise ValueError(
            f"Could not infer view from path '{video_path}' (parent folder '{parent}' "
            f"is not one of {VALID_VIEWS}). Place the video under data/raw/videos/<view>/ "
            "or pass --view explicitly."
        )
    return parent


def _encode_single_row(feature_row: Dict[str, Any], expected_columns: List[str]) -> pd.DataFrame:
    """Encode one aggregated feature row the same way training data was encoded."""
    row_df = pd.DataFrame([feature_row])
    row_df = pd.get_dummies(row_df, columns=[c for c in CATEGORICAL_COLUMNS if c in row_df.columns])
    row_df = row_df.apply(lambda col: col.astype(int) if col.dtype == bool else col)

    for col in expected_columns:
        if col not in row_df.columns:
            row_df[col] = 0

    return row_df[expected_columns]


def load_model_bundle(module: int = 1) -> Dict[str, Any]:
    if module != 1:
        raise NotImplementedError("Only Module 1 prediction is implemented so far.")

    model_path = os.path.join(MODELS_DIR, "module1_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model at {model_path}. Run `python -m src.models.train_model` first.")
    return joblib.load(model_path)


def score_feature_row(feature_row: Dict[str, Any], *, module: int = 1, bundle: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Score an already-aggregated feature row (e.g. loaded from module1_features.csv)."""
    bundle = bundle or load_model_bundle(module)
    model, expected_columns = bundle["model"], bundle["features"]

    X = _encode_single_row(feature_row, expected_columns)
    prediction = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]

    importances = getattr(model, "feature_importances_", None)
    top_features = []
    if importances is not None:
        ranked = sorted(zip(expected_columns, importances), key=lambda kv: kv[1], reverse=True)
        top_features = [
            {"feature": name, "value": feature_row.get(name), "importance": float(score)}
            for name, score in ranked
            if name in feature_row
        ][:5]

    return {
        "video_id": feature_row.get("video_id"),
        "view": feature_row.get("view"),
        "predicted_label": prediction,
        "predicted_class": LABEL_NAMES.get(prediction, str(prediction)),
        "confidence": float(proba[prediction]),
        "class_probabilities": {LABEL_NAMES.get(i, str(i)): float(p) for i, p in enumerate(proba)},
        "top_features": top_features,
        "feature_row": feature_row,
    }


def predict_video(
    video_path: str,
    *,
    view: Optional[str] = None,
    module: int = 1,
    show_display: bool = False,
    bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if module != 1:
        raise NotImplementedError("Only Module 1 prediction is implemented so far.")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    resolved_view = view or _infer_view(video_path)
    video_id = os.path.splitext(os.path.basename(video_path))[0]

    interim_csv = os.path.join(INTERIM_DIR, "module1", resolved_view, f"{video_id}.csv")
    raw_df = analyze_video(video_path, interim_csv, show_display=show_display)
    cleaned_df = clean_module1_features(raw_df)

    feature_row = aggregate_module1_video(cleaned_df, video_id, resolved_view)
    return score_feature_row(feature_row, module=module, bundle=bundle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict lift quality for a new video (Module 1)")
    parser.add_argument("--video", required=True, help="Path to the video, ideally under data/raw/videos/<view>/")
    parser.add_argument("--view", choices=VALID_VIEWS, default=None, help="Camera view (inferred from parent folder if omitted)")
    parser.add_argument("--module", type=int, default=1, help="Which module's model to use (default: 1)")
    parser.add_argument("--show", action="store_true", help="Display the annotated video while processing")
    args = parser.parse_args()

    result = predict_video(args.video, view=args.view, module=args.module, show_display=args.show)

    print(f"\n===== PREDICTION: {result['video_id']} ({result['view']}) =====")
    print(f"Predicted class     : {result['predicted_class']}  (confidence {result['confidence']:.1%})")
    probs = ", ".join(f"{k}={v:.1%}" for k, v in result["class_probabilities"].items())
    print(f"Class probabilities : {probs}")

    print("\nTop contributing features (model importance, this video's value):")
    for item in result["top_features"]:
        print(f"  {item['feature']}: value={item['value']}  importance={item['importance']:.4f}")

    print(
        "\nNote: this is Module 1 (trunk/spine) acting alone. Its cross-validated "
        "accuracy on trunk/spine features is ~66%, barely above the majority-class "
        "baseline -- treat this as one component signal, not a final verdict, until "
        "the other modules are merged into a master dataset."
    )


if __name__ == "__main__":
    main()
