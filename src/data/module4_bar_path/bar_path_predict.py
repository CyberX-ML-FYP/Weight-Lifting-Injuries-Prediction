"""
Module 4 (bar_path) — Pass 4: run the full pipeline on a single new video
and produce an injury-risk report.

Video -> raw landmark extraction (raw_extractor) -> lift-phase trim +
cleaning (raw_cleaner) -> per-lift summary features (feature_extractor) ->
RF + XGBoost ensemble -> risk report.

This is the "give it one lift, get a report back" entry point a coach
would actually run, as opposed to the batch scripts (raw_extractor.py /
raw_cleaner.py / bar_path_train.py / bar_path_train_xgb.py) which process
the whole training set.

The final injury-risk score averages RF and XGBoost's P(bad) (see
ENSEMBLE_WEIGHTS below). The LSTM (bar_path_train_lstm.py) is deliberately
NOT part of the ensemble: on only 58 training sequences its cross-
validated predictions were unreliable (all crammed into a narrow
probability band, AUC=1.0 but precision 0.57 -- a small-data overfitting
signature, not real skill; see models/bar_path_lstm_report.json). Revisit
once more labelled videos exist.

Run:
    python -m src.data.module4_bar_path.bar_path_predict --video path/to/lift.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .bar_path_train import FEATURE_COLUMNS, risk_band
from .config import BarPathConfig
from .feature_extractor import extract_bar_path_features, summarize_bar_path_features
from .raw_cleaner import clean_raw_landmarks
from .raw_extractor import download_model_if_needed, extract_raw_landmarks, find_lift_window
from .utils import setup_logger

logger = setup_logger(__name__)

# (model filename, ensemble weight) -- weights need not sum to 1, they are
# normalised at combine time so adding/removing a model is a one-line change.
ENSEMBLE_MODELS = [
    ("rf", "bar_path_rf.pkl", 1.0),
    ("xgb", "bar_path_xgb.pkl", 1.0),
]


def load_model(config: BarPathConfig, filename: str):
    model_path = config.root_dir / "models" / filename
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run the matching "
            "bar_path_train*.py script first."
        )
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["feature_columns"]


def extract_step(video_path: Path, config: BarPathConfig) -> pd.DataFrame:
    """Stage 1: raw landmark extraction + lift-phase detection. Split out
    from clean_step/summarize_step so callers (e.g. the Streamlit app) can
    show per-stage progress instead of one opaque multi-second call."""
    video_id = video_path.stem
    download_model_if_needed(config)

    rows = extract_raw_landmarks(video_path, config)
    if not rows:
        raise RuntimeError(f"No frames could be processed for {video_path}")

    raw_df = pd.DataFrame(rows)
    raw_df.insert(0, "video_id", video_id)

    start, end = find_lift_window(raw_df)
    raw_df["lift_phase"] = False
    raw_df.loc[start:end, "lift_phase"] = True
    logger.info(
        "Detected lift phase: frames %s-%s of %s",
        raw_df["frame_id"].iloc[start], raw_df["frame_id"].iloc[end], len(raw_df),
    )
    return raw_df


def clean_step(raw_df: pd.DataFrame, config: BarPathConfig, video_id: str) -> pd.DataFrame:
    """Stage 2: lift-phase filter + wrist-visibility filter + outlier
    removal + smoothing (raw_cleaner.clean_raw_landmarks)."""
    cleaned_df = clean_raw_landmarks(raw_df, config, video_id)
    if cleaned_df.empty:
        raise RuntimeError(
            f"No usable frames survived cleaning for video '{video_id}' "
            "(check wrist visibility / lift detection)"
        )
    return cleaned_df


def summarize_step(cleaned_df: pd.DataFrame, video_id: str) -> pd.DataFrame:
    """Stage 3: per-frame bar-path features -> one-row per-lift summary,
    the exact feature vector the trained models expect."""
    frame_features = extract_bar_path_features(cleaned_df, video_id)
    lift_features = summarize_bar_path_features(frame_features, video_id)
    return lift_features


def run_pipeline(video_path: Path, config: BarPathConfig) -> pd.DataFrame:
    """Video -> one-row lift-level feature DataFrame, reusing the exact
    same extraction/cleaning/feature functions the training set was built
    from, so a prediction is comparable to what the model was trained on."""
    video_id = video_path.stem
    logger.info("Extracting raw landmarks for %s", video_path.name)
    raw_df = extract_step(video_path, config)
    cleaned_df = clean_step(raw_df, config, video_id)
    return summarize_step(cleaned_df, video_id)


def predict_from_features(lift_features: pd.DataFrame, config: BarPathConfig | None = None) -> dict:
    """Ensemble scoring given an already-computed lift feature row --
    the part of predict() that doesn't need the video at all, split out so
    the UI can call it after its own staged extract/clean/summarize calls
    without re-running the pipeline end to end."""
    config = config or BarPathConfig()
    video_id = lift_features["video_id"].iloc[0] if "video_id" in lift_features.columns else "lift"

    per_model_scores = {}
    weighted_sum = 0.0
    weight_total = 0.0
    contributions_by_model = {}

    for model_name, filename, weight in ENSEMBLE_MODELS:
        model, feature_columns = load_model(config, filename)
        X = lift_features[feature_columns].to_numpy(dtype=float)

        score = float(model.predict_proba(X)[0, 1])
        per_model_scores[model_name] = round(score, 4)
        weighted_sum += weight * score
        weight_total += weight

        contributions_by_model[model_name] = dict(zip(
            feature_columns,
            (model.feature_importances_ * X[0]).tolist(),
        ))

    ensemble_score = weighted_sum / weight_total

    # Combine per-model feature contributions (averaged across the
    # ensemble) to surface the top drivers of the FINAL blended score,
    # not just one model's view of it.
    combined_contributions: dict[str, float] = {}
    for contributions in contributions_by_model.values():
        for feature, value in contributions.items():
            combined_contributions[feature] = combined_contributions.get(feature, 0.0) + value / len(contributions_by_model)
    top_features = sorted(combined_contributions.items(), key=lambda kv: kv[1], reverse=True)[:3]

    report = {
        "video_id": video_id,
        "predicted_quality": "bad" if ensemble_score >= 0.5 else "good",
        "injury_risk_score": round(ensemble_score, 4),
        "injury_risk_band": risk_band(ensemble_score),
        "per_model_risk_scores": per_model_scores,
        "top_contributing_features": [name for name, _ in top_features],
        "features": {
            col: float(lift_features[col].iloc[0]) for col in FEATURE_COLUMNS
        },
    }
    return report


def predict(video_path: Path, config: BarPathConfig | None = None) -> dict:
    config = config or BarPathConfig()
    lift_features = run_pipeline(video_path, config)
    return predict_from_features(lift_features, config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict bar-path injury risk for one video.")
    parser.add_argument("--video", required=True, help="Path to a front-view lift video.")
    parser.add_argument("--out", default=None, help="Optional path to save the JSON report.")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    config = BarPathConfig()
    report = predict(video_path, config)

    print(json.dumps(report, indent=2))

    out_path = Path(args.out) if args.out else (
        config.root_dir / "reports" / "figures" / "module4" / f"{video_path.stem}_risk_report.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("Saved report to %s", out_path)


if __name__ == "__main__":
    main()
