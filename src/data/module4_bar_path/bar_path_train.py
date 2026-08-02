"""
Module 4 (bar_path) — Pass 3: train a Random Forest on the per-lift
bar-path features to classify lift quality (good/bad) and derive an
injury-risk score from it.

There is no clinical injury-outcome data for this project -- only the
coach-assigned good/bad quality label parsed from each video's filename.
Injury risk is therefore not a separately-labelled target; it is read off
the SAME classifier as a proxy: a bad bar path (large deviation from
vertical, jerky/unsmooth trajectory, sudden corrections) forces the lifter
to compensate with the back/shoulders to keep the bar under control, and
that compensation is the mechanical link to injury risk. So:

    risk_score  = P(label == 1 "bad")  from the trained classifier
    risk_band   = Low  (risk_score < 0.33)
                  Moderate (0.33 <= risk_score < 0.66)
                  High (risk_score >= 0.66)

With only 58 labelled lifts, a single held-out test split would leave too
few samples to trust (e.g. a 15% test split is ~9 lifts). Evaluation uses
stratified 5-fold cross-validation across the whole dataset instead, and
the final model saved to disk is refit on all 58 rows.

Run:
    python -m src.data.module4_bar_path.bar_path_train
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .config import BarPathConfig
from .utils import setup_logger

logger = setup_logger(__name__)

FEATURE_COLUMNS = [
    "max_deviation",
    "avg_deviation",
    "path_smoothness",
    "peak_vertical_velocity",
    "time_to_peak_velocity",
    "total_displacement",
    "jerk_like_movements",
]

RISK_LOW_MAX = 0.33
RISK_MODERATE_MAX = 0.66

N_FOLDS = 5
RANDOM_STATE = 42


def risk_band(risk_score: float) -> str:
    if risk_score < RISK_LOW_MAX:
        return "Low"
    if risk_score < RISK_MODERATE_MAX:
        return "Moderate"
    return "High"


def load_training_data(config: BarPathConfig) -> pd.DataFrame:
    df = pd.read_csv(config.features_output_path)
    df = df.dropna(subset=FEATURE_COLUMNS + ["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df


def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def evaluate_cv(df: pd.DataFrame, model_factory=build_model) -> tuple[dict, np.ndarray]:
    """Stratified k-fold CV over the whole dataset -- with 58 rows this is
    more trustworthy than a single train/test split, which would leave a
    test fold too small (~9 rows at 15%) to draw any real conclusion from.

    model_factory lets other training scripts (e.g. XGBoost) reuse this
    same CV/metrics logic instead of duplicating it.
    """
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)

    n_folds = min(N_FOLDS, int(np.bincount(y).min()))
    if n_folds < 2:
        raise ValueError("Not enough samples per class for cross-validation")

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    model = model_factory()

    pred_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    pred_label = (pred_proba >= 0.5).astype(int)

    metrics = {
        "n_folds": n_folds,
        "accuracy": float(accuracy_score(y, pred_label)),
        "precision": float(precision_score(y, pred_label, zero_division=0)),
        "recall": float(recall_score(y, pred_label, zero_division=0)),
        "f1": float(f1_score(y, pred_label, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, pred_proba)),
    }
    return metrics, pred_proba


def train_final_model(df: pd.DataFrame, model_factory=build_model):
    """Refit on all available data for the model that actually gets saved
    and used for prediction -- the CV folds above are for honest evaluation
    only, not the deployed model."""
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    model = model_factory()
    model.fit(X, y)
    return model


def train_and_report(
    model_name: str,
    model_factory,
    model_filename: str,
    report_filename: str,
    get_importances,
) -> None:
    """Shared pipeline for any tabular (fixed-feature-vector) classifier:
    load data -> CV evaluate -> refit on everything -> save model +
    predictions + report. RF and XGBoost both call this; only the model
    itself and how importances are read off it differ.
    """
    config = BarPathConfig()
    df = load_training_data(config)
    logger.info("[%s] Loaded %s labelled lifts (%s good / %s bad)",
                model_name, len(df), int((df.label == 0).sum()), int((df.label == 1).sum()))

    metrics, cv_proba = evaluate_cv(df, model_factory)
    logger.info("[%s] Cross-validated metrics (%s-fold): %s", model_name, metrics["n_folds"], metrics)

    model = train_final_model(df, model_factory)

    importances = get_importances(model)
    ranked_importances = dict(
        sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
    )
    logger.info("[%s] Feature importances: %s", model_name, ranked_importances)

    models_dir = config.root_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / model_filename
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, model_path)
    logger.info("[%s] Saved trained model to %s", model_name, model_path)

    df = df.copy()
    df["cv_risk_score"] = cv_proba
    df["cv_risk_band"] = df["cv_risk_score"].apply(risk_band)
    predictions_path = config.interim_output_dir / f"bar_path_cv_predictions_{model_name}.csv"
    df[["video_id", "label", "cv_risk_score", "cv_risk_band"]].to_csv(
        predictions_path, index=False
    )
    logger.info("[%s] Saved per-lift CV predictions to %s", model_name, predictions_path)

    report_path = models_dir / report_filename
    report = {
        "model": model_name,
        "n_samples": len(df),
        "feature_columns": FEATURE_COLUMNS,
        "cv_metrics": metrics,
        "feature_importances": ranked_importances,
        "risk_bands": {
            "Low": f"< {RISK_LOW_MAX}",
            "Moderate": f"{RISK_LOW_MAX} - {RISK_MODERATE_MAX}",
            "High": f">= {RISK_MODERATE_MAX}",
        },
    }
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("[%s] Saved training report to %s", model_name, report_path)


def _rf_importances(model: RandomForestClassifier) -> dict:
    return dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist()))


def main() -> None:
    train_and_report(
        model_name="rf",
        model_factory=build_model,
        model_filename="bar_path_rf.pkl",
        report_filename="bar_path_rf_report.json",
        get_importances=_rf_importances,
    )


if __name__ == "__main__":
    main()
