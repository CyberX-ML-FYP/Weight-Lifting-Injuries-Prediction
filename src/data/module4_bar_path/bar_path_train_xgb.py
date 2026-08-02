"""
Module 4 (bar_path) — XGBoost variant of bar_path_train.py.

Same 7 summary features, same target (good/bad label), same injury-risk-
as-P(bad) framing, same stratified-CV honesty check -- only the model
differs. Reuses bar_path_train.train_and_report() for everything else so
the two scripts can't drift apart on evaluation methodology.

Run:
    python -m src.data.module4_bar_path.bar_path_train_xgb
"""
from __future__ import annotations

from xgboost import XGBClassifier

from .bar_path_train import FEATURE_COLUMNS, RANDOM_STATE, train_and_report


def build_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )


def _xgb_importances(model: XGBClassifier) -> dict:
    return dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist()))


def main() -> None:
    train_and_report(
        model_name="xgb",
        model_factory=build_model,
        model_filename="bar_path_xgb.pkl",
        report_filename="bar_path_xgb_report.json",
        get_importances=_xgb_importances,
    )


if __name__ == "__main__":
    main()
