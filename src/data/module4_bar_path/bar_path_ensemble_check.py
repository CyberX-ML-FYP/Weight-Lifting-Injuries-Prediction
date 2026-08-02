"""
Module 4 (bar_path) — repeatable gate for whether the LSTM is safe to add
to bar_path_predict.py's ENSEMBLE_MODELS.

Background: at 58 training lifts the LSTM's cross-validated predictions
looked good on paper (ROC-AUC 1.0) but were actually unreliable -- every
predicted probability was crammed into a narrow 0.32-0.68 band (never
confident either direction) and precision was only 0.57 (it was mostly
just guessing "bad"). RF/XGBoost did not show this pattern. See
models/bar_path_lstm_report.json's "caveat" field and the bar_path_train_
lstm.py module docstring for the full story.

This script re-checks that exact failure signature every time you retrain
on a larger dataset, instead of eyeballing the reports by hand:

  1. PRECONDITION -- data/features/bar_path_features.csv has >= MIN_SAMPLES
     rows. Below this, don't even bother evaluating LSTM's report; 58 was
     the core problem last time and needs to move by a lot, not a little.
  2. LSTM's CV accuracy is not meaningfully worse than RF's (allows a small
     margin so it doesn't have to be strictly the best model, just not the
     worst).
  3. LSTM's CV precision and recall both clear reasonable floors -- targets
     the "just predicts bad most of the time" failure directly.
  4. LSTM's CV F1 clears a floor -- general balance check.
  5. LSTM's per-lift CV risk scores actually spread toward 0 and 1 (not
     crammed in the middle) -- targets the "never confident" failure
     directly. This is the check accuracy/AUC alone cannot catch.

All must pass for this script to say LSTM is safe to add. This is a
recommendation, not an automatic code change -- ENSEMBLE_MODELS in
bar_path_predict.py is still edited by hand once you've reviewed the
verdict.

Run:
    python -m src.data.module4_bar_path.bar_path_ensemble_check
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .bar_path_predict import ENSEMBLE_MODELS
from .config import BarPathConfig
from .utils import setup_logger

logger = setup_logger(__name__)

MIN_SAMPLES = 200
MAX_ACCURACY_GAP = 0.02   # LSTM accuracy must be >= RF accuracy - this
MIN_PRECISION = 0.70
MIN_RECALL = 0.60
MIN_F1 = 0.70
SPREAD_LOW_MAX = 0.15     # at least one CV prediction must be <= this
SPREAD_HIGH_MIN = 0.85    # at least one CV prediction must be >= this


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def _load_report(config: BarPathConfig, filename: str) -> dict:
    path = config.root_dir / "models" / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the matching bar_path_train*.py script first."
        )
    return json.loads(path.read_text())


def check_sample_size(config: BarPathConfig) -> CheckResult:
    df = pd.read_csv(config.features_output_path)
    n = len(df)
    passed = n >= MIN_SAMPLES
    return CheckResult(
        name="sample_size",
        passed=passed,
        detail=f"{n} labelled lifts in bar_path_features.csv (need >= {MIN_SAMPLES})",
    )


def check_accuracy_not_worse(rf_report: dict, lstm_report: dict) -> CheckResult:
    rf_acc = rf_report["cv_metrics"]["accuracy"]
    lstm_acc = lstm_report["cv_metrics"]["accuracy"]
    passed = lstm_acc >= rf_acc - MAX_ACCURACY_GAP
    return CheckResult(
        name="accuracy_not_worse_than_rf",
        passed=passed,
        detail=f"LSTM accuracy {lstm_acc:.3f} vs RF {rf_acc:.3f} (allowed gap {MAX_ACCURACY_GAP})",
    )


def check_precision_recall(lstm_report: dict) -> CheckResult:
    precision = lstm_report["cv_metrics"]["precision"]
    recall = lstm_report["cv_metrics"]["recall"]
    passed = precision >= MIN_PRECISION and recall >= MIN_RECALL
    return CheckResult(
        name="precision_recall_floor",
        passed=passed,
        detail=f"precision={precision:.3f} (need >= {MIN_PRECISION}), recall={recall:.3f} (need >= {MIN_RECALL})",
    )


def check_f1(lstm_report: dict) -> CheckResult:
    f1 = lstm_report["cv_metrics"]["f1"]
    passed = f1 >= MIN_F1
    return CheckResult(
        name="f1_floor",
        passed=passed,
        detail=f"f1={f1:.3f} (need >= {MIN_F1})",
    )


def check_probability_spread(config: BarPathConfig) -> CheckResult:
    path = config.interim_output_dir / "bar_path_cv_predictions_lstm.csv"
    if not path.exists():
        return CheckResult(
            name="probability_spread",
            passed=False,
            detail=f"Missing {path} -- run bar_path_train_lstm.py",
        )

    df = pd.read_csv(path)
    scores = df["cv_risk_score"]
    passed = bool((scores <= SPREAD_LOW_MAX).any() and (scores >= SPREAD_HIGH_MIN).any())
    return CheckResult(
        name="probability_spread",
        passed=passed,
        detail=(
            f"min={scores.min():.3f} (need <= {SPREAD_LOW_MAX}), "
            f"max={scores.max():.3f} (need >= {SPREAD_HIGH_MIN}) -- "
            "catches the 'never confident, all predictions crammed in the middle' failure"
        ),
    )


def is_lstm_in_ensemble() -> bool:
    return any(name == "lstm" for name, _, _ in ENSEMBLE_MODELS)


def run_all_checks(config: BarPathConfig | None = None) -> tuple[bool, list[CheckResult]]:
    config = config or BarPathConfig()

    if is_lstm_in_ensemble():
        already_added = CheckResult(
            name="already_in_ensemble",
            passed=True,
            detail="LSTM is already listed in ENSEMBLE_MODELS -- nothing to check.",
        )
        return True, [already_added]

    size_check = check_sample_size(config)
    if not size_check.passed:
        # Precondition failed -- don't bother evaluating the reports at all.
        return False, [size_check]

    rf_report = _load_report(config, "bar_path_rf_report.json")
    lstm_report = _load_report(config, "bar_path_lstm_report.json")

    results = [
        size_check,
        check_accuracy_not_worse(rf_report, lstm_report),
        check_precision_recall(lstm_report),
        check_f1(lstm_report),
        check_probability_spread(config),
    ]
    safe_to_add = all(r.passed for r in results)
    return safe_to_add, results


def main() -> None:
    config = BarPathConfig()

    if is_lstm_in_ensemble():
        print("LSTM is already listed in ENSEMBLE_MODELS -- nothing to check.")
        logger.info("safe_to_add=n/a (already in ensemble)")
        return

    safe_to_add, results = run_all_checks(config)

    print("LSTM ensemble-readiness check")
    print("=" * 40)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")
    print("=" * 40)

    if safe_to_add:
        print(
            "VERDICT: safe to add LSTM to ENSEMBLE_MODELS in bar_path_predict.py.\n"
            'Add: ("lstm", "bar_path_lstm.pt", 1.0)  -- note the LSTM loader/scorer\n'
            "needs its own code path there (torch model, not a joblib sklearn bundle)."
        )
    else:
        print("VERDICT: NOT safe to add LSTM to the ensemble yet. See failed checks above.")

    logger.info("safe_to_add=%s", safe_to_add)


if __name__ == "__main__":
    main()
