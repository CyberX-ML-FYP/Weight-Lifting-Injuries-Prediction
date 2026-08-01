"""Model Trainer — trains and evaluates classifiers on a module's master feature file.

Mirrors the approach already used in ``src/module3_arm_analysis/train_model.py``
(RandomForest vs XGBoost, ANOVA F-test feature selection scaled to sample
size), with one change: each lift (``video_id``) can appear as up to three
rows here -- one per camera view -- so plain K-fold would let the model see
one view of a lift in training and another view of the *same* lift in the
test fold, leaking label information. Cross-validation is grouped by
``video_id`` instead, so every view of a given lift stays on the same side
of the split.

Currently only Module 1 is implemented.
"""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")

# Below this many distinct lifts, 5-fold CV is too noisy to trust -- use
# leave-one-group-out instead. Once the dataset grows past this, switch to
# stratified group k-fold.
SMALL_SAMPLE_CV_THRESHOLD = 20

ID_COLUMNS = ("video_id", "label")
CATEGORICAL_COLUMNS = ("view",)

# Recording-artifact columns: how long the clip / lift-phase window is. These
# correlate strongly with label (bad attempts tend to get cut short), which
# makes them predictive but not biomechanical -- a model trained on them
# learns "how long is this clip", not "was the lifter's form correct". Kept
# out by default; pass --include-duration-features to add them back in.
DURATION_COLUMNS = ("n_frames_total", "n_lift_phase_frames", "lift_phase_duration_ms", "used_full_video_fallback")


def _prepare_data(csv_path: str, *, include_duration_features: bool = False):
    """Load a module's master feature file and split into X/y/groups."""
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Dataset is empty: {csv_path}")

    groups = df["video_id"]
    y = df["label"]

    drop_columns = list(ID_COLUMNS)
    if not include_duration_features:
        drop_columns += list(DURATION_COLUMNS)

    feature_df = df.drop(columns=drop_columns, errors="ignore")
    feature_df = pd.get_dummies(feature_df, columns=[c for c in CATEGORICAL_COLUMNS if c in feature_df.columns])
    feature_df = feature_df.apply(lambda col: col.astype(int) if col.dtype == bool else col)

    X = feature_df.fillna(feature_df.mean(numeric_only=True))
    return X, y, groups


def _select_k(n_samples: int, n_features: int) -> int:
    """Feature count for SelectKBest, scaling with sample size."""
    k = int(np.clip(n_samples // 2, 3, 8))
    return min(k, n_features)


def _select_features(X: pd.DataFrame, y: pd.Series) -> list[str]:
    """
    Report the top-k features an ANOVA F-test fit on the FULL dataset would
    keep. This is only used to pick the feature set for the final, deployed
    model -- cross-validation below refits selection inside each fold
    instead of reusing this, to avoid leaking test-fold labels into feature
    selection.
    """
    k = _select_k(len(y), X.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)

    scores = pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)
    selected = list(scores.index[:k])

    print("\n===== FEATURE SELECTION (SelectKBest, ANOVA F-test) =====")
    print(f"Samples: {len(y)}  ->  keeping top {k} of {X.shape[1]} features")
    for feat in selected:
        print(f"  {feat}: F={scores[feat]:.2f}")

    return selected


def _make_cv(n_groups: int):
    """Leave-one-group-out for few lifts; stratified group k-fold once there's more data."""
    if n_groups <= SMALL_SAMPLE_CV_THRESHOLD:
        return LeaveOneGroupOut()
    return StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)


def _evaluate_models(X: pd.DataFrame, y: pd.Series, groups: pd.Series):
    """
    Evaluate both models via group-aware cross-validation, grouped by lift
    (video_id) so a model is never trained on one view of a lift and tested
    on another view of that same lift. Feature selection is refit inside
    each fold (via Pipeline) so the reported accuracy isn't inflated by
    selecting features on data that includes the held-out fold's own labels.
    """
    n_groups = groups.nunique()
    cv = _make_cv(n_groups)
    k = _select_k(len(y), X.shape[1])
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            max_depth=2,
            random_state=42,
            eval_metric="logloss",
        ),
    }

    scores = {}
    for name, model in models.items():
        pipeline = Pipeline([
            ("select", SelectKBest(score_func=f_classif, k=k)),
            ("clf", model),
        ])
        cv_scores = cross_val_score(pipeline, X, y, groups=groups, cv=cv, scoring="accuracy")
        scores[name] = cv_scores.mean()
        print(f"{name} mean CV accuracy: {scores[name]:.4f}  ({cv.__class__.__name__}, {len(cv_scores)} folds, {n_groups} lifts)")

    best_name = max(scores, key=scores.get)
    return best_name, models[best_name], scores


def _save_feature_importance(model, feature_names, output_path: str, module_label: str) -> None:
    """Save feature importance bar chart and print ranking."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    importances.plot(kind="bar")
    plt.title(f"{module_label} Feature Importance")
    plt.ylabel("Importance")
    plt.xlabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("\nMost important features:")
    for feature, score in importances.items():
        print(f"  {feature}: {score:.4f}")


def train_module_model(module: int, *, include_duration_features: bool = False) -> object:
    """Train the best model on full data and save model + feature importance."""
    module_label = f"module{module}"
    data_path = os.path.join(FEATURES_DIR, module_label, f"{module_label}_features.csv")
    model_path = os.path.join(MODELS_DIR, f"{module_label}_model.pkl")
    figure_path = os.path.join(FIGURES_DIR, module_label, "feature_importance.png")

    X, y, groups = _prepare_data(data_path, include_duration_features=include_duration_features)
    best_name, best_model, _ = _evaluate_models(X, y, groups)

    selected_features = _select_features(X, y)
    X_selected = X[selected_features]
    best_model.fit(X_selected, y)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"model": best_model, "features": selected_features}, model_path)
    print(f"\nBest model: {best_name}")
    print(f"Saved model to: {model_path}")

    _save_feature_importance(best_model, selected_features, figure_path, f"Module {module}")
    print(f"Saved feature importance chart to: {figure_path}")

    return best_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a classifier on a module's master feature file")
    parser.add_argument("--module", type=int, default=1, help="Which module to train (default: 1)")
    parser.add_argument(
        "--include-duration-features",
        action="store_true",
        help="Include clip-length/lift-phase-duration columns (recording artifacts, not biomechanics)",
    )
    args = parser.parse_args()

    train_module_model(args.module, include_duration_features=args.include_duration_features)


if __name__ == "__main__":
    main()
