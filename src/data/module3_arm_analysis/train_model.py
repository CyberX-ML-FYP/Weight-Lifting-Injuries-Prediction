"""
Module 3 — Model Trainer
Trains and evaluates classifiers on the module3 master dataset.

Author: Pasindu (214027H)
"""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data.module3_arm_analysis.config import BASE_DIR, MASTER_DATASET_PATH, OUTPUT_DIR

# Below this many lifts, 5-fold CV is too noisy to trust -- use leave-one-out
# instead. Once the dataset grows past this, switch to stratified k-fold.
SMALL_SAMPLE_CV_THRESHOLD = 20


def _prepare_data(csv_path):
    """Load dataset and split into X/y."""
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Dataset is empty: {csv_path}")

    feature_df = df.drop(columns=["lift_id", "label"], errors="ignore")
    X = feature_df.copy()
    X = X.fillna(X.mean(numeric_only=True))
    y = df["label"]
    return X, y


def _select_k(n_samples, n_features):
    """Feature count for SelectKBest, scaling with sample size."""
    k = int(np.clip(n_samples // 2, 3, 8))
    return min(k, n_features)


def _select_features(X, y):
    """
    Report the top-k features an ANOVA F-test fit on the FULL dataset would
    keep. This is only used to pick the feature set for the final, deployed
    model -- cross-validation below refits selection inside each fold
    instead of reusing this, to avoid leaking test-fold labels into
    feature selection.
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


def _make_cv(n_samples):
    """Leave-one-out for small datasets; stratified k-fold once there's more data."""
    if n_samples <= SMALL_SAMPLE_CV_THRESHOLD:
        return LeaveOneOut()
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def _evaluate_models(X, y):
    """
    Evaluate both models via cross-validation appropriate for the sample
    size. Feature selection is refit inside each fold (via Pipeline) so the
    reported accuracy isn't inflated by selecting features on data that
    includes the held-out fold's own labels.
    """
    cv = _make_cv(len(y))
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
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
        scores[name] = cv_scores.mean()
        print(f"{name} mean CV accuracy: {scores[name]:.4f}  ({cv.__class__.__name__}, {len(cv_scores)} folds)")

    best_name = max(scores, key=scores.get)
    return best_name, models[best_name], scores


def _save_feature_importance(model, feature_names, output_path):
    """Save feature importance bar chart and print ranking."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    importances.plot(kind="bar")
    plt.title("Module 3 Feature Importance")
    plt.ylabel("Importance")
    plt.xlabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("\nMost important features:")
    for feature, score in importances.items():
        print(f"  {feature}: {score:.4f}")


def train_module3_model():
    """Train best model on full data and save model + feature importance."""
    data_path = MASTER_DATASET_PATH
    model_path = os.path.join(BASE_DIR, "models", "module3_model.pkl")
    figure_path = os.path.join(OUTPUT_DIR, "feature_importance.png")

    X, y = _prepare_data(data_path)
    best_name, best_model, _ = _evaluate_models(X, y)

    selected_features = _select_features(X, y)
    X_selected = X[selected_features]
    best_model.fit(X_selected, y)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "features": selected_features,
            # Training-set means for the selected features, so predict.py
            # can fill in a feature it can't compute (e.g. a camera view
            # wasn't provided) the same principled way training data gaps
            # were filled -- never from the single new lift being scored.
            "feature_means": X_selected.mean().to_dict(),
        },
        model_path,
    )
    print(f"\nBest model: {best_name}")
    print(f"Saved model to: {model_path}")

    _save_feature_importance(best_model, selected_features, figure_path)
    print(f"Saved feature importance chart to: {figure_path}")

    return best_model


if __name__ == "__main__":
    train_module3_model()
