"""
Module 4 (bar_path) — attention-LSTM on the raw resampled bar trajectory.

Unlike RF/XGBoost (bar_path_train.py / bar_path_train_xgb.py), which only
see 7 hand-summarised numbers per lift, this model reads the actual
per-frame (x, y) bar path -- data/processed/module4/bar_path_sequences.npz,
built by bar_path_sequences.py -- so it can in principle learn WHEN in the
lift a deviation happens, not just how much deviation happened overall.

Caveat that matters for how these results should be read: there are only
58 sequences to train on. The README's spec (LSTM 128 -> 64 + attention)
is sized for a much larger dataset; used as-is here it would almost
certainly overfit 58 examples. This script intentionally uses a smaller
network (32 hidden units, single layer, dropout 0.4, weight decay) and
the SAME stratified k-fold CV discipline as the tabular models, so its
reported accuracy is an honest comparison point rather than an optimistic
train-set number. Do not read a strong-looking CV score here as proof the
architecture generalises -- 58 sequences is not enough to be confident of
that regardless of the number.

Model: single-layer LSTM over the (150, 2) trajectory -> additive
(Bahdanau-style) attention pooling over the LSTM outputs -> linear -> 1
sigmoid unit (P(bad)). The attention weights double as an explanation:
they show which portion of the lift (early pull vs. catch vs. jerk drive)
most influenced the prediction.

Run:
    python -m src.data.module4_bar_path.bar_path_train_lstm
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch import nn

from .bar_path_train import RISK_LOW_MAX, RISK_MODERATE_MAX, risk_band
from .bar_path_sequences import SEQUENCES_PATH_NAME
from .config import BarPathConfig
from .utils import setup_logger

logger = setup_logger(__name__)

HIDDEN_SIZE = 32
DROPOUT = 0.4
WEIGHT_DECAY = 1e-3
LEARNING_RATE = 1e-3
MAX_EPOCHS = 60
PATIENCE = 10  # early stopping
N_FOLDS = 5
RANDOM_STATE = 42


class AttentionLSTM(nn.Module):
    def __init__(self, input_size: int = 2, hidden_size: int = HIDDEN_SIZE, dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.Linear(hidden_size, 1)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x):
        outputs, _ = self.lstm(x)  # (batch, T, hidden)
        outputs = self.dropout(outputs)

        attn_scores = self.attention(outputs).squeeze(-1)  # (batch, T)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, T)
        context = torch.einsum("bt,bth->bh", attn_weights, outputs)  # (batch, hidden)

        logits = self.classifier(context).squeeze(-1)  # (batch,)
        return logits, attn_weights


def load_sequence_data(config: BarPathConfig):
    path = config.root_dir / "data" / "processed" / "module4" / SEQUENCES_PATH_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"No sequence dataset at {path}. Run "
            "`python -m src.data.module4_bar_path.bar_path_sequences` first."
        )
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"], data["video_ids"]


def _train_one_fold(X_train, y_train, X_val, y_val, seed: int = RANDOM_STATE):
    torch.manual_seed(seed)
    model = AttentionLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(X_val_t)
            val_pos_weight = torch.tensor(
                [(y_train == 0).sum() / max((y_train == 1).sum(), 1)], dtype=torch.float32
            )
            val_loss = nn.functional.binary_cross_entropy_with_logits(
                val_logits, torch.tensor(y_val, dtype=torch.float32), pos_weight=val_pos_weight
            ).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model


def evaluate_cv(X: np.ndarray, y: np.ndarray) -> tuple[dict, np.ndarray]:
    n_folds = min(N_FOLDS, int(np.bincount(y).min()))
    if n_folds < 2:
        raise ValueError("Not enough samples per class for cross-validation")

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    pred_proba = np.zeros(len(y), dtype=float)

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        model = _train_one_fold(X[train_idx], y[train_idx], X[val_idx], y[val_idx])
        model.eval()
        with torch.no_grad():
            logits, _ = model(torch.tensor(X[val_idx], dtype=torch.float32))
            pred_proba[val_idx] = torch.sigmoid(logits).numpy()
        logger.info("Fold %s/%s done", fold_idx + 1, n_folds)

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


def main() -> None:
    config = BarPathConfig()
    X, y, video_ids = load_sequence_data(config)
    logger.info("Loaded %s sequences, shape=%s (%s good / %s bad)",
                len(y), X.shape, int((y == 0).sum()), int((y == 1).sum()))

    metrics, cv_proba = evaluate_cv(X, y)
    logger.info("Cross-validated metrics (%s-fold): %s", metrics["n_folds"], metrics)

    logger.info("Refitting final model on all %s sequences (80/20 internal split for early stopping)", len(y))
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.permutation(len(y))
    split = max(1, int(len(y) * 0.8))
    train_idx, val_idx = idx[:split], idx[split:]
    final_model = _train_one_fold(X[train_idx], y[train_idx], X[val_idx], y[val_idx])

    models_dir = config.root_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "bar_path_lstm.pt"
    torch.save(final_model.state_dict(), model_path)
    logger.info("Saved trained LSTM weights to %s", model_path)

    df_path = config.interim_output_dir / "bar_path_cv_predictions_lstm.csv"
    import pandas as pd
    pred_df = pd.DataFrame({
        "video_id": video_ids,
        "label": y,
        "cv_risk_score": cv_proba,
    })
    pred_df["cv_risk_band"] = pred_df["cv_risk_score"].apply(risk_band)
    pred_df.to_csv(df_path, index=False)
    logger.info("Saved per-lift CV predictions to %s", df_path)

    report_path = models_dir / "bar_path_lstm_report.json"
    report = {
        "model": "lstm",
        "n_samples": len(y),
        "sequence_length": X.shape[1],
        "hidden_size": HIDDEN_SIZE,
        "cv_metrics": metrics,
        "risk_bands": {
            "Low": f"< {RISK_LOW_MAX}",
            "Moderate": f"{RISK_LOW_MAX} - {RISK_MODERATE_MAX}",
            "High": f">= {RISK_MODERATE_MAX}",
        },
        "caveat": (
            "Trained and evaluated on only 58 sequences. CV metrics are "
            "honest (stratified k-fold, no leakage) but the sample size is "
            "too small to be confident the model generalises beyond this "
            "dataset. Compare directly against bar_path_rf_report.json / "
            "bar_path_xgb_report.json before deciding whether to include "
            "this model in an ensemble."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Saved training report to %s", report_path)

    # Auto-run the ensemble-readiness gate right after retraining, since
    # this is the moment new numbers actually exist to check against. This
    # only prints a recommendation -- it never edits ENSEMBLE_MODELS itself.
    from .bar_path_ensemble_check import main as check_ensemble_readiness
    print()
    check_ensemble_readiness()


if __name__ == "__main__":
    main()
