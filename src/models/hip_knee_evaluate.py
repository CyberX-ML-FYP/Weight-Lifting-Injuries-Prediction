"""
Hip & Knee Analysis — Model evaluation pipeline.

Loads the already-trained model (``models/hip_knee_lstm.pth``) and fitted
label encoder (``models/hip_knee_label_encoder.pkl``), reproduces the exact
stratified 80/20 train/test split used by ``src.models.hip_knee_train``
(same ``random_state``/``test_size``/``stratify`` arguments), and evaluates
the model on the held-out test split only.

Saves:
    reports/hip_knee_metrics.json   — accuracy, balanced accuracy, macro/weighted
                                       precision/recall/F1, per-class metrics,
                                       confusion matrix, misclassified samples.
    reports/classification_report.txt — sklearn's classification_report text.
    reports/confusion_matrix.png    — confusion-matrix heatmap.
    reports/training_curves.png     — train/val loss & accuracy curves, built
                                       from models/hip_knee_training_history.json.

This module does NOT retrain the model, does NOT regenerate the dataset, and
does NOT modify the saved model weights — it only reads existing artifacts
and writes new report files.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe: this script only saves figures, never shows them
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.features.hip_knee_config import BASE_DIR, CLASS_NAMES
from src.features.hip_knee_logger import get_logger
from src.models.hip_knee_lstm import HipKneeLSTMClassifier
from src.models.hip_knee_train import (
    DEFAULT_X_PATH,
    DEFAULT_Y_PATH,
    HISTORY_PATH,
    LABEL_ENCODER_PATH,
    MODEL_PATH,
    TrainingConfig,
    load_dataset,
)

logger = get_logger(__name__)

# ── Output report paths (top-level reports/ dir, per evaluation-pipeline spec) ─
REPORTS_DIR: Path = BASE_DIR / "reports"
METRICS_JSON_PATH: Path = REPORTS_DIR / "hip_knee_metrics.json"
CLASSIFICATION_REPORT_PATH: Path = REPORTS_DIR / "classification_report.txt"
CONFUSION_MATRIX_PNG_PATH: Path = REPORTS_DIR / "confusion_matrix.png"
TRAINING_CURVES_PNG_PATH: Path = REPORTS_DIR / "training_curves.png"


def load_label_encoder(path: Path = LABEL_ENCODER_PATH) -> LabelEncoder:
    """Load the ``LabelEncoder`` that was fitted during training."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_trained_model(
    model_path: Path,
    input_size: int,
    num_classes: int,
    config: TrainingConfig,
) -> HipKneeLSTMClassifier:
    """Rebuild ``HipKneeLSTMClassifier`` with the training-time architecture and load its saved weights."""
    model = HipKneeLSTMClassifier(
        input_size=input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_classes=num_classes,
        dropout=config.dropout,
    )
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def reproduce_test_split(
    X: np.ndarray,
    y: np.ndarray,
    label_encoder: LabelEncoder,
    config: TrainingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reproduce the exact stratified 80/20 split used during training.

    Uses ``label_encoder.transform`` (not ``fit_transform``) on the already
    fitted encoder, and the same ``test_size``/``random_state``/``stratify``
    arguments as ``src.models.hip_knee_train.prepare_splits`` so the resulting
    test set is identical to the one used to select the saved checkpoint.
    """
    y_encoded = label_encoder.transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y_encoded,
        shuffle=True,
    )
    return X_train, X_test, y_train, y_test


def predict(model: HipKneeLSTMClassifier, X: np.ndarray) -> np.ndarray:
    """Run inference on ``X`` and return predicted class indices."""
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return logits.argmax(dim=1).numpy()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: tuple[str, ...]) -> dict:
    """Compute overall + per-class accuracy/precision/recall/F1/support/balanced-accuracy."""
    n_classes = len(class_names)
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        y_true, y_pred, labels=range(n_classes), zero_division=0,
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0,
    )

    per_class = {
        class_names[i]: {
            "accuracy": float(recall_per[i]),  # per-class accuracy == recall for that class
            "precision": float(precision_per[i]),
            "recall": float(recall_per[i]),
            "f1_score": float(f1_per[i]),
            "support": int(support_per[i]),
        }
        for i in range(n_classes)
    }

    return {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "n_test_samples": int(len(y_true)),
        "per_class": per_class,
    }


def find_misclassified(y_true: np.ndarray, y_pred: np.ndarray, class_names: tuple[str, ...]) -> list[dict]:
    """Return a list of ``{index, true_label, predicted_label}`` for every wrong prediction."""
    return [
        {"index": int(i), "true_label": class_names[t], "predicted_label": class_names[p]}
        for i, (t, p) in enumerate(zip(y_true, y_pred))
        if t != p
    ]


def build_evaluation_report(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: tuple[str, ...],
) -> tuple[dict, np.ndarray, list[dict]]:
    """Bundle metrics, confusion matrix and misclassified-sample list into one report."""
    metrics = compute_metrics(y_true, y_pred, class_names)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    misclassified = find_misclassified(y_true, y_pred, class_names)

    metrics["confusion_matrix"] = cm.tolist()
    metrics["confusion_matrix_labels"] = list(class_names)
    metrics["misclassified_samples"] = misclassified
    metrics["n_misclassified"] = len(misclassified)
    return metrics, cm, misclassified


def save_metrics_json(metrics: dict, path: Path = METRICS_JSON_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Saved metrics JSON -> %s", path)


def save_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: tuple[str, ...], path: Path = CLASSIFICATION_REPORT_PATH,
) -> str:
    report_text = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_text, encoding="utf-8")
    logger.info("Saved classification report -> %s", path)
    return report_text


def plot_confusion_matrix(cm: np.ndarray, class_names: tuple[str, ...], path: Path = CONFUSION_MATRIX_PNG_PATH) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Hip & Knee — Confusion Matrix (Test Set)")

    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix plot -> %s", path)


def plot_training_curves(history_path: Path = HISTORY_PATH, path: Path = TRAINING_CURVES_PNG_PATH) -> dict:
    """Plot train/val loss and train/val accuracy curves from the saved training history."""
    history = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = history["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train Accuracy")
    axes[1].plot(epochs, history["val_acc"], label="Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("Hip & Knee LSTM — Training Curves")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved training curves plot -> %s", path)
    return history


def _print_header(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


def _print_overall_metrics(metrics: dict) -> None:
    print(f"Overall Test Accuracy   : {metrics['test_accuracy']:.4f}")
    print(f"Balanced Accuracy       : {metrics['balanced_accuracy']:.4f}")
    print(f"Precision (macro)       : {metrics['precision_macro']:.4f}")
    print(f"Recall (macro)          : {metrics['recall_macro']:.4f}")
    print(f"F1 Score (macro)        : {metrics['f1_macro']:.4f}")


def _print_per_class_accuracy(cm: np.ndarray, class_names: tuple[str, ...]) -> None:
    print("\nPer-class Accuracy:")
    for i, name in enumerate(class_names):
        row_total = cm[i].sum()
        acc = cm[i, i] / row_total if row_total else 0.0
        print(f"  {name:>6}: {acc:.4f}  (n={row_total})")


def _print_confusion_matrix(cm: np.ndarray, class_names: tuple[str, ...]) -> None:
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print("        " + "".join(f"{name:>8}" for name in class_names))
    for i, name in enumerate(class_names):
        row = "".join(f"{value:>8}" for value in cm[i])
        print(f"  {name:>6}{row}")


def _print_misclassified(misclassified: list[dict], n_total: int) -> None:
    print(f"\nMisclassified Samples: {len(misclassified)} / {n_total}")
    for sample in misclassified:
        print(f"  idx={sample['index']:>3}  true={sample['true_label']:<6}  predicted={sample['predicted_label']}")


def print_evaluation_summary(metrics: dict, cm: np.ndarray, class_names: tuple[str, ...], misclassified: list[dict]) -> None:
    _print_header("HIP & KNEE — MODEL EVALUATION SUMMARY")
    _print_overall_metrics(metrics)
    _print_per_class_accuracy(cm, class_names)
    _print_confusion_matrix(cm, class_names)
    _print_misclassified(misclassified, metrics["n_test_samples"])
    print("=" * 60)


def main(
    x_path: Path = DEFAULT_X_PATH,
    y_path: Path = DEFAULT_Y_PATH,
    model_path: Path = MODEL_PATH,
    label_encoder_path: Path = LABEL_ENCODER_PATH,
    history_path: Path = HISTORY_PATH,
    config: TrainingConfig | None = None,
) -> dict:
    """Run the full evaluation pipeline end-to-end and return the metrics dict."""
    config = config or TrainingConfig()

    X, y = load_dataset(x_path, y_path)
    label_encoder = load_label_encoder(label_encoder_path)
    class_names = tuple(label_encoder.classes_) or CLASS_NAMES

    _, X_test, _, y_test = reproduce_test_split(X, y, label_encoder, config)
    model = load_trained_model(model_path, input_size=X.shape[-1], num_classes=len(class_names), config=config)

    y_pred = predict(model, X_test)
    metrics, cm, misclassified = build_evaluation_report(y_test, y_pred, class_names)

    save_metrics_json(metrics)
    save_classification_report(y_test, y_pred, class_names)
    plot_confusion_matrix(cm, class_names)
    plot_training_curves(history_path)

    print_evaluation_summary(metrics, cm, class_names, misclassified)
    logger.info(
        "Evaluation complete | accuracy=%.4f balanced_accuracy=%.4f f1_macro=%.4f",
        metrics["test_accuracy"], metrics["balanced_accuracy"], metrics["f1_macro"],
    )
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained Hip & Knee LSTM lift-quality classifier.")
    parser.add_argument("--x-path", type=Path, default=DEFAULT_X_PATH)
    parser.add_argument("--y-path", type=Path, default=DEFAULT_Y_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--label-encoder-path", type=Path, default=LABEL_ENCODER_PATH)
    parser.add_argument("--history-path", type=Path, default=HISTORY_PATH)
    parser.add_argument("--test-size", type=float, default=TrainingConfig().test_size)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cli_config = TrainingConfig(test_size=args.test_size)
    main(
        x_path=args.x_path,
        y_path=args.y_path,
        model_path=args.model_path,
        label_encoder_path=args.label_encoder_path,
        history_path=args.history_path,
        config=cli_config,
    )
