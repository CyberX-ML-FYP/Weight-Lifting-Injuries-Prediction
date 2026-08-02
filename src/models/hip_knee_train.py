"""
Hip & Knee Analysis — LSTM training pipeline.

Loads the combined keypoint-sequence dataset produced by
``src.data.hip_knee_build_dataset`` (``data/processed/hip_knee/combined_X.npy``
and ``combined_y.npy``), splits it into a stratified 80/20 train/test set,
and trains ``HipKneeLSTMClassifier`` (reused from ``src.models.hip_knee_lstm``)
with:

- A mini-batch training loop (``torch.utils.data.DataLoader``).
- Early stopping on validation loss (``EarlyStopping``).
- A best-model checkpoint callback that persists weights to disk on every
  improvement (``ModelCheckpoint``), so the saved model is always the best
  one seen — not just whatever epoch training happened to stop on.
- A ``ReduceLROnPlateau`` learning-rate scheduler.
- Per-epoch console + log output of train/validation loss and accuracy.

Saves three artifacts:
    models/hip_knee_lstm.pth               — best model weights (state_dict)
    models/hip_knee_label_encoder.pkl      — fitted LabelEncoder
    models/hip_knee_training_history.json  — per-epoch metrics + run metadata

This module ONLY trains the model. Evaluation (confusion matrix, ROC/AUC,
precision/recall/F1) is a separate, later stage handled by
``src.models.hip_knee_evaluate`` and is intentionally not invoked here.
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from src.features.hip_knee_config import BASE_DIR, PROCESSED_DIR
from src.features.hip_knee_logger import get_logger, log_execution_time
from src.models.hip_knee_lstm import HipKneeLSTMClassifier

logger = get_logger(__name__)

# ── Output paths (top-level models/ directory, per training-pipeline spec) ───
MODELS_DIR: Path = BASE_DIR / "models"
MODEL_PATH: Path = MODELS_DIR / "hip_knee_lstm.pth"
LABEL_ENCODER_PATH: Path = MODELS_DIR / "hip_knee_label_encoder.pkl"
HISTORY_PATH: Path = MODELS_DIR / "hip_knee_training_history.json"

# ── Default input dataset paths ───────────────────────────────────────────────
DEFAULT_X_PATH: Path = PROCESSED_DIR / "combined_X.npy"
DEFAULT_Y_PATH: Path = PROCESSED_DIR / "combined_y.npy"


@dataclass
class TrainingConfig:
    """Hyperparameters for one training run (all overridable via CLI flags)."""

    epochs: int = 200
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 2e-4
    hidden_size: int = 32
    num_layers: int = 2
    dropout: float = 0.3
    use_class_weights: bool = True
    test_size: float = 0.2
    random_state: int = 42
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 8
    lr_scheduler_min_lr: float = 1e-6


def load_dataset(x_path: Path = DEFAULT_X_PATH, y_path: Path = DEFAULT_Y_PATH) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the combined keypoint-sequence dataset.

    Raises:
        FileNotFoundError: If either ``.npy`` file is missing (i.e. the
            dataset-build stage has not been run yet).
        ValueError: If ``X`` and ``y`` have a mismatched number of rows.
    """
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Combined dataset not found ({x_path}, {y_path}). Run "
            "`python -m src.data.hip_knee_build_dataset` first."
        )

    X = np.load(x_path)
    y = np.load(y_path)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X/y row count mismatch: {X.shape[0]} != {y.shape[0]}")

    logger.info("Loaded dataset | X.shape=%s y.shape=%s classes=%s", X.shape, y.shape, sorted(set(y.tolist())))
    return X, y


def prepare_splits(
    X: np.ndarray,
    y: np.ndarray,
    config: TrainingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """Label-encode ``y`` and perform a stratified 80/20 train/test split."""
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y_encoded,
        shuffle=True,
    )
    logger.info(
        "Stratified split | train=%d test=%d classes=%s",
        len(X_train), len(X_test), list(label_encoder.classes_),
    )
    return X_train, X_test, y_train, y_test, label_encoder


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """Wrap a (X, y) split in a ``DataLoader`` of float32/long tensors."""
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=shuffle, num_workers=0)


def compute_class_weights(y_train: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Inverse-frequency class weights for ``nn.CrossEntropyLoss(weight=...)``.

    The Hip & Knee dataset is imbalanced (roughly 65% Good / 35% Poor), which
    otherwise biases the model toward predicting the majority class "Good"
    (confirmed by the pre-improvement confusion matrix: 9/12 Poor samples
    misclassified as Good). Weighting the minority class more heavily in the
    loss counteracts this without altering the dataset itself.
    """
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0  # avoid division by zero for an absent class
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


class EarlyStopping:
    """Stops training when a monitored (lower-is-better) metric stalls."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_value: float | None = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Record the latest monitored value. Returns ``True`` if it is a new best."""
        if self.best_value is None or value < self.best_value - self.min_delta:
            self.best_value = value
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


class ModelCheckpoint:
    """Persists ``model.state_dict()`` to disk whenever the monitored metric improves."""

    def __init__(self, filepath: Path, mode: str = "min") -> None:
        self.filepath = filepath
        self.mode = mode
        self.best_value: float | None = None
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        return value < self.best_value if self.mode == "min" else value > self.best_value

    def step(self, value: float, model: nn.Module, epoch: int) -> bool:
        """Save a checkpoint if ``value`` improves on the best seen so far."""
        if not self._is_better(value):
            return False

        self.best_value = value
        torch.save(model.state_dict(), self.filepath)
        logger.info("Checkpoint saved (epoch=%d, metric=%.4f) -> %s", epoch, value, self.filepath)
        return True


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one training (``optimizer`` given) or evaluation epoch. Returns ``(loss, accuracy)``."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, correct, total = 0.0, 0, 0
    for batch_x, batch_y in loader:
        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        correct += (outputs.argmax(dim=1) == batch_y).sum().item()
        total += batch_x.size(0)

    return total_loss / total, correct / total


def _print_progress_header() -> None:
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10} | {'Train Acc':>10} | {'Val Acc':>9} | {'LR':>9}")
    print("-" * 68)


def _print_progress_row(epoch: int, train_loss: float, val_loss: float, train_acc: float, val_acc: float, lr: float) -> None:
    print(f"{epoch:>6} | {train_loss:>10.4f} | {val_loss:>10.4f} | {train_acc:>10.4f} | {val_acc:>9.4f} | {lr:>9.2e}")


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    config: TrainingConfig,
) -> tuple[HipKneeLSTMClassifier, dict]:
    """
    Train ``HipKneeLSTMClassifier`` with early stopping, LR scheduling and
    checkpointing, restoring the best (lowest validation-loss) weights
    before returning.

    Returns:
        ``(model, history)`` — ``history`` contains per-epoch
        ``train_loss``/``val_loss``/``train_acc``/``val_acc``/``learning_rate``
        lists plus run metadata (``best_val_loss``, ``epochs_trained``,
        ``early_stopped``).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(label_encoder.classes_)
    model = HipKneeLSTMClassifier(
        input_size=X_train.shape[-1],
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_classes=num_classes,
        dropout=config.dropout,
    ).to(device)

    train_loader = _make_loader(X_train, y_train, config.batch_size, shuffle=True)
    val_loader = _make_loader(X_test, y_test, config.batch_size, shuffle=False)

    class_weights = compute_class_weights(y_train, num_classes).to(device) if config.use_class_weights else None
    if class_weights is not None:
        logger.info("Using class-weighted CrossEntropyLoss | weights=%s classes=%s", class_weights.tolist(), list(label_encoder.classes_))
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
        min_lr=config.lr_scheduler_min_lr,
    )
    early_stopping = EarlyStopping(patience=config.early_stopping_patience, min_delta=config.early_stopping_min_delta)
    checkpoint = ModelCheckpoint(MODEL_PATH, mode="min")

    history: dict = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "learning_rate": []}

    _print_progress_header()
    with log_execution_time(logger, "LSTM training"):
        for epoch in range(1, config.epochs + 1):
            train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = _run_epoch(model, val_loader, criterion, optimizer=None)
            current_lr = optimizer.param_groups[0]["lr"]

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["learning_rate"].append(current_lr)

            _print_progress_row(epoch, train_loss, val_loss, train_acc, val_acc, current_lr)
            logger.info(
                "epoch=%03d train_loss=%.4f val_loss=%.4f train_acc=%.4f val_acc=%.4f lr=%.2e",
                epoch, train_loss, val_loss, train_acc, val_acc, current_lr,
            )

            checkpoint.step(val_loss, model, epoch)
            scheduler.step(val_loss)
            early_stopping.step(val_loss)

            if early_stopping.should_stop:
                msg = f"Early stopping triggered at epoch {epoch} (best val_loss={early_stopping.best_value:.4f})"
                logger.info(msg)
                print(f"\n{msg}")
                break

    # The checkpoint on disk is always the best-seen model; reload it so the
    # returned/saved model matches history["best_val_loss"] exactly, even if
    # training continued (and drifted) for several epochs after the best one.
    model.load_state_dict(torch.load(checkpoint.filepath, map_location=device, weights_only=True))
    model.eval()

    history["best_val_loss"] = checkpoint.best_value
    history["epochs_trained"] = len(history["epoch"])
    history["early_stopped"] = early_stopping.should_stop
    history["classes"] = list(label_encoder.classes_)
    return model, history


def save_artifacts(label_encoder: LabelEncoder, history: dict) -> None:
    """Save the fitted label encoder and training history (the model
    weights are already saved incrementally by ``ModelCheckpoint``)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)
    HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")

    logger.info("Saved label encoder -> %s", LABEL_ENCODER_PATH)
    logger.info("Saved training history -> %s", HISTORY_PATH)


def _verify_outputs() -> list[str]:
    """Confirm all three training artifacts were written to disk."""
    return [f"Missing output file: {path}" for path in (MODEL_PATH, LABEL_ENCODER_PATH, HISTORY_PATH) if not path.exists()]


def count_parameters(model: nn.Module) -> int:
    """Total number of (trainable) parameters in ``model``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(
    config: TrainingConfig | None = None,
    x_path: Path = DEFAULT_X_PATH,
    y_path: Path = DEFAULT_Y_PATH,
) -> dict:
    """Run the full training pipeline end-to-end and return the training history."""
    config = config or TrainingConfig()

    X, y = load_dataset(x_path, y_path)
    X_train, X_test, y_train, y_test, label_encoder = prepare_splits(X, y, config)
    model, history = train_model(X_train, y_train, X_test, y_test, label_encoder, config)
    save_artifacts(label_encoder, history)

    n_params = count_parameters(model)
    problems = _verify_outputs()

    print(f"\nModel parameter count: {n_params:,}")
    if problems:
        print("\n⚠ Verification found issues:")
        for problem in problems:
            print(f"  - {problem}")
            logger.error(problem)
    else:
        print("\n✔ Training artifacts saved and verified:")
        print(f"  - {MODEL_PATH}")
        print(f"  - {LABEL_ENCODER_PATH}")
        print(f"  - {HISTORY_PATH}")
        logger.info("Training pipeline completed and verified successfully")

    return history


def _parse_args() -> argparse.Namespace:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description="Train the Hip & Knee LSTM lift-quality classifier.")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--hidden-size", type=int, default=defaults.hidden_size)
    parser.add_argument("--num-layers", type=int, default=defaults.num_layers)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--no-class-weights", action="store_true", help="Disable class-weighted CrossEntropyLoss.")
    parser.add_argument("--patience", type=int, default=defaults.early_stopping_patience, help="Early stopping patience (epochs)")
    parser.add_argument("--test-size", type=float, default=defaults.test_size)
    parser.add_argument("--x-path", type=Path, default=DEFAULT_X_PATH)
    parser.add_argument("--y-path", type=Path, default=DEFAULT_Y_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cli_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_class_weights=not args.no_class_weights,
        early_stopping_patience=args.patience,
        test_size=args.test_size,
    )
    main(cli_config, x_path=args.x_path, y_path=args.y_path)
