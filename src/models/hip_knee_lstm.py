"""
Hip & Knee Analysis — LSTM lift-quality classifier: model, training, saving.

WHY these improvements help Hip & Knee analysis
------------------------------------------------
- The LSTM architecture and early-stopping training loop are kept
  functionally identical to the research notebook (same layer sizes,
  optimizer, stratified split) so previously validated behaviour is not
  regressed — only type hints, docstrings, structured logging and a
  training-history return value are added.
- ``save_model_artifacts``/``load_model_artifacts`` persist the trained
  weights, label encoder classes AND the adaptive thresholds together
  (Improvement 9), so a saved model is always paired with the exact
  normalization parameters it was trained/scored with, rather than relying
  on a hardcoded tuple living in a notebook cell.
"""
from __future__ import annotations

import copy
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.features.hip_knee_config import MODELS_DIR
from src.features.hip_knee_logger import get_logger, log_execution_time
from src.features.hip_knee_scoring import AdaptiveThresholds, load_thresholds, save_thresholds

logger = get_logger(__name__)


class HipKneeLSTMClassifier(nn.Module):
    """
    LSTM classifier over per-frame lower-body keypoint sequences.

    Same architecture as the original research notebook: a 2-layer LSTM
    followed by a linear head applied to the final time-step's hidden state.
    """

    def __init__(
        self,
        input_size: int = 16,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # nn.LSTM only applies inter-layer dropout when num_layers > 1
            # (a single-layer LSTM has no "between layers" to regularize).
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass; ``x`` has shape ``(batch, seq_len, input_size)``."""
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


def train_hip_knee_classifier(
    X: np.ndarray,
    y_encoded: np.ndarray,
    epochs: int = 100,
    patience: int = 30,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[HipKneeLSTMClassifier, dict[str, list[float]]]:
    """
    Train ``HipKneeLSTMClassifier`` with early stopping on the best test accuracy.

    Args:
        X: Keypoint sequences, shape ``(n_samples, seq_len, n_features)``.
        y_encoded: Integer-encoded labels, shape ``(n_samples,)``.
        epochs: Maximum training epochs.
        patience: Epochs without test-accuracy improvement before stopping.
        learning_rate: Adam optimizer learning rate.
        weight_decay: Adam L2 weight-decay regularization strength, used to
            reduce overfitting on the small lift-count datasets typical for
            this project.
        test_size: Fraction of data held out for evaluation.
        random_state: Split/reproducibility seed.

    Returns:
        ``(model, history)`` where ``history`` contains per-epoch
        ``train_loss``, ``test_loss``, ``train_acc`` and ``test_acc`` lists
        (used later for evaluation plots).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, shuffle=True, stratify=y_encoded,
    )

    x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    model = HipKneeLSTMClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history: dict[str, list[float]] = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    best_acc, best_epoch, best_state, no_improve = 0.0, 0, None, 0

    with log_execution_time(logger, "LSTM training"):
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(x_train_tensor)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                train_pred = model(x_train_tensor)
                test_pred = model(x_test_tensor)
                train_loss = criterion(train_pred, y_train_t).item()
                test_loss = criterion(test_pred, y_test_t).item()
                train_acc = (train_pred.argmax(dim=1) == y_train_t).float().mean().item()
                test_acc = (test_pred.argmax(dim=1) == y_test_t).float().mean().item()

            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)

            logger.info(
                "epoch=%03d train_loss=%.4f test_loss=%.4f train_acc=%.4f test_acc=%.4f",
                epoch + 1, train_loss, test_loss, train_acc, test_acc,
            )

            if test_acc >= best_acc:
                best_acc, best_epoch, no_improve = test_acc, epoch + 1, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                no_improve += 1

            if no_improve >= patience:
                logger.info("Early stopping at epoch %d (best epoch %d, best_acc=%.4f)", epoch + 1, best_epoch, best_acc)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    logger.info("Training complete | best_epoch=%d best_test_acc=%.4f", best_epoch, best_acc)
    return model, history


def save_model_artifacts(
    model: HipKneeLSTMClassifier,
    label_encoder: LabelEncoder,
    thresholds: AdaptiveThresholds,
    output_dir: Path = MODELS_DIR,
    model_name: str = "hip_knee_lstm",
) -> None:
    """
    Save trained weights, label encoder classes and normalization thresholds.

    Args:
        model: Trained classifier.
        label_encoder: Fitted ``LabelEncoder`` used for ``y_encoded``.
        thresholds: Adaptive Rule A-D thresholds learned on the same split.
        output_dir: Directory artifacts are written to.
        model_name: File stem shared by all artifact files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / f"{model_name}.pt")
    joblib.dump(label_encoder, output_dir / f"{model_name}_label_encoder.joblib")
    save_thresholds(thresholds, output_dir / f"{model_name}_thresholds.json")
    logger.info("Saved model artifacts to %s (name=%s)", output_dir, model_name)


def load_model_artifacts(
    output_dir: Path = MODELS_DIR,
    model_name: str = "hip_knee_lstm",
) -> tuple[HipKneeLSTMClassifier, LabelEncoder, AdaptiveThresholds]:
    """Load a previously saved model, label encoder and adaptive thresholds."""
    model = HipKneeLSTMClassifier()
    # weights_only=True avoids unpickling arbitrary objects from the checkpoint
    # file, mitigating insecure-deserialization risk from a tampered/untrusted
    # model file (OWASP A08: Software and Data Integrity Failures).
    state_dict = torch.load(output_dir / f"{model_name}.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    label_encoder = joblib.load(output_dir / f"{model_name}_label_encoder.joblib")
    thresholds = load_thresholds(output_dir / f"{model_name}_thresholds.json")

    logger.info("Loaded model artifacts from %s (name=%s)", output_dir, model_name)
    return model, label_encoder, thresholds
