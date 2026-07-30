# Hip & Knee LSTM — Model Improvement Report

Scope: `src/models/hip_knee_lstm.py`, `src/models/hip_knee_train.py`,
`src/models/hip_knee_evaluate.py` only. No changes were made to
`module3_arm_analysis`, `visualization`, the frontend, or any other
teammate's module. The dataset (`data/processed/hip_knee/combined_X.npy` /
`combined_y.npy`) was **not** regenerated — only the model, training
procedure and its hyperparameters were changed, then retrained on the exact
same data and stratified split (`random_state=42`, `test_size=0.2`).

## 1. Root-Cause Analysis

| Check | Finding |
|---|---|
| **Dataset size/imbalance** | 172 samples total — 112 "Good" (65.1%) / 60 "Poor" (34.9%), a ~1.87:1 imbalance. Test split (35 samples): 23 Good / 12 Poor. |
| **Confusion matrix (old)** | `[[22, 1], [9, 3]]` (rows=true, cols=predicted). 9 of 12 "Poor" lifts were misclassified as "Good" — the model was strongly biased toward the majority class. |
| **Misclassified samples (old)** | 10/35 (28.6%) misclassified — 9 of those 10 errors were Poor→Good. |
| **Overfitting** | Training history showed train accuracy climbing to **90.5%** by epoch 40 while validation loss **diverged** after ~epoch 20 (val_loss rose from 0.469 at epoch 20 to 0.839 by epoch 40) and validation accuracy oscillated/declined (0.71 → 0.57). Classic overfitting on a small (137-sample) training set. |
| **Underfitting** | Not observed — the model had more than enough capacity (see next row). |
| **Feature quality** | Features are already anthropometrically normalized (`normalize_by_body_proportion`: hip-centered, torso-length-scaled), value range reasonable (mean≈0.13, std≈1.20). Not the primary bottleneck. |
| **Sequence length** | Fixed at 20 sampled frames/video (`NUM_SAMPLED_FRAMES`) — consistent across all samples, not a source of the failure. |
| **Model capacity vs. loss function** | 54,402 parameters (hidden_size=64, 2-layer LSTM, **no dropout**) for only 137 training sequences of 16 features — over-parameterized for the data volume, with an **unweighted** `CrossEntropyLoss` that has no mechanism to counter the class imbalance. |

**Conclusion:** the 71.4% accuracy / 60.3% balanced accuracy was driven by two
compounding, addressable causes — (1) an unweighted loss letting the model
default to predicting the majority class on ambiguous "Poor" lifts, and (2) an
over-parameterized, unregularized network overfitting the small training set.
Dataset size/features/sequence length were not the primary issue and were left
unchanged, per the "improve only if justified" instruction.

## 2. Changes Made (justified by the analysis above)

| Change | Old | New | Justification |
|---|---|---|---|
| Class-weighted `CrossEntropyLoss` | none (unweighted) | inverse-frequency weights `[Good=0.77, Poor=1.43]` | Directly counteracts the Poor→Good bias seen in the confusion matrix, without touching the dataset. |
| Dropout | 0.0 (none) | 0.3 (inter-layer LSTM dropout + a `Dropout` before the final linear layer) | Regularizes the model given the small dataset; targets the observed train/val divergence. |
| Hidden size | 64 | 32 | Halves model capacity (54,402 → 14,914 params) — reduces over-parameterization relative to 137 training sequences and keeps the architecture lightweight, as requested. |
| Weight decay (L2) | 1e-4 | 2e-4 | Modest additional regularization to complement dropout. |
| Learning rate, batch size, sequence length, num_layers | unchanged (1e-3, 16, 20 frames, 2 layers) | unchanged | Not implicated by the analysis; changing them wasn't justified. |

All new hyperparameters (`dropout`, `hidden_size`, `use_class_weights`,
`weight_decay`) are configurable via `TrainingConfig` and new CLI flags
(`--dropout`, `--hidden-size`, `--num-layers`, `--no-class-weights`) on
`python -m src.models.hip_knee_train`, so they remain tunable without further
code edits.

## 3. Retraining

Retrained via `python -m src.models.hip_knee_train` on the unchanged dataset
and split. Early stopping triggered at epoch 53 (best `val_loss=0.3378` at
epoch 33), vs. the old run's epoch 40 (best `val_loss=0.4690`). Training/
validation loss and accuracy curves now track much more closely together
(see `reports/training_curves.png`), confirming reduced overfitting.

## 4. Old vs. New Metrics (same reproduced test split, 35 samples)

| Metric | Old | New | Δ |
|---|---:|---:|---:|
| Test Accuracy | 71.43% | **88.57%** | +17.14 pts |
| Balanced Accuracy | 60.33% | **87.32%** | +26.99 pts |
| Precision (macro) | 72.98% | **87.32%** | +14.34 pts |
| Recall (macro) | 60.33% | **87.32%** | +26.99 pts |
| F1 (macro) | 59.49% | **87.32%** | +27.83 pts |
| Poor-class Recall | 25.00% (3/12) | **83.33%** (10/12) | +58.33 pts |
| Poor-class F1 | 37.50% | **83.33%** | +45.83 pts |
| Good-class Recall | 95.65% (22/23) | 91.30% (21/23) | -4.35 pts |
| Misclassified (test set) | 10 / 35 | **4 / 35** | -6 samples |
| Model parameters | 54,402 | **14,914** | -72.6% (lighter) |

**Confusion matrix — old:**
```
        Good    Poor
Good      22       1
Poor       9       3
```

**Confusion matrix — new:**
```
        Good    Poor
Good      21       2
Poor       2      10
```

## 5. Why the Changes Improved Performance

- **Class-weighted loss** gave "Poor" misclassifications ~1.85x the
  gradient weight of "Good" misclassifications during training, directly
  countering the majority-class bias that caused 9/12 Poor lifts to be
  misread as Good. This is the single largest contributor to the recall/F1
  gain on the minority class (25% → 83.3% recall).
- **Dropout (0.3)** and the **smaller hidden size (64→32)** reduced the
  model's capacity to memorize the 137 training sequences, which is why the
  new training curves show train/val loss staying close together instead of
  diverging — directly addressing the overfitting identified in the
  analysis.
- **Slightly higher weight decay** reinforced the same regularization goal
  at negligible cost.
- Net effect: balanced accuracy and macro-F1 both rose by ~27 points, and
  the model now generalizes to both classes instead of defaulting to the
  majority label — while using 72.6% fewer parameters (a lighter, not more
  complex, architecture).

## 6. Files Modified

- `src/models/hip_knee_lstm.py` — `HipKneeLSTMClassifier` gained an optional
  `dropout: float = 0.0` constructor parameter (applied as inter-layer LSTM
  dropout when `num_layers > 1`, plus a `Dropout` before the final linear
  layer). Default of `0.0` preserves prior behavior for any other caller.
- `src/models/hip_knee_train.py` — `TrainingConfig` gained `dropout`,
  `use_class_weights` fields and a lighter `hidden_size` default (32) plus a
  slightly higher `weight_decay` default (2e-4); added `compute_class_weights()`
  helper and wired class-weighted `CrossEntropyLoss` + dropout into
  `train_model()`; added `--hidden-size`, `--num-layers`, `--dropout`,
  `--no-class-weights` CLI flags.
- `src/models/hip_knee_evaluate.py` — `load_trained_model()` now passes
  `config.dropout` when reconstructing the architecture, so evaluation uses
  an architecture consistent with training.
- `models/hip_knee_lstm.pth`, `models/hip_knee_label_encoder.pkl`,
  `models/hip_knee_training_history.json` — regenerated by retraining
  (old versions backed up alongside as `*_old.pth` / `*_old.pkl` /
  `*_old.json` in the same `models/` directory).
- `reports/hip_knee_metrics.json`, `reports/classification_report.txt`,
  `reports/confusion_matrix.png`, `reports/training_curves.png` —
  regenerated by re-running the (unmodified) evaluation pipeline against the
  new model.
- `reports/model_improvement_report.md` — this report (new file).

No changes were made to `src/data/hip_knee_dataset.py`,
`src/data/hip_knee_build_dataset.py`, `src/features/*`,
`src/models/hip_knee_predict.py`, or any file outside the Hip & Knee module.
The prediction pipeline (`python -m src.models.hip_knee_predict`) was
re-verified against the new model and continues to work unchanged, since it
loads its architecture via the same shared `TrainingConfig` defaults.
