# Hip & Knee — Model Explainability Report (Captum Integrated Gradients)

## 1. Purpose

This report explains ONE prediction made by the already-trained, already
validated Hip & Knee LSTM classifier (`models/hip_knee_lstm.pth`), using
[Captum](https://captum.ai/)'s Integrated Gradients (IG) attribution method.

**No retraining was performed. No prediction logic was modified. No
teammate module was touched.** The model is loaded read-only and run only
in `eval()` mode; Captum computes gradients of the model's OUTPUT with
respect to its INPUT — it never updates the model's weights.

## 2. Sample explained

| Field | Value |
|---|---|
| Video | `D:\fyp\Weight-Lifting-Injuries-Prediction\data\raw\hip_knee\Side view\13bad.MOV` |
| Frames used | 20 |
| Predicted class | **Good** |
| Prediction confidence | 52.22% |
| Low-confidence flag | True |

## 3. Method

1. **Load the trained LSTM** — `HipKneeLSTMClassifier`, rebuilt with the
   training-time architecture and its saved `state_dict`
   (`src.models.hip_knee_evaluate.load_trained_model`, the same loader used
   for evaluation/prediction). Weights are frozen/read-only throughout.
2. **Load one feature vector** — the real `(seq_len=20, 16)` normalized
   keypoint sequence produced for the video above, built by reusing the
   exact same pipeline stages `predict_video` uses internally: frame
   extraction, YOLO11-pose keypoint extraction, and
   `normalize_by_body_proportion` (hip-centered, torso-scaled). This is the
   literal tensor the LSTM consumes in production — nothing is re-derived
   or approximated for this step.
3. **Compute Integrated Gradients** — `captum.attr.IntegratedGradients`
   attributes the predicted class's output logit back to every one of the
   16 raw input features at every one of the 20 sampled
   frames, integrating gradients along a straight-line path from a
   zero-vector baseline (a neutral "hip-centered" pose, since the model's
   inputs are already centered on the hip midpoint) to the real input, using
   `n_steps=50`.
   - Convergence delta (completeness-axiom diagnostic, closer to 0 is
     better): **-0.000000**
4. **Produce feature importance** — the raw `(20, 16)` attribution tensor is
   reduced to 4 anatomical joint groups (shoulder, hip, knee, ankle,
   bilateral-combined) by summing `|attribution|` over the x/y components and
   all sampled frames. These 4 joint-group totals are then mapped onto the
   11 requested, human-readable features using the documented weight table
   in §4, and the result is **normalized to sum to 1.0**.

## 4. Anatomical / functional attribution mapping

The 16 raw LSTM input dimensions per frame are 8 bilateral keypoints
(shoulder, hip, knee, ankle) x (x, y). Rather than guessing, each requested
feature's mapping weight is derived directly from the REAL formulas already
used elsewhere in this codebase (`src/features/hip_knee_biomechanics.py`):

- **Hip ROM / Hip Peak** come from the hip flexion angle,
  `angle(shoulder, hip, knee)` -> weighted toward shoulder+hip+knee.
- **Knee ROM / Knee Peak** come from the knee flexion angle,
  `angle(hip, knee, ankle)` -> weighted toward hip+knee+ankle.
- **Sequential Delay** (Rule A), **Synchronization** (Rule C score),
  **Hip Dominance** (Rule B score), **Correlation** (Rule C raw
  coefficient) and **RFD** (Rule D) are all derived from BOTH the hip-angle
  AND knee-angle velocity/acceleration signals together -> weighted equally
  across hip and knee (with a smaller shoulder/ankle contribution, since
  those angles are the vertices/endpoints of the hip and knee angle
  definitions).
- **Confidence** is the LSTM's own softmax confidence — not anatomically
  localized to one joint, so it is weighted uniformly across all 4 groups
  (a global-sensitivity proxy).
- **Anthropometric Features** come from `leg_length` (hip->knee->ankle) and
  `torso_length` (shoulder->hip), the SAME reference lengths the pipeline
  already uses to normalize distance-dependent features
  (`reports/anthropometric_normalization.md`).

| Feature | Shoulder | Hip | Knee | Ankle |
|---|---|---|---|---|
| Hip ROM | 0.250 | 0.500 | 0.250 | 0.000 |
| Knee ROM | 0.000 | 0.250 | 0.500 | 0.250 |
| Hip Peak | 0.250 | 0.500 | 0.250 | 0.000 |
| Knee Peak | 0.000 | 0.250 | 0.500 | 0.250 |
| Sequential Delay | 0.125 | 0.375 | 0.375 | 0.125 |
| Synchronization | 0.125 | 0.375 | 0.375 | 0.125 |
| Hip Dominance | 0.125 | 0.375 | 0.375 | 0.125 |
| Correlation | 0.125 | 0.375 | 0.375 | 0.125 |
| RFD | 0.125 | 0.375 | 0.375 | 0.125 |
| Confidence | 0.250 | 0.250 | 0.250 | 0.250 |
| Anthropometric Features | 0.200 | 0.400 | 0.200 | 0.200 |

**This mapping is a documented heuristic used only to translate raw,
per-keypoint Integrated Gradients attribution into human-readable
categories.** It does not alter the model, its inputs, or its prediction —
see §6 for its limitations.

## 5. Feature importance (normalized, sums to 100%)

| Feature | Normalized importance | Source value (this sample) | Description |
|---|---|---|---|
| Confidence | 12.56% | `prediction_confidence` = 0.522233784198761 | The LSTM's own softmax confidence in the predicted class. |
| Anthropometric Features | 10.24% | `leg_length_reference_px` = 601.8157609381642 | Leg/torso-length reference and the linear ROM/peak-velocity features normalized by it. |
| Knee ROM | 9.13% | `knee_rom` = 46.718524959408285 | Angular range of motion of the knee joint across the lift (degrees). |
| Knee Peak | 9.13% | `knee_peak` = 181.76287866066053 | Peak knee flexion angle reached during the lift (degrees). |
| Sequential Delay | 8.58% | `synchronization_delay` = 0.0 | Rule A - timing delay (s) between peak knee and peak hip angular velocity. |
| Synchronization | 8.58% | `rule_c_score` = 99.90284139054252 | Rule C score (0-100) - how closely the hip and knee velocity signals move together. |
| Hip Dominance | 8.58% | `rule_b_score` = 9.063201055371998 | Rule B score (0-100) - ratio of peak hip to peak knee angular velocity. |
| Correlation | 8.58% | `correlation` = 0.9964037477166987 | Rule C raw Pearson correlation coefficient between hip and knee angular velocity. |
| RFD | 8.58% | `rate_of_force_development` = 0.9952626497930227 | Rule D - ratio of peak hip to peak knee angular acceleration (rate-of-force-development proxy). |
| Hip ROM | 8.02% | `hip_rom` = 43.97989747935128 | Angular range of motion of the hip joint across the lift (degrees). |
| Hip Peak | 8.02% | `hip_peak` = 179.37310837649994 | Peak hip flexion angle reached during the lift (degrees). |

See `reports/feature_importance.csv` for the same table in machine-readable
form, and `reports/integrated_gradients.png` for the attribution heatmap
(time x joint group) and the feature-importance bar chart.

## 6. Limitations

1. **Single-sample explanation** — this report explains one specific video's
   prediction, not the model's general/global behaviour. Re-running against
   a different video (`--video <path>`) can and should give a different
   feature-importance ranking.
2. **Anatomical mapping is a heuristic, not a second model** — Integrated
   Gradients is computed rigorously on the model's real raw input tensor;
   the subsequent grouping of those 16 raw-dimension attributions into the
   11 requested named features (§4) is a documented, deterministic weighting
   scheme based on which joints each metric's real formula depends on, not
   an independent attribution computation for each named feature. Rule
   A-D-derived features (Sequential Delay, Synchronization, Hip Dominance,
   Correlation, RFD) share very similar weights because they are all
   functions of the same two underlying hip/knee angular signals.
3. **Zero-vector baseline** — a neutral, hip-centered "zero pose" was used
   as the IG baseline. A different baseline (e.g. the dataset mean pose)
   would shift the absolute attribution magnitudes, though the *relative*
   ranking across joint groups is expected to be similar for this kind of
   pose-classification model.
4. **Confidence is not spatially localized** — unlike the other 10
   features, "Confidence" has no direct anatomical dependency in the
   biomechanics formulas, so it is assigned a uniform weight across all 4
   joint groups as a global-sensitivity proxy rather than a joint-specific one.
5. **No retraining, no logic changes** — this module only reads existing
   model/threshold/label-encoder artifacts and calls existing, unmodified
   pipeline functions; it does not change `src/models/hip_knee_predict.py`,
   `src/models/hip_knee_lstm.py`, or any teammate module.

## 7. Files generated

- `reports/model_explainability.md` — this report.
- `reports/integrated_gradients.png` — attribution heatmap + feature-importance bar chart.
- `reports/feature_importance.csv` — normalized feature importance table.
