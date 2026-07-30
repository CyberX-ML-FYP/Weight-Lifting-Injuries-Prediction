# Hip & Knee Confidence-Weighted Rule A-D Scoring

## Scope and Constraint Compliance
- Task scope: Hip & Knee biomechanics and prediction reporting only.
- No retraining performed.
- No dataset regeneration performed.
- No changes made to teammate modules (`module3_arm_analysis`, frontend, visualization).

## Algorithm
1. Compute the original Rule A-D biomechanical metrics exactly as before.
2. Compute per-rule confidence from YOLO pose confidence, aligned to the actual frame dependencies of each rule:
- Rule A confidence: mean confidence at the knee-velocity-peak frame and hip-velocity-peak frame.
- Rule B confidence: mean confidence at the absolute hip-velocity-peak frame and absolute knee-velocity-peak frame.
- Rule C confidence: mean confidence across all analyzed frames (Rule C uses full-signal correlation).
- Rule D confidence: mean confidence at the absolute hip-acceleration-peak frame and absolute knee-acceleration-peak frame.
3. Compute overall confidence as a weighted blend of rule confidences using existing Rule A-D weights.
4. Compute per-rule normalized scores (0-100) using existing adaptive thresholds.
5. Compute confidence-weighted final score by combining each rule score with both:
- its domain importance weight, and
- its rule-specific confidence.
6. Apply low-confidence flagging if confidence is below threshold:
- prediction softmax confidence < 0.60, or
- overall rule confidence < 0.60.

## Formulas
Let the fixed rule weights be:
\[
(w_A, w_B, w_C, w_D) = (0.35, 0.30, 0.20, 0.15)
\]

Rule confidences:
\[
C_A = \frac{c[t_{knee\_peak}] + c[t_{hip\_peak}]}{2}
\]
\[
C_B = \frac{c[t_{|hip\_vel|\_peak}] + c[t_{|knee\_vel|\_peak}]}{2}
\]
\[
C_C = \frac{1}{N}\sum_{t=1}^{N} c[t]
\]
\[
C_D = \frac{c[t_{|hip\_acc|\_peak}] + c[t_{|knee\_acc|\_peak}]}{2}
\]
where \(c[t] \in [0,1]\) is frame-level YOLO keypoint confidence.

Overall confidence:
\[
C_{overall} = w_A C_A + w_B C_B + w_C C_C + w_D C_D
\]

Confidence-weighted rule blend (on normalized rule scores \(S_A..S_D\)):
\[
S_{blend} = \frac{\sum_{i\in\{A,B,C,D\}} w_i C_i S_i}{\sum_{i\in\{A,B,C,D\}} w_i C_i}
\]
Fallback: if denominator is near zero, use the original weight-only blend.

Final score mapping (backward-compatible semantics):
\[
S_{final} =
\begin{cases}
50 + 0.5\,S_{blend}, & \text{if predicted class is Good} \\
0.5\,S_{blend}, & \text{if predicted class is Poor}
\end{cases}
\]

Low-confidence rule:
\[
\text{low\_confidence} = (p_{model} < 0.60) \lor (C_{overall} < 0.60)
\]

## Why This Helps
- Reduces over-trust in noisy/occluded joints by down-weighting only affected rules.
- Preserves high-confidence rules instead of uniformly penalizing the whole clip.
- Keeps existing Rule A-D biomechanics and adaptive-threshold scoring intact.
- Makes uncertainty explicit with a low-confidence flag.

## Advantages
- Per-rule reliability is explicit and auditable.
- Better robustness to partial occlusion and motion blur.
- Backward compatibility preserved for existing dataclass call sites.
- `prediction.json`, console output, and `PredictionReport` now expose confidence internals.

## Limitations
- Confidence comes from 2D detector estimates, not ground-truth uncertainty.
- Peak-frame confidence can still be sensitive to momentary detector noise.
- Confidence threshold (0.60) is heuristic and may be tuned per dataset.
- Does not change LSTM weights; it only improves score reliability and reporting.

## Files Modified
- `src/features/hip_knee_config.py`
- `src/features/hip_knee_biomechanics.py`
- `src/features/hip_knee_scoring.py`
- `src/models/hip_knee_predict.py`
- `reports/confidence_weighted_rules.md`

## Verification Run (No Retraining)
Command used:
`python -m src.models.hip_knee_predict --video "data/raw/hip_knee/Side view/13bad.MOV" --skip-video`

Observed key outputs:
- Predicted class: Good
- Model confidence: 0.5260
- Overall rule confidence: 0.9939
- Confidence-weighted final score: 58.31
- Low-confidence flag: True (triggered by model confidence < 0.60)
