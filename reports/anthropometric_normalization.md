# Hip & Knee Biomechanics — Anthropometric Normalization

Scope: `src/features/hip_knee_biomechanics.py` and `src/models/hip_knee_predict.py`
only. **The LSTM model was NOT retrained, its weights were NOT changed, and
the training dataset was NOT regenerated** — this change only adds new,
additional biomechanical measurements computed post-hoc from the same
extracted keypoints. `module3_arm_analysis`, `visualization`, the frontend,
and all other teammate modules were left untouched.

## 1. Problem

Several biomechanical quantities in this pipeline are measured directly in
**pixel space** (raw image coordinates from YOLO11-pose). A raw pixel
distance or pixel-per-second velocity is **not comparable** across:

- athletes of different heights/limb lengths,
- videos filmed at different camera distances or zoom levels,
- different camera resolutions/aspect ratios.

The existing Rule A-D metrics (delay, hip-dominance ratio, synchronization
correlation, RFD ratio) and the angular Hip/Knee ROM & peak angle are already
computed from **joint angles** (degrees), which are inherently scale- and
translation-invariant (see §4 for the proof) — they did **not** need
normalization and were left unchanged.

However, two genuinely **distance-dependent** quantities were identified
that had no anthropometric normalization:

1. **Linear (vertical, pixel-space) range of motion** of the hip and knee
   joints — how far the joint physically travels during the lift.
2. **Linear (pixel/second) peak vertical velocity** of the hip and knee
   joints — how fast the joint physically moves.

These are new metrics added specifically to be normalized (they did not
exist as normalized quantities before). They complement, rather than
replace, the existing angle-based Rule A-D / ROM metrics.

## 2. Reference Lengths

For each analyzed clip (sequence of `BodyKeypoints`), three candidate
anthropometric reference lengths are computed per frame, then reduced with
the **median** across all frames in the clip (robust to single-frame pose
jitter/occlusion outliers):

| Reference length | Formula (per frame) |
|---|---|
| **Leg length** `L_leg` | $\lVert hip - knee \rVert + \lVert knee - ankle \rVert$ (thigh + shank segments, bilateral average of left/right) |
| **Torso length** `L_torso` | $\lVert \text{shoulder}_{mid} - \text{hip}_{mid} \rVert$, where $\text{shoulder}_{mid} = \frac{\text{left\_shoulder} + \text{right\_shoulder}}{2}$, $\text{hip}_{mid} = \frac{\text{left\_hip} + \text{right\_hip}}{2}$ |
| **Shoulder width** `L_shoulder` | $\lVert \text{left\_shoulder} - \text{right\_shoulder} \rVert$ |

$$L_{ref} = \mathrm{median}_{t=1..T}\big(L_{leg}(t)\big)$$

**Leg length was chosen as the primary reference** (`primary_reference`)
because Hip/Knee ROM and velocity are lower-body measurements — a reference
length taken from the same kinematic chain is the most direct and
biomechanically meaningful scale factor. Torso length and shoulder width are
still computed and stored (`AnthropometricReference.torso_length`,
`.shoulder_width`) for transparency and potential future use, per the
task's candidate reference list.

Leg length is computed as **segment sum** (hip→knee + knee→ankle) rather
than a straight hip→ankle line, because the straight-line distance
under-estimates true limb length whenever the knee is bent (which happens
throughout a squat/clean movement) — the segment sum stays a much more
stable, angle-independent proxy for the athlete's actual leg length.

## 3. Normalization Formulas

### 3.1 Linear Range of Motion (ROM)

For a joint's vertical pixel position across the clip, $y_j(t)$ (midpoint of
left/right hip or knee y-coordinates):

$$\text{ROM}_{raw} = \max_t\big(y_j(t)\big) - \min_t\big(y_j(t)\big) \quad \text{(pixels)}$$

$$\text{ROM}_{norm} = \frac{\text{ROM}_{raw}}{L_{ref}} \quad \text{(unitless, expressed in "leg-lengths")}$$

Applied to both the hip ($j = \text{hip}$) and knee ($j = \text{knee}$)
joint trajectories → `hip_linear_rom_px` / `hip_linear_rom_normalized`,
`knee_linear_rom_px` / `knee_linear_rom_normalized`.

### 3.2 Linear Velocity

Using the same vertical position signal $y_j(t)$, sampled at the clip's
frame rate (`fps`, default from `DEFAULT_FPS`):

$$v_j(t) = \frac{d y_j}{dt} \approx \mathrm{gradient}\big(y_j(t),\, \Delta t = \tfrac{1}{fps}\big) \quad \text{(pixels/second)}$$

$$v_{peak,\,raw} = \max_t \big| v_j(t) \big|$$

$$v_{peak,\,norm} = \frac{v_{peak,\,raw}}{L_{ref}} \quad \text{(leg-lengths/second)}$$

Applied to hip and knee → `hip_peak_linear_velocity_px_s` /
`hip_peak_linear_velocity_normalized`, `knee_peak_linear_velocity_px_s` /
`knee_peak_linear_velocity_normalized`.

### 3.3 Degenerate-reference guard

$$\text{normalize}(v, L_{ref}) = \begin{cases} v & \text{if } L_{ref} < 10^{-6} \\ v / L_{ref} & \text{otherwise} \end{cases}$$

(`normalize_distance()`) — avoids division-by-zero/`inf` if a clip's
reference length could not be reliably estimated (e.g. too few usable
frames), instead of raising an exception.

## 4. Why the Existing Angular Metrics Did NOT Need Normalization

The joint angle at vertex $b$ between points $a, b, c$ is:

$$\theta = \cos^{-1}\!\left(\frac{\vec{BA} \cdot \vec{BC}}{\lVert \vec{BA} \rVert \, \lVert \vec{BC} \rVert}\right)$$

If every keypoint is uniformly scaled by a factor $s$ (e.g. the athlete is
filmed twice as close to the camera) and translated by an offset $\delta$
(e.g. a different position in frame), then $\vec{BA} \to s\,\vec{BA}$ and
$\vec{BC} \to s\,\vec{BC}$ (translation cancels in the subtraction that forms
each vector), and:

$$\theta' = \cos^{-1}\!\left(\frac{s\vec{BA} \cdot s\vec{BC}}{\lVert s\vec{BA}\rVert \lVert s\vec{BC}\rVert}\right) = \cos^{-1}\!\left(\frac{s^2(\vec{BA}\cdot\vec{BC})}{s^2 \lVert \vec{BA}\rVert \lVert \vec{BC}\rVert}\right) = \theta$$

i.e. **angles are invariant to uniform scale and translation** by
construction. Consequently, angular Hip/Knee ROM, angular peak, angular
velocity/acceleration and every Rule A-D metric derived from them (delay,
hip-dominance ratio, synchronization correlation, RFD ratio — all ratios or
timing values, not raw distances) were already scale-fair and were left
completely unmodified.

## 5. Why This Improves Fairness

- **Cross-athlete fairness:** a taller athlete with proportionally longer
  legs will naturally produce a larger raw-pixel hip/knee travel distance
  for an equivalent quality lift than a shorter athlete — without
  normalization this could be misread as "more range of motion" when it is
  really just a difference in body size.
- **Cross-camera fairness:** the same lift filmed closer to the camera (or
  with a longer lens) produces larger raw pixel displacements/velocities for
  identical physical motion. Dividing by a reference length measured from
  the *same frame* cancels this out, since both the joint displacement and
  the reference length scale by the same camera-distance/zoom factor.
- **Consistent units across the whole clip:** using the clip's own median
  leg length (rather than a fixed constant) means each analysis is
  self-calibrating — no manual camera-calibration step is required.

## 6. Advantages

- Makes linear ROM/velocity comparable across different athletes, camera
  setups and recording distances, which raw pixel values are not.
- Self-calibrating — derived entirely from the same pose keypoints already
  being extracted; no extra calibration video, marker, or camera
  intrinsics needed.
- Median aggregation over the whole clip is robust to individual noisy
  frames (occlusion, momentary bad pose detections).
- Backward-compatible: `LiftMetrics`'s new fields default to `0.0` /
  an all-zero `AnthropometricReference`, so older code paths that construct
  `LiftMetrics` from historic 5-column data (e.g.
  `hip_knee_build_dataset.py`'s adaptive-threshold learning from the master
  CSV) continue to work unmodified.
- No retraining required: the LSTM classifier's input features
  (`normalize_by_body_proportion`) and Rule A-D thresholds are untouched: these
  new normalized fields are purely additive, descriptive report metrics.

## 7. Limitations

- **2D projection only:** all reference lengths and displacements are
  measured in the 2D image plane. Perspective foreshortening (e.g. the
  athlete rotating relative to the camera, or the leg length appearing
  shorter at different depths within the frame) is not corrected — a true
  3D anthropometric normalization would require depth information or a
  calibrated multi-camera setup.
- **Leg length itself varies within a clip:** because it is derived from
  2D keypoints of a bending leg, apparent leg length changes slightly
  frame-to-frame even for a rigid limb, due to viewing-angle changes. The
  median reduces but does not eliminate this noise.
- **Assumes both legs are visible enough** to compute a bilateral leg
  length; if one side is heavily occluded, `check_frame_quality` may reject
  the frame first (upstream in `hip_knee_pose_utils.py`) — but a majority-
  occluded clip could still yield an unstable reference length.
- **Only vertical displacement/velocity are normalized.** Horizontal
  translation of the joints (e.g. lateral sway) is not currently measured
  or normalized — could be added later using the same `L_ref` if needed.
- **Angular metrics (Rule A-D, angular ROM) are unaffected by this change**
  — they were already scale-invariant, so this work does not change any
  existing classifier input, Rule A-D score, or the model's predictions.

## 8. Files Modified

- `src/features/hip_knee_biomechanics.py`:
  - Added `AnthropometricReference` frozen dataclass (`leg_length`,
    `torso_length`, `shoulder_width`, `primary_reference`).
  - Added `compute_reference_lengths()`, `normalize_distance()`,
    `compute_linear_rom()`, `compute_linear_velocity()`.
  - Extended `LiftMetrics` with 9 new **defaulted** fields (`reference`,
    `hip_linear_rom_px`, `hip_linear_rom_normalized`, `knee_linear_rom_px`,
    `knee_linear_rom_normalized`, `hip_peak_linear_velocity_px_s`,
    `hip_peak_linear_velocity_normalized`, `knee_peak_linear_velocity_px_s`,
    `knee_peak_linear_velocity_normalized`) — defaults preserve backward
    compatibility with existing 5-argument `LiftMetrics(...)` construction
    in `src/data/hip_knee_build_dataset.py` (unmodified).
  - Wired the new computations into `analyze_lift()`.
- `src/models/hip_knee_predict.py`:
  - `PredictionReport` gained `leg_length_reference_px`,
    `hip_linear_rom_normalized`, `knee_linear_rom_normalized`,
    `hip_peak_velocity_normalized`, `knee_peak_velocity_normalized`.
  - `build_prediction_report()` maps these from the new `LiftMetrics` fields.
  - `print_prediction_report()` prints the new values in the console report.
  - `reports/prediction.json` now includes the new fields automatically
    (via `PredictionReport.to_dict()`).
- `reports/anthropometric_normalization.md` — this report (new file).

**Not modified:** `src/models/hip_knee_train.py`, `src/models/hip_knee_lstm.py`,
`src/models/hip_knee_evaluate.py`, `data/processed/hip_knee/combined_X.npy`
/`combined_y.npy`, any saved model weights, `module3_arm_analysis`,
`visualization`, or the frontend. No retraining was performed — re-running
`python -m src.models.hip_knee_predict --video <path>` on a sample video
confirmed the pipeline still loads the existing model/weights unchanged and
now additionally reports the normalized linear ROM/velocity values (e.g.
`leg_length_reference_px=602.4`, `hip_linear_rom_normalized=0.0542`,
`knee_linear_rom_normalized=0.0425`).
