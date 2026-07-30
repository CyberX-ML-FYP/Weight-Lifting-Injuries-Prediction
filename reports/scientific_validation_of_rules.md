# Scientific Validation of the Hip & Knee Biomechanical Metrics

**Scope of this document.** This is a *retrospective scientific review* of
the Hip & Knee module as it exists today. **No source code was modified. No
model was retrained. The module is treated as finalized** — this report
only reads and evaluates `src/features/hip_knee_biomechanics.py`,
`src/features/hip_knee_scoring.py`, `src/models/hip_knee_predict.py`,
`src/features/hip_knee_pose_utils.py`, and `src/features/hip_knee_config.py`.

**Citation policy.** Per the request, **no citations are invented**. Where
a metric corresponds to a well-established, textbook-level biomechanical
concept (e.g. joint range of motion, angular velocity, Pearson correlation
as a synchrony measure), that general concept is named as such. Where a
*specific* empirical claim (a numeric threshold, a named published study,
or a specific coefficient) would normally require a citation and none is
present in this codebase or its existing reports, this document explicitly
states: **"Further literature review is required to confirm this."** No
author names, journal titles, or publication years are fabricated anywhere
below.

---

## 1. Three-tier framework: what each metric actually is

Every metric in this pipeline must be placed into exactly one of three
categories, because conflating them is the single biggest scientific risk
in a system like this:

| Tier | Definition | Where it lives in this project |
|---|---|---|
| **1. International Weightlifting Federation (IWF) judging rules** | Formal, rulebook-defined criteria used by human referees to decide whether a competition lift is *technically valid* (e.g. full elbow lockout, no re-bend of the knees before the down signal, bar path within the frame of reference). | **Not implemented anywhere in this codebase.** A repo-wide search for `IWF`, `International Weightlifting`, and `judging` returns zero matches. This module does **not** encode or claim to encode IWF referee criteria. |
| **2. Scientific sports-biomechanics methods** | General, textbook-level concepts from kinesiology/biomechanics (joint angle via three landmarks, angular velocity/acceleration as time-derivatives of angle, proximal-to-distal ("kinetic chain") segment sequencing, cross-correlation as a synchrony measure, rate of force development as the time-derivative of force). These concepts are widely taught and used, but this project does **not** cite a specific paper validating each numeric choice made here. | Used as the *conceptual basis* for Hip ROM, Knee ROM, Hip Peak, Knee Peak, Sequential Delay, Synchronization, Hip Dominance, Correlation, and RFD. |
| **3. Our implementation** | The concrete engineering choices made in this repository: which joints are tracked (YOLO11-pose, COCO-17 subset), which finite-difference formulas are used, which smoothing filter is applied, which weights (0.35/0.30/0.20/0.15) blend the four "Rule A–D" scores, and how thresholds are *learned from this project's own 172-sample dataset* rather than from a published normative range. | Everything in `hip_knee_biomechanics.py`, `hip_knee_scoring.py`, `hip_knee_predict.py`. |

**Key finding, stated up front:** the "Good"/"Poor" ground-truth labels
this whole pipeline is trained against come from **folder names assigned
during data collection** (`src/data/hip_knee_build_dataset.py`,
`CLASS_NAMES = ("Good", "Poor")`), i.e. a **human/coach judgment call**
about lift quality — not a formal, documented IWF technical-validity
ruling, and not an independent biomechanics-literature ground truth. This
means every metric below is ultimately validated against *this project's
own subjective labels*, and its predictive value should be read in that
light.

---

## 2. Per-metric scientific validation

For each metric: **Definition**, **Mathematical implementation** (exact
formula and code location), **Scientific rationale**, **Advantages**,
**Limitations**, **Future improvements**. A **Tier classification**
(1/2/3, per §1) and **literature-support status** are given first.

---

### 2.1 Hip ROM (Range of Motion)

- **Tier**: 2 (concept) + 3 (implementation). Not an IWF rule.
- **Literature support**: The concept of joint range of motion (max angle
  − min angle across a movement cycle) is a standard, textbook
  biomechanics/goniometry measure. **No specific published study is cited
  in this codebase validating the exact hip-ROM thresholds learned here —
  further literature review is required to confirm normative ROM ranges
  for the hip hinge/squat pattern in weightlifting specifically.**

**Definition.** The total angular excursion of the hip joint (shoulder–
hip–knee angle) over the sampled portion of the lift.

**Mathematical implementation** (`compute_rom_and_peaks`,
[hip_knee_predict.py](../src/models/hip_knee_predict.py#L152)):
$$\text{HipROM} = \max_t(\theta_{hip}(t)) - \min_t(\theta_{hip}(t))$$
where $\theta_{hip}(t) = \tfrac{1}{2}\big(\theta_{hip}^{L}(t) + \theta_{hip}^{R}(t)\big)$
is the bilateral average of
$\theta_{hip} = \cos^{-1}\!\left(\dfrac{\vec{BA}\cdot\vec{BC}}{|\vec{BA}||\vec{BC}|}\right)$
with $A=\text{shoulder}$, $B=\text{hip}$, $C=\text{knee}$
(`calculate_angle`, [hip_knee_biomechanics.py](../src/features/hip_knee_biomechanics.py#L84)).
The angle sequence is Savitzky–Golay smoothed
(`smooth_angle_sequence`, window=7, polyorder=2) before the min/max is taken.

**Scientific rationale.** Joint angle computed from three 2-D landmarks via
the dot-product/arccosine formula is the standard planar-angle
computation used throughout markerless-pose biomechanics research (a
well-established *method*, not a project-specific invention). Using ROM
(rather than raw angle at one instant) captures how much the hip actually
moves during the lift, which is a reasonable proxy for hip-hinge depth/
engagement.

**Advantages**: scale-invariant (an angle in degrees does not depend on
camera distance or athlete height, unlike a pixel distance); bilateral
averaging reduces single-side occlusion noise; smoothing reduces
pose-jitter noise before the min/max (which is otherwise very sensitive to
single-frame outliers).

**Limitations**:
1. Computed from a **single 2-D camera view** — true 3-D hip flexion is
   foreshortened/distorted by perspective and camera angle; no
   multi-view triangulation or depth sensing is used.
2. `NUM_SAMPLED_FRAMES = 20` frames per clip is a sparse temporal sample
   of the full lift; ROM could be underestimated if the true min/max
   occurs between sampled frames.
3. min/max is sensitive to a single mis-tracked frame surviving the
   quality filter (`MIN_KEYPOINT_CONFIDENCE = 0.5`) even after smoothing.
4. No published normative "good" hip-ROM range for this specific lift is
   cited; the "good"/"poor" ranges used for scoring are **learned from
   this project's own 172-sample dataset**
   (`learn_adaptive_thresholds`), not from literature.

**Future improvements**: validate against a marker-based (e.g. optical
motion capture) or multi-camera 3-D ground truth on a held-out subset;
increase temporal sampling density or use the full frame sequence rather
than 20 sampled frames; conduct a literature review specifically for
hip-hinge/squat ROM normative ranges before treating the learned
thresholds as clinically meaningful.

---

### 2.2 Knee ROM

- **Tier**: 2 (concept) + 3 (implementation). Not an IWF rule.
- **Literature support**: Same as Hip ROM — joint ROM is a standard
  biomechanics concept; **no specific citation for knee-ROM normative
  thresholds in weightlifting is present in this codebase — further
  literature review required.**

**Definition.** Total angular excursion of the knee joint (hip–knee–ankle
angle) over the sampled clip.

**Mathematical implementation**: identical formula/pipeline to Hip ROM,
substituting $A=\text{hip}, B=\text{knee}, C=\text{ankle}$:
$$\text{KneeROM} = \max_t(\theta_{knee}(t)) - \min_t(\theta_{knee}(t))$$
(`compute_rom_and_peaks`, [hip_knee_predict.py](../src/models/hip_knee_predict.py#L152)).

**Scientific rationale**: same as Hip ROM (§2.1) — standard 3-point
planar-angle ROM computation.

**Advantages / Limitations / Future improvements**: identical to Hip ROM
(§2.1) — same 2-D single-view caveat, same sparse-sampling caveat, same
dataset-derived (not literature-derived) thresholds, same recommendation
to validate against ground-truth motion capture and conduct a targeted
literature review.

---

### 2.3 Hip Peak

- **Tier**: 2 (concept) + 3 (implementation). Not an IWF rule.
- **Literature support**: "peak joint angle" is a standard kinematic
  descriptor; **no specific literature-derived peak-angle threshold is
  cited for this lift — further review required.**

**Definition.** The maximum hip flexion/extension angle reached during the
sampled clip (a single scalar, not an excursion).

**Mathematical implementation** (`compute_rom_and_peaks`,
[hip_knee_predict.py](../src/models/hip_knee_predict.py#L161)):
$$\text{HipPeak} = \max_t(\theta_{hip}(t))$$
using the same smoothed, bilaterally-averaged angle sequence as §2.1.

**Scientific rationale.** Peak joint angle is a standard descriptor used
alongside ROM in kinematic analyses; here it specifically captures the
most extended hip position reached (relevant to detecting incomplete lockout
at the top of the movement).

**Advantages**: simple, scale-invariant, directly interpretable in
degrees.

**Limitations**:
1. A pure `max()` over 20 sampled frames — **not** robust to a single
   outlier frame with a spuriously large angle (smoothing mitigates but
   does not eliminate this).
2. Says nothing about *when* in the lift the peak occurred, or whether it
   was sustained (a momentary peak vs. a held lockout are indistinguishable
   by this metric alone).
3. Same 2-D/single-view/sparse-sampling caveats as §2.1.

**Future improvements**: report peak *and* the duration the angle stays
within some tolerance of the peak (to distinguish a held lockout from a
transient spike); cross-validate against manual frame-by-frame
goniometric measurement on a sample of clips.

---

### 2.4 Knee Peak

- **Tier**: 2 (concept) + 3 (implementation). Not an IWF rule.
- **Literature support**: same as Hip Peak — standard concept, no specific
  cited threshold; **further review required.**

**Definition.** Maximum knee extension angle reached during the sampled
clip.

**Mathematical implementation** (`compute_rom_and_peaks`,
[hip_knee_predict.py](../src/models/hip_knee_predict.py#L162)):
$$\text{KneePeak} = \max_t(\theta_{knee}(t))$$

**Scientific rationale / Advantages / Limitations / Future improvements**:
identical to Hip Peak (§2.3), substituting the knee angle sequence.

---

### 2.5 Sequential Delay (Rule A — sequential extension)

- **Tier**: 2 (concept: proximal-to-distal / hip-knee-ankle triple
  extension timing) + 3 (this project's specific implementation). Not an
  IWF rule.
- **Literature support**: The general concept that lower-body joints extend
  in a coordinated *temporal sequence* during explosive triple-extension
  movements is a recognized topic in weightlifting/jumping biomechanics
  discussions. **However, this codebase does not cite a specific published
  source establishing an optimal hip-after-knee delay value, and none is
  invented here — further literature review is explicitly required before
  treating any specific delay value (including the delay ranges learned
  from this project's own data) as a validated biomechanical norm.**

**Definition.** The timing offset between the frame at which knee angular
velocity peaks and the frame at which hip angular velocity peaks.

**Mathematical implementation** (`rule_a_sequential_extension`,
[hip_knee_biomechanics.py](../src/features/hip_knee_biomechanics.py#L305)):
$$\text{delay} = \frac{t_{hip\_peak} - t_{knee\_peak}}{\text{fps}}$$
where $t_{knee\_peak} = \arg\max_t(\dot\theta_{knee}(t))$,
$t_{hip\_peak} = \arg\max_t(\dot\theta_{hip}(t))$, and
$\dot\theta = \text{np.gradient}(\theta, dt)$ is the first time-derivative
of the (Savitzky–Golay-smoothed) angle sequence
(`compute_derivatives`, [hip_knee_biomechanics.py](../src/features/hip_knee_biomechanics.py#L296)).
A positive value means the hip's velocity peak occurs *after* the knee's.

**Scientific rationale.** Uses `np.argmax` on signed velocity (not
absolute value), so it specifically measures "hip extends fastest *after*
knee extends fastest," matching the intended "hip-after-knee" narrative
described in the code comments. This is a reasonable, if simplified,
operationalization of proximal-to-distal segment sequencing.

**Advantages**: simple, deterministic, directly interpretable in seconds;
positive/negative sign carries physical meaning (which joint leads).

**Limitations**:
1. Uses **signed** velocity `argmax`, not `argmax(abs(velocity))` — if a
   joint's largest-magnitude velocity is actually negative (e.g. during a
   descent phase captured in the sampled window), the "peak" this rule
   finds may not correspond to the concentric extension phase the metric
   is intended to describe. This is a genuine implementation subtlety,
   not a flaw in the underlying biomechanical concept.
2. Extremely sensitive to noise: it depends on the location of a **single
   maximum** of a **twice-differentiated, jitter-amplified** signal
   (angle → velocity via `np.gradient`), even after smoothing.
3. Only 20 sampled frames at (assumed) `DEFAULT_FPS = 30` gives coarse
   temporal resolution (~33 ms per frame *if* frames were evenly spaced
   at 30 fps, which depends on how sampling was performed upstream) —
   too coarse to resolve delays reported in the tens-of-milliseconds
   range that finer-grained biomechanics literature might discuss.
4. No published normative "good" delay value is cited; the good/poor
   ranges are learned purely from this project's 172-sample dataset.

**Future improvements**: switch to `argmax(abs(velocity))` if a
"largest-magnitude timing difference" (rather than "largest positive
velocity timing difference") is the intended definition, or explicitly
document why signed velocity was chosen; validate the fps assumption used
for frame-to-time conversion against the actual sampling method; seek a
specific literature source for expected proximal-to-distal timing
sequences in the clean/snatch/squat before treating any delay threshold as
clinically meaningful.

---

### 2.6 Synchronization (Rule C)

- **Tier**: 2 (concept: inter-joint coordination/synchrony) + 3
  (implementation as a single Pearson coefficient over one clip). Not an
  IWF rule.
- **Literature support**: Cross-correlation / Pearson correlation between
  two joint kinematic time-series is a standard, general statistical tool
  used in movement-coordination research. **No specific published study is
  cited that validates Pearson's r on angular velocity specifically as the
  correct "synchronization" measure for this lift, nor a specific
  correlation threshold — further literature review is required.**

**Note on naming**: this project reports **two** related but distinct
quantities under two different names — "Synchronization" (the normalized
0–100 `rule_c_score`) and "Correlation" (the raw Pearson coefficient,
`report.correlation`). Both derive from the same underlying computation
(§2.6/§2.8), and are documented together here and separately in §2.8 for
completeness, matching how they are exposed as two distinct named fields
in `PredictionReport`.

**Mathematical implementation** (`rule_c_synchronization`,
[hip_knee_biomechanics.py](../src/features/hip_knee_biomechanics.py#L318)):
$$r = \text{Pearson}\big(\dot\theta_{knee}(t),\ \dot\theta_{hip}(t)\big)$$
computed via `scipy.stats.pearsonr` over the full smoothed angular-velocity
sequences, guarded to return `0.0` if either signal has near-zero
variance (`np.std < 1e-9`) or fewer than 2 samples — a defensive
implementation detail not present in the original notebook this was
adapted from. The **score** shown to users
(`rule_c_score`/"Synchronization") is this same $r$ passed through
`normalize_score(r, min, max)` — a min-max rescale into `[0, 100]` using
ranges learned from this project's own Good/Poor training clips
(`learn_adaptive_thresholds`), **not** an independently-validated
normalization.

**Scientific rationale.** Pearson correlation between two velocity signals
is a legitimate, widely-used way to quantify whether two joints move "in
time" with each other (high |r| = tightly coupled temporal profiles,
regardless of amplitude). Using velocity (rather than raw angle)
emphasizes *timing* synchrony over static pose similarity, which is
conceptually appropriate for a coordination measure.

**Advantages**: bounded, well-understood statistic (`[-1, 1]`); computed
over the whole clip rather than a single frame, so it is less sensitive to
any one frame's noise than Sequential Delay; the zero-variance guard
prevents a crash on degenerate/static clips.

**Limitations**:
1. Pearson's r measures **linear** association only; genuinely
   coordinated but nonlinearly-related or phase-shifted (lagged) velocity
   profiles could score poorly even if biomechanically "synchronized" —
   the metric has no explicit lag/phase-shift search (unlike, e.g.,
   cross-correlation-with-lag).
2. A high correlation can arise trivially if both signals are dominated
   by the same single acceleration/deceleration event of the whole body
   (e.g. camera or torso sway), rather than genuine joint-to-joint
   coordination — no control for this confound is implemented.
3. The `min/max`-based `normalize_score` rescaling is entirely
   dataset-relative: a correlation value that would be considered
   "average" under some external biomechanical standard could still map
   to a high or low score here purely because it falls near this
   project's own observed min/max.
4. No literature-backed correlation threshold exists in this codebase for
   "well-synchronized" vs. "poorly-synchronized" hip-knee extension.

**Future improvements**: consider a lagged cross-correlation (to allow for
a small, expected phase offset rather than penalizing it as
"desynchronized"); investigate whether whole-body motion (e.g. via a
stable reference such as the pelvis center's own acceleration) should be
regressed out before correlating hip/knee signals; seek literature on
inter-joint coordination measures used in weightlifting-specific
biomechanics before treating the current threshold as validated.

---

### 2.7 Hip Dominance (Rule B)

- **Tier**: 2 (concept: relative segment contribution/proximal
  "hip-dominant" strategy) + 3 (this project's ratio implementation). Not
  an IWF rule.
- **Literature support**: The general idea of a "hip-dominant" vs.
  "knee-dominant" lifting strategy is a commonly discussed qualitative
  concept in strength & conditioning coaching material. **No specific
  peer-reviewed source is cited in this codebase quantifying an "ideal"
  hip:knee peak-angular-velocity ratio — further literature review is
  required before treating any numeric ratio threshold as validated.**

**Definition.** The ratio of the peak hip angular velocity magnitude to
the peak knee angular velocity magnitude during the clip.

**Mathematical implementation** (`rule_b_hip_dominance`,
[hip_knee_biomechanics.py](../src/features/hip_knee_biomechanics.py#L312)):
$$\text{HipDominance} = \frac{\max_t|\dot\theta_{hip}(t)|}{\max_t|\dot\theta_{knee}(t)| + \epsilon}, \quad \epsilon=10^{-6}$$
($\epsilon$ is a pure divide-by-zero guard, not a biomechanical
parameter.)

**Scientific rationale.** A ratio of peak angular velocities is a simple,
interpretable way to express which joint moves faster (proportionally) at
its most explosive moment — a reasonable, if coarse, proxy for "which
joint dominates the movement."

**Advantages**: dimensionless (a ratio of two angular velocities in the
same units cancels units), simple to compute and interpret, robust to
absolute camera scale (since both numerator and denominator are angular,
not linear/pixel, velocities).

**Limitations**:
1. Compares only the **single peak instant** of each joint — two lifts
   with very different overall velocity *profiles* could have identical
   peak ratios.
2. Does not account for the *timing* of each peak (that is a separate
   concern already covered, imperfectly, by Sequential Delay §2.5) — a
   hip peak and knee peak occurring at very different lift phases still
   contribute equally to this ratio.
3. No literature-derived "ideal" ratio value is cited; scoring thresholds
   are learned purely from this project's own 172-sample dataset.
4. The name "dominance" implies a causal/mechanical relationship (e.g.
   force/power contribution) that a velocity ratio alone does not
   establish — true joint power dominance would additionally require
   joint torque, which is not computed anywhere in this pipeline (no
   inverse dynamics, no force plate, no EMG).

**Future improvements**: incorporate joint power (torque × angular
velocity) rather than velocity alone, if inverse-dynamics or force data
ever become available; report the full velocity-ratio time-series rather
than a single peak-to-peak ratio; seek literature specifically addressing
quantitative hip- vs. knee-dominant lifting-strategy metrics.

---

### 2.8 Correlation (raw Rule C coefficient)

- **Tier**: 2 (concept, same as §2.6) + 3 (implementation). Not an IWF
  rule.
- **Literature support**: same as §2.6 — Pearson correlation is a standard
  statistical method; **no specific citation validates its use as a
  weightlifting synchrony measure — further review required.**

**Definition.** The raw Pearson correlation coefficient between knee and
hip angular velocity signals over the clip (this is the **unnormalized**
value; see §2.6 for the normalized 0–100 "Synchronization" score derived
from the same number).

**Mathematical implementation**: identical formula to §2.6,
`rule_c_synchronization`
([hip_knee_biomechanics.py](../src/features/hip_knee_biomechanics.py#L318)),
exposed directly (not min-max rescaled) as `PredictionReport.correlation`
([hip_knee_predict.py](../src/models/hip_knee_predict.py#L229)).

**Scientific rationale / Advantages / Limitations / Future
improvements**: identical to §2.6. This entry exists separately in this
report only because the pipeline exposes the *raw* coefficient and the
*rescaled* score as two distinct named outputs (`correlation` vs.
`rule_c_score`) — scientifically they are the same underlying
computation and share the same validation status.

---

### 2.9 Rate of Force Development (RFD, Rule D)

- **Tier**: 2 (concept, borrowed terminology) + 3 (this project's proxy
  implementation — **important divergence flagged below**). Not an IWF
  rule.
- **Literature support**: In the sports-science/strength-training
  literature, **Rate of Force Development (RFD) is formally defined as
  the time-derivative of measured FORCE** (typically from a force plate:
  $RFD = dF/dt$, in newtons per second), used to characterize explosive
  strength. **This is a well-established concept, but this codebase does
  not measure force at all** — no force plate, load cell, or barbell load
  data is used anywhere in this pipeline. **Further literature review is
  required to determine whether an angular-acceleration ratio derived
  from video is an acceptable proxy for true force-based RFD; as
  implemented, this is a named approximation, not a validated
  substitute.**

**Definition (as implemented, NOT the textbook force-based definition)**:
the ratio of peak hip angular acceleration to peak knee angular
acceleration.

**Mathematical implementation** (`rule_d_rfd`,
[hip_knee_biomechanics.py](../src/features/hip_knee_biomechanics.py#L325)):
$$\text{RFD}_{ratio} = \frac{\max_t|\ddot\theta_{hip}(t)|}{\max_t|\ddot\theta_{knee}(t)| + \epsilon}$$
where $\ddot\theta = \text{np.gradient}(\dot\theta, dt)$ is the **second**
time-derivative of the smoothed angle signal (`compute_derivatives`,
[hip_knee_biomechanics.py](../src/features/hip_knee_biomechanics.py#L296)).

**Scientific rationale.** The name "Rate of Force Development" is
borrowed from strength-training terminology where RFD = dF/dt of measured
force. This implementation substitutes **angular acceleration** (a
*kinematic* quantity, degrees/s²) for **force** (a *kinetic* quantity,
newtons), because no force-measurement hardware is available in this
video-only pipeline. Angular acceleration is loosely related to net joint
torque via inverse dynamics (which itself depends on segment mass, moment
of inertia, and more), but **no such inverse-dynamics calculation is
performed here** — the metric is a kinematic *proxy*, several
simplifying steps removed from actual force.

**Advantages**: computable from pose-only data (no instrumentation
needed); directionally plausible (a joint accelerating explosively likely
does correlate with higher force production, all else equal); consistent
units (ratio of two angular accelerations is dimensionless).

**Limitations** (this metric carries the largest scientific gap of the
eleven, and this should be stated plainly):
1. **It is not RFD in the textbook sense.** No force is measured or
   estimated. The name should be read as "an angular-acceleration-ratio
   proxy inspired by the RFD concept," not as a validated
   biomechanical force measurement.
2. Second derivatives (`np.gradient` applied twice) are **very
   sensitive to pose-tracking noise** — even with Savitzky–Golay smoothing
   of the angle signal, acceleration estimates from 20 sparsely sampled
   video frames are inherently noisy compared to force-plate-derived RFD
   (typically sampled at hundreds–thousands of Hz).
3. Converting angular acceleration to true force/torque would require
   segment mass and moment of inertia (anthropometric parameters not
   estimated anywhere in this codebase) — this is a fundamentally
   different, more complex calculation than what is implemented.
4. No literature-derived "good" RFD-ratio threshold is cited; ranges are
   learned solely from this project's 172-sample dataset.

**Future improvements**: rename the metric in user-facing output (e.g.
"Angular Acceleration Ratio") to avoid implying a validated force-based
RFD measurement, unless/until true kinetic data is available; if
force/torque estimation is ever pursued, this would require a full
inverse-dynamics model with anthropometric segment parameters, which is a
substantial addition beyond the current pose-only pipeline; explicitly
review the force-plate RFD literature to determine whether any
video-derived kinematic proxy can be defensibly validated against it.

---

### 2.10 Confidence Weighting

- **Tier**: 3 only (pure engineering reliability heuristic — not a
  biomechanical concept, and not an IWF rule). No literature claim is made
  or needed here since this is a data-quality/measurement-reliability
  mechanism, not a claim about the athlete's body.

**Definition.** A `[0, 1]` value representing how reliable the underlying
pose-detection data was for a given rule or clip, used to discount
unreliable rule scores rather than trusting every detection equally.

**Mathematical implementation** (multiple functions,
[hip_knee_biomechanics.py](../src/features/hip_knee_biomechanics.py#L263)):
- Per-frame confidence = mean YOLO11-pose keypoint confidence over the 8
  required keypoints (`BodyKeypoints.mean_confidence`, set in
  `extract_body_keypoints`, [hip_knee_pose_utils.py](../src/features/hip_knee_pose_utils.py#L128)).
- Per-rule confidence is grounded in the *specific frames each rule
  depends on* (`compute_rule_confidences`): Rule A/B/D use the mean
  confidence at the 1-2 frames where the relevant peak occurs; Rule C uses
  the mean confidence across **all** frames (since it depends on the whole
  velocity signal).
- Overall confidence blends the four per-rule confidences using the
  **same** `RULE_WEIGHTS = (0.35, 0.30, 0.20, 0.15)` used for the Rule A–D
  score blend (`compute_overall_confidence`).
- Final score weighting: either a single scalar multiply
  (`compute_performance_score`: `raw_score *= conf`) or a
  confidence-weighted re-blend of scores
  (`compute_confidence_weighted_score`,
  [hip_knee_scoring.py](../src/features/hip_knee_scoring.py#L177)):
$$\text{FinalScore} = \frac{\sum_i w_i \cdot conf_i \cdot score_i}{\sum_i w_i \cdot conf_i}$$
  falling back to an unweighted blend if all confidences are ≈0 (guards
  divide-by-zero).
- A prediction is flagged `is_low_confidence` if **either** the LSTM's own
  softmax confidence **or** the overall Rule A-D confidence falls below
  `LOW_CONFIDENCE_THRESHOLD = 0.6` (`is_low_confidence_prediction`,
  [hip_knee_predict.py](../src/models/hip_knee_predict.py#L177)).

**Scientific rationale.** This is a **measurement-reliability /
uncertainty-quantification heuristic**, not a biomechanical model of the
athlete. Discounting a metric that was derived from low-confidence pose
detections (partial occlusion, motion blur, ambiguous multi-person frames)
before trusting it is a defensible data-quality engineering practice,
analogous in spirit (though not in method) to how sensor-fusion systems
down-weight low-confidence/noisy sensor readings.

**Advantages**: makes the reliability of each prediction visible rather
than silently reporting every prediction as equally trustworthy; grounding
each rule's confidence in the specific frames it actually depends on
(rather than one blanket clip-level number for every rule) is a more
precise design than a single global confidence value; the `0.6` threshold
and `RULE_WEIGHTS` reuse are transparent, auditable constants (not hidden
in a black box).

**Limitations**:
1. `LOW_CONFIDENCE_THRESHOLD = 0.6` and `RULE_WEIGHTS = (0.35, 0.30, 0.20,
   0.15)` are **project-chosen constants**, not derived from a
   sensitivity study or external validation — **further work is required**
   to confirm `0.6` is the "right" cutoff rather than an arbitrary
   round number.
2. YOLO11-pose's own per-keypoint confidence score is itself a
   *model-internal* detection-confidence estimate, not an independently
   calibrated measurement-uncertainty value — it may not linearly
   correspond to true anatomical landmark accuracy.
3. Confidence weighting can mask a systematically bad rule computation
   (e.g. the signed-velocity `argmax` issue noted in §2.5) — a
   high-confidence pose detection does not guarantee the *rule formula*
   downstream is capturing what it claims to.

**Future improvements**: validate `LOW_CONFIDENCE_THRESHOLD` and
`RULE_WEIGHTS` against labelled examples of known-good vs. known-bad pose
tracking (a calibration study), rather than treating them as fixed;
consider reporting a calibrated uncertainty interval per metric instead of
a single blended confidence scalar.

---

### 2.11 Anthropometric Normalization

- **Tier**: 2 (concept: scaling distance measures by a body-proportional
  reference length is standard practice in markerless video biomechanics
  to remove camera-distance/subject-height confounds) + 3
  (this project's specific reference-length choice). Not an IWF rule.
- **Literature support**: normalizing linear/pixel measurements by a
  body-segment reference length (rather than using raw pixels) is a
  widely used general technique in markerless motion analysis to make
  measurements comparable across subjects and camera setups. **No specific
  published study is cited validating "thigh+shank segment length" as the
  optimal reference length choice over alternatives (e.g. torso length,
  shoulder width, or estimated total height) — further literature review
  is required.**

**Definition.** Pixel-space distance/velocity measurements (which are
inherently camera-distance- and athlete-height-dependent) are divided by a
clip-level anthropometric reference length, making them comparable across
different athletes and camera setups.

**Mathematical implementation**
(`compute_reference_lengths`, [hip_knee_biomechanics.py](../src/features/hip_knee_biomechanics.py#L143)):
- `leg_length` = median-across-clip of (bilateral-averaged) thigh + shank
  segment length: $|\text{hip}\to\text{knee}| + |\text{knee}\to\text{ankle}|$,
  averaged left/right, then reduced by **median** (not mean, for
  robustness to single-frame jitter) across all sampled frames. This is
  used as `primary_reference`.
- `torso_length` and `shoulder_width` are also computed (median
  shoulder-midpoint↔hip-midpoint distance; median left↔right shoulder
  distance) but are **not** currently used as the primary normalization
  reference — only `leg_length` is.
- Normalization itself (`normalize_distance`,
  [hip_knee_biomechanics.py](../src/features/hip_knee_biomechanics.py#L228)):
$$\text{value}_{normalized} = \frac{\text{value}_{raw\_px}}{\text{leg\_length}_{px}}$$
  (returned unchanged, as a safe no-op, if `leg_length < 1e-6`).
- Applied to: `hip_linear_rom` / `knee_linear_rom` (vertical pixel travel
  of the hip/knee joint, `compute_linear_rom`) and `hip`/`knee_peak
  _linear_velocity` (peak vertical pixel velocity, `compute_linear_rom`
  and `compute_linear_velocity`).
- **Separately**, `normalize_by_body_proportion`
  ([hip_knee_pose_utils.py](../src/features/hip_knee_pose_utils.py#L147))
  normalizes the LSTM's raw *input* keypoints by translating to the
  hip-midpoint origin and scaling by **torso length** (not leg length) —
  a **different reference length is used for the model's raw input
  normalization than for the reported linear-ROM/velocity features**, a
  detail worth being aware of when interpreting the two together.

**Scientific rationale.** Hip/knee joint angles (§2.1–2.4) are already
scale-invariant by construction (an angle does not depend on distance from
the camera). However, **linear** pixel-space measurements (vertical joint
travel, pixel velocity) are NOT scale-invariant — they grow with camera
zoom/proximity and with athlete height. Dividing by a body-proportional
reference length is a standard way to make such measurements comparable
across different athletes/camera setups, analogous in spirit to
normalizing measurements by body height/segment length in general
markerless-video biomechanics practice.

**Advantages**: uses **median** (not mean) across the clip, which is
robust to a single occluded/jittery frame; falls back gracefully (returns
the raw value rather than `inf`/`nan`) if the reference length is
degenerate; leg length (thigh+shank sum) is a reasonable choice because it
tracks true limb length even when the knee is bent (unlike a straight
hip→ankle line, which foreshortens as the knee bends).

**Limitations**:
1. The reference length is itself measured from the **same single 2-D
   camera view** as the quantity being normalized — it is subject to the
   same perspective distortion (e.g. an athlete standing at an angle to
   the camera will have a foreshortened leg-length reference, which does
   not fully correct for that same foreshortening in the ROM/velocity
   values being normalized by it).
2. No independent ground-truth (e.g. actual measured leg length in cm) is
   used to validate that the pixel-based reference length is
   proportionally consistent across different athletes/videos.
3. Two *different* reference lengths are used in different parts of the
   pipeline (leg_length for reported ROM/velocity features vs. torso_length
   for the LSTM's raw input normalization) — this is a deliberate design
   choice documented in the code, but means "normalized" values from
   different parts of the pipeline are not directly on the same relative
   scale as each other.
4. No specific published study is cited validating leg-length-based
   normalization over the alternatives (torso length, shoulder width, or
   estimated stature) for this specific application.

**Future improvements**: validate the pixel-based reference length
against actual measured anthropometric data (if available) for a subset
of athletes; consider camera-angle correction (e.g. requiring a
consistent sagittal-plane camera position, which the "Side view" folder
convention already partially encourages) before relying on the reference
length for cross-athlete comparison; conduct a literature review
specifically comparing reference-length choices (leg vs. torso vs.
stature) for markerless video anthropometric normalization.

---

## 3. Cross-cutting scientific limitations (apply to all 11 metrics)

1. **Single 2-D camera view.** All angles, distances, and velocities are
   computed from one camera perspective (Side/Angle/Front view folders).
   True 3-D joint kinematics are not recovered; all values are subject to
   perspective foreshortening that varies with the athlete's exact
   orientation to the camera on a given rep.
2. **YOLO11-pose (COCO-17) keypoint detector**, not a marker-based or
   multi-camera motion-capture system. Its per-keypoint confidence score
   is a model-internal detection estimate, not an independently calibrated
   measurement-uncertainty value (§2.10, limitation 2).
3. **Sparse temporal sampling** (`NUM_SAMPLED_FRAMES = 20` per clip) limits
   temporal resolution for all rate/timing-based metrics (Sequential
   Delay, RFD, peak velocity/acceleration).
4. **Savitzky–Golay smoothing** (window=7, polyorder=2) is applied
   uniformly to reduce jitter before differentiating, but any smoothing
   filter trades noise reduction for some loss of genuine high-frequency
   signal content — its effect on the specific quantities reported here
   has not been separately validated against an unsmoothed or
   ground-truth reference.
5. **All adaptive thresholds (`AdaptiveThresholds`, "good"/"poor" ranges
   for Rule A-D) are learned empirically from this project's own 172-clip
   dataset** (112 Good / 60 Poor, per
   [model_improvement_report.md](model_improvement_report.md)) — **not**
   from an external, peer-reviewed normative dataset. Every "good" vs.
   "poor" numeric range in this system should be read as *relative to this
   project's own sample*, not as an externally validated clinical or
   competitive standard.
6. **Ground truth labels are folder-assigned Good/Poor judgments** made
   during data collection (§1), not a documented, reproducible application
   of IWF technical-validity rules, and not derived from an independent
   biomechanics gold-standard. The entire pipeline's notion of "quality"
   is only as valid as this original labeling process, which is outside
   the scope of the code reviewed here.
7. **No kinetic (force/torque/EMG) data anywhere in this pipeline.** Every
   metric is purely kinematic (angles and their derivatives). This is most
   consequential for the "Rate of Force Development" metric (§2.9), whose
   name implies a kinetic measurement that is not actually present.

---

## 4. Summary table

| # | Metric | Concept tier | IWF rule? | Literature-backed concept? | Specific numeric thresholds literature-validated? |
|---|---|---|---|---|---|
| 1 | Hip ROM | 2+3 | No | Yes (joint ROM is standard) | No — further review required |
| 2 | Knee ROM | 2+3 | No | Yes (joint ROM is standard) | No — further review required |
| 3 | Hip Peak | 2+3 | No | Yes (peak joint angle is standard) | No — further review required |
| 4 | Knee Peak | 2+3 | No | Yes (peak joint angle is standard) | No — further review required |
| 5 | Sequential Delay | 2+3 | No | Partially (segment-sequencing concept exists; this exact timing operationalization is project-specific, with a signed-vs-absolute `argmax` caveat) | No — further review required |
| 6 | Synchronization | 2+3 | No | Yes (Pearson r is a standard synchrony statistic) | No — further review required |
| 7 | Hip Dominance | 2+3 | No | Partially (qualitative concept common in coaching; this exact ratio is project-specific) | No — further review required |
| 8 | Correlation | 2+3 | No | Yes (same as Synchronization) | No — further review required |
| 9 | Rate of Force Development | 2+3 | No | **No** — true RFD is force-based; this is a kinematic (angular-acceleration) proxy, not the textbook quantity | No — further review required, and the name itself should be reconsidered |
| 10 | Confidence Weighting | 3 only | No | N/A (engineering reliability heuristic, not a biomechanical claim) | No — thresholds/weights are project-chosen constants |
| 11 | Anthropometric Normalization | 2+3 | No | Yes (reference-length scaling is a standard general technique) | No — further review required (reference-length choice not literature-validated) |

**None of the eleven metrics implement or reference International
Weightlifting Federation judging rules.** All eleven are internal
engineering metrics (named "Rule A–D" plus derived ROM/peak/normalization
features in the original research notebook this project inherited),
conceptually inspired by general biomechanics ideas, but validated in this
project only against this project's own 172-sample, folder-labeled
dataset — not against an external biomechanics or competition-judging
gold standard.

---

## 5. Overall recommendation

Before any claim of clinical, competitive, or coaching validity is made
about this system's outputs:

1. Conduct a **targeted literature review** for each metric in §2 where
   "further literature review is required" is stated, specifically
   searching for weightlifting/powerlifting biomechanics literature on
   hip-knee extension sequencing, coordination/synchrony measures, and
   rate of force development.
2. **Rename or caveat "Rate of Force Development"** (§2.9) in
   user-facing output, since it is a kinematic proxy, not a force
   measurement — this is the single largest terminology/validity gap
   identified in this review.
3. Validate the **learned adaptive thresholds** (§3, item 5) against an
   independent dataset or expert-coach review, since they currently
   reflect only this project's own 172 labeled clips.
4. Consider a **3-D or multi-camera validation study** on a subset of
   clips to quantify how much single-camera perspective distortion affects
   the ROM/peak/velocity metrics reported here.

This document makes no further code or model changes; it is a scientific
review artifact only.
