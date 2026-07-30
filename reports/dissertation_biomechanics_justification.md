# Dissertation Biomechanics Justification — Hip & Knee Metrics

**Purpose of this document.** This report exists solely to **strengthen the
scientific defensibility** of the Hip & Knee module's existing metrics for
dissertation writing and viva defense. **No source code, dataset, model
weights, Rule A–D formulas, thresholds, or the prediction pipeline were
changed to produce this document** — it is a documentation-only artifact
that describes, justifies, and honestly scopes the *already-finalized*
implementation.

**Citation policy.** No citations are invented anywhere in this document.
Where a metric corresponds to a broadly-taught biomechanics concept, that
concept is named generically (e.g. "joint range of motion," "rate of force
development as a strength-training construct"). Wherever a specific
numeric threshold, published normative value, or named study would
normally require a citation, this document either omits the claim or
states plainly: **"no literature validation is claimed for this specific
value; further review is required."**

**Three-tier distinction (used throughout).** Every metric below is
explicitly tagged against three categories that must not be conflated:

| Tag | Meaning |
|---|---|
| **[IWF]** | Official International Weightlifting Federation judging rules (lift validity as ruled by referees). **A repo-wide search confirms no IWF rule text, threshold, or citation is implemented anywhere in this codebase.** None of the metrics below are IWF criteria, and none should be presented as such in the dissertation. |
| **[Biomechanics]** | A general, widely-taught sports-biomechanics/kinesiology principle (e.g. joint angle from three landmarks, angular velocity as a time-derivative, force plate-based RFD). Cited only at the level of "this is a standard/textbook concept," never with a specific author/year/journal that isn't verifiable in this codebase. |
| **[Implementation]** | The specific engineering choice made in this repository (exact formula, smoothing filter, weighting scheme, threshold-learning procedure). This is what was actually built and is the only part being defended as "this project's contribution." |

---

## Rule A — Sequential Extension

**1. What it measures biomechanically.** The timing offset between the
frame at which knee angular velocity reaches its peak and the frame at
which hip angular velocity reaches its peak:
$$\text{delay} = \frac{t_{hip\_peak} - t_{knee\_peak}}{\text{fps}}, \quad t_{joint\_peak} = \arg\max_t \dot\theta_{joint}(t)$$
computed in `rule_a_sequential_extension`
([hip_knee_biomechanics.py](../src/features/hip_knee_biomechanics.py#L305))
from the bilaterally-averaged, Savitzky–Golay-smoothed hip/knee angle
sequences. A positive delay means the hip's velocity peak occurs after the
knee's.

**2. Why meaningful in Olympic Weightlifting [Biomechanics].** The
snatch and clean & jerk both rely on a coordinated "triple extension" of
the ankle, knee, and hip during the pull — a proximal-to-distal segment
sequencing pattern that is a commonly discussed concept in
weightlifting/jumping biomechanics coaching and instructional material.
Whether the hip contribution peaks appropriately relative to the knee is
one intuitive way to characterize whether that sequencing pattern was
executed as intended. **[IWF]**: the IWF's own judging criteria do not
formally define or measure "sequential extension timing" — they judge the
final lift outcome (successful lockout, no press-out, etc.), not
intermediate joint-timing kinematics. This metric is not, and does not
claim to be, an IWF rule.

**3. Why appropriate for a markerless RGB-video system
[Implementation].** Computing a *timing* offset only requires knowing
*when* each joint's velocity peaks, which is derivable purely from 2-D
joint angle sequences over time — no absolute distance, force, or 3-D
depth information is required. This makes it one of the more
naturally-suited metrics for a single-camera, pose-estimation-only
pipeline (YOLO11-pose), since timing/sequencing information survives
2-D projection reasonably well even though absolute magnitudes do not.

**4. Vision-based proxy vs. laboratory kinetic measurement.** This is a
**kinematic** (motion-only) timing measurement, not a kinetic one — no
force, torque, or muscle activation is measured. Laboratory studies of
segmental sequencing typically combine motion capture with force plates
and/or EMG to relate timing to force/power production; this
implementation captures only the *timing* half of that picture, from
video alone.

**5. Advantages.** Simple and deterministic; interpretable sign (which
joint leads); computed from bilaterally-averaged, smoothed angle data,
which reduces single-side occlusion and frame-jitter noise relative to a
naive single-side, unsmoothed computation.

**6. Limitations.** Uses **signed** velocity `argmax` rather than
`argmax(|velocity|)`, so it specifically finds the *peak positive*
angular velocity frame for each joint — a subtlety worth being explicit
about in the dissertation, since it means the metric is not simply "the
frame of largest-magnitude angular speed" but "the frame of fastest
extension in the positive direction," which may or may not coincide with
what an examiner would intuitively expect from the name. Sparse sampling
(20 frames/clip) limits temporal resolution. As a second-order-sensitive
quantity built from twice-differentiated pose data (angle → velocity), it
is inherently more noise-prone than the raw angle signal itself, even
after smoothing. No published normative "ideal" delay value is used —
the good/poor ranges applied for scoring are learned entirely from this
project's own dataset.

**7. Future improvements (not implemented).** Cross-validating this
timing measure against marker-based motion capture or higher-frame-rate
video could quantify how much temporal resolution is lost to the current
20-sample sparsity; a sensitivity analysis of `argmax(signed)` vs.
`argmax(|·|)` could clarify which definition better matches the intended
biomechanical construct; a targeted literature search for
weightlifting-specific proximal-to-distal timing values would allow the
learned thresholds to be compared against an external reference.

---

## Rule B — Hip Dominance

**1. What it measures biomechanically.** The ratio of peak hip angular
velocity magnitude to peak knee angular velocity magnitude over the clip:
$$\text{HipDominance} = \frac{\max_t|\dot\theta_{hip}(t)|}{\max_t|\dot\theta_{knee}(t)| + \epsilon}$$
(`rule_b_hip_dominance`,
[hip_knee_biomechanics.py](../src/features/hip_knee_biomechanics.py#L312);
$\epsilon = 10^{-6}$ is a divide-by-zero guard only).

**2. Why meaningful in Olympic Weightlifting [Biomechanics].** Coaching
literature commonly distinguishes "hip-dominant" from "knee-dominant"
lifting/jumping strategies as a qualitative descriptive concept — athletes
who complete the pull primarily by driving hip extension versus those who
rely comparatively more on knee extension. A simple ratio of peak angular
velocities is a reasonable, interpretable first-order way to place a given
lift on that spectrum. **[IWF]**: not an IWF judging criterion — the IWF
does not evaluate or rule on "hip dominance."

**3. Why appropriate for a markerless RGB-video system
[Implementation].** Being a *ratio* of two angular velocities in the same
units, this metric is dimensionless and camera-scale-invariant — it does
not require knowing absolute distances, so it degrades gracefully under
the depth/scale ambiguity inherent to a single 2-D camera.

**4. Vision-based proxy vs. laboratory kinetic measurement.** This is a
**kinematic velocity ratio**, not a measurement of joint power or torque
contribution. A true "which joint dominates the lift" analysis in a
biomechanics lab would typically use inverse dynamics (requiring segment
mass/inertia and force-plate ground reaction force) to compute actual
joint power contributions — that is explicitly **not** what is computed
here. This implementation should be described in the dissertation as a
**velocity-based proxy** for the qualitative "hip-dominant vs.
knee-dominant" coaching concept, not as a power or force-contribution
measurement.

**5. Advantages.** Simple, interpretable, dimensionless, scale-invariant;
computed directly from the same smoothed angle sequences already produced
for Rule A, requiring no additional pose-processing.

**6. Limitations.** Compares only the single peak instant of each joint —
two lifts with very different overall velocity *profiles* could still
share an identical peak ratio; ignores the *timing* relationship between
the two peaks (handled separately, and imperfectly, by Rule A); the name
"dominance" implies a mechanical/causal relationship (force or power
contribution) that a velocity ratio alone cannot establish without
inverse-dynamics data, which this pipeline does not compute.

**7. Future improvements (not implemented).** Incorporating estimated
joint torque/power (via inverse dynamics, if anthropometric segment
parameters and ground-reaction-force estimates were ever added) would
bring this metric closer to a genuine "dominance" measurement rather than
a velocity proxy; reporting the full velocity-ratio time series (rather
than a single peak-to-peak snapshot) could capture profile differences the
current single-ratio metric misses.

---

## Rule C — Synchronization

**1. What it measures biomechanically.** The Pearson correlation
coefficient between the knee and hip angular velocity time-series over
the clip:
$$r = \text{Pearson}\big(\dot\theta_{knee}(t), \dot\theta_{hip}(t)\big)$$
(`rule_c_synchronization`,
[hip_knee_biomechanics.py](../src/features/hip_knee_biomechanics.py#L318),
guarded against zero-variance/degenerate signals). The reported
"Synchronization" score is this coefficient rescaled to `[0, 100]` via
min-max normalization against ranges learned from this project's training
data (`normalize_score`).

**2. Why meaningful in Olympic Weightlifting [Biomechanics].** Whether
the hip and knee move in a temporally coordinated fashion (rather than
disjointed/erratic relative timing) is a reasonable general marker of
movement coordination quality, a concept broadly discussed in movement
coordination and motor-control literature. **[IWF]**: not an IWF
criterion — IWF judging does not assess statistical correlation between
joint velocity signals.

**3. Why appropriate for a markerless RGB-video system
[Implementation].** Pearson correlation requires only the shape/timing of
two signals, not their absolute magnitude or 3-D position — this makes it
naturally robust to the scale ambiguity of monocular video, since
correlation is invariant to linear rescaling of either input signal.

**4. Vision-based proxy vs. laboratory kinetic measurement.** This is a
purely **kinematic, statistical** synchrony measure. It is not a
neuromuscular coordination measurement (which in a laboratory setting
might use EMG cross-correlation between muscle groups) — it measures
*joint-angle-derived velocity signal* correlation only, from video-derived
pose estimates.

**5. Advantages.** Bounded, well-understood statistic (`[-1, 1]`);
computed over the whole clip rather than a single frame/instant, making it
less sensitive to any one frame's tracking error than Rule A or Rule D;
explicit zero-variance guard prevents runtime failure on degenerate
(near-static) clips.

**6. Limitations.** Pearson's r captures only **linear** association — a
genuinely coordinated but phase-shifted or nonlinearly related pair of
velocity profiles could score poorly despite being biomechanically
sensible (no lag-search/cross-correlation-at-lag is implemented); a high
correlation could also arise from a shared whole-body motion or camera
artifact rather than genuine joint-to-joint coordination, since no control
signal (e.g. pelvis/torso motion) is regressed out; the `[0,100]` score
rescaling is entirely relative to this project's own observed min/max, not
an externally validated scale.

**7. Future improvements (not implemented).** Exploring lagged
cross-correlation (allowing a small, physiologically expected phase
offset rather than penalizing it) could better separate genuine
desynchronization from an expected minor lead/lag; investigating whether
whole-body motion should be regressed out before correlating hip/knee
signals could reduce shared-artifact confounds.

---

## Rule D — Rate of Force Development Approximation

**1. What it measures biomechanically.** The ratio of peak hip angular
acceleration to peak knee angular acceleration:
$$\text{RFD}_{ratio} = \frac{\max_t|\ddot\theta_{hip}(t)|}{\max_t|\ddot\theta_{knee}(t)| + \epsilon}$$
(`rule_d_rfd`,
[hip_knee_biomechanics.py](../src/features/hip_knee_biomechanics.py#L325)),
where angular acceleration is the second time-derivative
(`np.gradient` applied twice) of the smoothed angle signal.

**2. Why meaningful in Olympic Weightlifting [Biomechanics].** Rate of
Force Development is a well-established strength-training construct,
formally defined as $RFD = dF/dt$ of measured force (typically from a
force plate), used to characterize an athlete's explosiveness. Explosive
triple extension is central to weightlifting performance, which motivates
wanting *some* proxy for explosiveness in a video-only system.

**3. Why appropriate for a markerless RGB-video system
[Implementation].** No force-measurement hardware (force plate, load
cell, instrumented barbell) is available in this pipeline — video-derived
angular acceleration is the closest quantity computable from pose data
alone, and is used here specifically because it requires no additional
instrumentation beyond the existing pose-estimation pipeline.

**4. Vision-based proxy vs. laboratory kinetic measurement — critical
distinction for the dissertation.** **This metric is explicitly a
kinematic (angular-acceleration) proxy, and is NOT the textbook,
force-based Rate of Force Development.** True RFD requires measured force
over time; this implementation substitutes angular acceleration, which is
only loosely and indirectly related to force via inverse dynamics
(which itself requires segment mass, moment of inertia, and other
anthropometric parameters not estimated anywhere in this codebase). **The
dissertation should present "Rule D" as "an RFD-inspired angular
acceleration proxy, motivated by the RFD construct in strength-training
literature but not equivalent to it," rather than as a validated force
measurement.** This is the single clearest example among the eleven
metrics where the vision-based-proxy-vs-laboratory-measurement distinction
must be stated explicitly to avoid overclaiming.

**5. Advantages.** Computable from pose-only data with no additional
instrumentation; directionally plausible (faster angular acceleration
plausibly correlates with more explosive effort, all else equal);
dimensionless ratio, consistent with Rules B/C in form.

**6. Limitations.** Second derivatives of noisy pose-tracking data are
inherently much noisier than first derivatives (angle → velocity →
acceleration compounds tracking jitter), even after Savitzky–Golay
smoothing; converting angular acceleration to actual joint torque/force
would require anthropometric segment parameters not estimated in this
pipeline; no published RFD-ratio threshold exists for this specific lift,
and the "good"/"poor" ranges used are learned solely from this project's
172-sample dataset, not from an external force-plate study.

**7. Future improvements (not implemented).** If force-plate or
instrumented-barbell data were ever collected alongside video for a
validation subset, the angular-acceleration proxy could be directly
correlated against true measured RFD to quantify how well it approximates
the textbook construct; a full inverse-dynamics model (with estimated or
measured segment mass/inertia) could bring this metric conceptually closer
to genuine force-based RFD; renaming the user-facing label to make the
"approximation"/proxy nature explicit (e.g. "Angular Acceleration Ratio
(RFD proxy)") would further reduce the risk of overclaiming in
presentation, without requiring any change to the underlying computation.

---

## Hip ROM

**1. What it measures biomechanically.** The total angular excursion of
the hip joint (shoulder–hip–knee angle) across the sampled clip:
$$\text{HipROM} = \max_t(\theta_{hip}(t)) - \min_t(\theta_{hip}(t))$$
(`compute_rom_and_peaks`,
[hip_knee_predict.py](../src/models/hip_knee_predict.py#L152)), from the
bilaterally-averaged, smoothed hip angle sequence, using
$\theta = \cos^{-1}\!\left(\dfrac{\vec{BA}\cdot\vec{BC}}{|\vec{BA}||\vec{BC}|}\right)$
(`calculate_angle`).

**2. Why meaningful in Olympic Weightlifting [Biomechanics].** Hip range
of motion during the pull/hinge phase is a widely used general kinematic
descriptor of hip-hinge depth and engagement in lifting movements — a
standard, textbook-level goniometric measure, not specific to this
project.

**3. Why appropriate for a markerless RGB-video system
[Implementation].** Because it is an **angle** (not a distance), Hip ROM
is inherently scale-invariant: it does not depend on how far the camera is
from the athlete or on the athlete's absolute height, which makes it
naturally robust to the depth ambiguity of monocular video — a
significant advantage over any linear/pixel-distance-based measure in this
setting.

**4. Vision-based proxy vs. laboratory kinetic measurement.** This is a
purely **kinematic** angular measurement, directly analogous to
goniometric ROM assessment; no force or kinetic quantity is involved, so
the proxy/laboratory distinction is less consequential here than for Rule
D — angle-from-video is a fairly direct measurement of the kinematic
quantity it claims to represent, modulo 2-D projection error (point 6).

**5. Advantages.** Scale-invariant by construction; bilateral averaging
reduces single-side occlusion noise; Savitzky–Golay smoothing reduces
sensitivity to single-frame pose jitter before the min/max is taken.

**6. Limitations.** Computed from a single 2-D camera view — true 3-D hip
flexion is subject to perspective foreshortening depending on the
athlete's orientation to the camera, which single-view pose estimation
cannot correct for; sparse temporal sampling (20 frames/clip) could miss
the true min/max if it occurs between sampled frames; no externally
published normative "good" hip-ROM range is used — the good/poor scoring
ranges are learned entirely from this project's own dataset.

**7. Future improvements (not implemented).** Multi-camera or 3-D
pose-estimation validation on a subset of clips could quantify how much
single-view perspective distortion affects the reported ROM values;
increasing temporal sampling density (or using the full frame sequence)
could reduce the risk of missing the true excursion extremes.

---

## Knee ROM

**1. What it measures biomechanically.** The total angular excursion of
the knee joint (hip–knee–ankle angle) across the sampled clip:
$$\text{KneeROM} = \max_t(\theta_{knee}(t)) - \min_t(\theta_{knee}(t))$$
(`compute_rom_and_peaks`,
[hip_knee_predict.py](../src/models/hip_knee_predict.py#L152)), computed
identically to Hip ROM with $A=\text{hip}, B=\text{knee}, C=\text{ankle}$.

**2. Why meaningful in Olympic Weightlifting [Biomechanics].** Knee
extension range is a standard general kinematic descriptor of leg drive
during the pull, and is commonly discussed alongside hip motion when
describing lower-body lift mechanics.

**3–7.** Identical justification, scale-invariance argument,
vision-based-proxy discussion, advantages, limitations, and future-work
recommendations as Hip ROM above, substituting the knee joint. As with Hip
ROM, this is a direct (not proxy) kinematic angular measurement subject
only to 2-D projection and sparse-sampling caveats, not a force/kinetic
approximation.

---

## Peak Hip Velocity

**1. What it measures biomechanically.** The peak **linear** (vertical,
pixel-space) velocity of the hip joint during the clip, both in raw pixels
per second and anthropometrically normalized:
$$v_{hip,raw} = \max_t\left|\frac{d}{dt}\, y_{hip}(t)\right|, \qquad v_{hip,norm} = \frac{v_{hip,raw}}{\text{leg\_length}_{px}}$$
where $y_{hip}(t) = \tfrac{1}{2}(y_{hip}^L(t) + y_{hip}^R(t))$ is the
bilateral-average vertical pixel position of the hip
(`compute_linear_velocity`,
[hip_knee_biomechanics.py](../src/features/hip_knee_biomechanics.py#L246)).
**Important clarification for the dissertation**: this is a *linear*
(translational, vertical-position) velocity of the hip joint's screen
position — a distinct quantity from the *angular* hip velocity
($\dot\theta_{hip}$) used internally by Rules A/B/D. Both exist in this
codebase; "Peak Hip Velocity" as a named, reported feature specifically
refers to the anthropometrically-normalized linear velocity.

**2. Why meaningful in Olympic Weightlifting [Biomechanics].** How fast
the hip translates vertically during the pull is a general, intuitive
descriptor of explosiveness/pulling speed, relevant to characterizing the
second-pull phase of the clean/snatch.

**3. Why appropriate for a markerless RGB-video system
[Implementation].** Unlike an angle, a *linear* pixel velocity is
inherently distance-and-height-dependent — it changes with camera zoom
and athlete height. This implementation explicitly compensates for that
by dividing by the clip's own anthropometric reference length
(`leg_length`, median thigh+shank segment length across the clip,
`compute_reference_lengths`), which is the appropriate and necessary
adaptation for using a *linear* velocity measure in a monocular,
uncalibrated-camera setting.

**4. Vision-based proxy vs. laboratory kinetic measurement.** This is a
**kinematic** measurement (position derivative), not a kinetic one — no
force or momentum is computed. It is also, itself, a **proxy for true 3-D
hip velocity**: what is measured is the *vertical pixel displacement over
time*, projected through an uncalibrated single camera, then rescaled by
an anthropometric reference length — an approximation of true metric
(cm/s or m/s) hip velocity, not an actual metric measurement (no camera
calibration or known real-world scale factor is used).

**5. Advantages.** Anthropometric normalization (leg-length-based)
specifically corrects for the biggest confound in monocular linear-
velocity measurement (camera distance/athlete height), making values more
comparable across clips than raw pixel velocity would be; derived from the
same robust (median-reduced) reference-length computation used elsewhere
in the pipeline.

**6. Limitations.** Only the **vertical** velocity component is measured
— horizontal hip translation (e.g. during the turnover/pull-under phase)
is not captured; the reference length used for normalization is itself
measured from the same single 2-D view and is subject to the same
perspective foreshortening it is meant to help correct for; no
camera calibration or real-world scale factor is used, so normalized
values are in "leg-lengths per second," not true metric units (m/s); this
prevents direct comparison to laboratory-reported RFD/velocity values
from force-plate or motion-capture studies, which typically report metric
units.

**7. Future improvements (not implemented).** Camera calibration (e.g.
via a known reference object in frame) could convert normalized
"leg-lengths/second" values into approximate metric velocities, enabling
direct comparison with laboratory literature; incorporating the
horizontal velocity component (full 2-D velocity vector magnitude, rather
than vertical-only) could give a more complete picture of hip
translational speed.

---

## Peak Knee Velocity

**1. What it measures biomechanically.** The peak linear (vertical,
pixel-space) velocity of the knee joint, raw and anthropometrically
normalized, computed identically to Peak Hip Velocity with the knee's
bilateral-average vertical position substituted
(`compute_linear_velocity`,
[hip_knee_biomechanics.py](../src/features/hip_knee_biomechanics.py#L246)).
As with Peak Hip Velocity, this is distinct from the *angular* knee
velocity used internally by Rules A/B/D.

**2–7.** Identical justification, appropriateness argument, proxy
discussion, advantages, limitations, and future-work recommendations as
Peak Hip Velocity above, substituting the knee joint's vertical pixel
trajectory.

---

## Correlation

**1. What it measures biomechanically.** The raw Pearson correlation
coefficient between the knee and hip angular velocity signals over the
clip — the same underlying quantity computed for Rule C's
"Synchronization" score, but exposed as an unnormalized `[-1, 1]`
coefficient (`PredictionReport.correlation`,
[hip_knee_predict.py](../src/models/hip_knee_predict.py#L229)) rather than
a rescaled `[0, 100]` score.

**2–6.** Identical biomechanical meaning, Olympic-weightlifting rationale,
implementation appropriateness, vision-based-proxy discussion,
advantages, and limitations as Rule C — Synchronization above (§"Rule C").
This entry is documented separately in this report only because the
pipeline exposes the raw coefficient and the rescaled score as two
distinct named fields (`correlation` vs. `rule_c_score`); scientifically
they share the same justification and the same caveats (linear-only
association, no lag search, potential shared-motion confound).

**7. Future improvements (not implemented).** Same as Rule C —
Synchronization: lagged cross-correlation and whole-body-motion
deconfounding are the primary avenues identified, not yet implemented.

---

## Confidence Weighting

**1. What it measures biomechanically.** Not a biomechanical quantity at
all — it is a **measurement-reliability weighting mechanism**. Per-frame
pose confidence (mean YOLO11-pose keypoint confidence over the 8 required
keypoints) is aggregated into per-rule confidence values grounded in the
*specific frames each rule depends on* (peak-timing frames for Rules
A/B/D; all frames for Rule C, since it uses the entire signal),
(`compute_rule_confidences`,
[hip_knee_biomechanics.py](../src/features/hip_knee_biomechanics.py#L263)),
then blended into an overall confidence using the same `RULE_WEIGHTS =
(0.35, 0.30, 0.20, 0.15)` used to blend the Rule A–D scores themselves
(`compute_overall_confidence`).

**2. Why meaningful in Olympic Weightlifting [Biomechanics] /
[Implementation].** This has no direct biomechanical-literature analogue
as a "metric of the lift" — it is a data-quality/uncertainty-
quantification safeguard, conceptually similar in spirit (not method) to
how sensor-fusion or measurement systems down-weight low-confidence
readings. Its relevance to weightlifting analysis is purely that it
determines *how much to trust* the biomechanical metrics above when
occlusion, motion blur, or ambiguous multi-person detection degrades pose
quality. **[IWF]**: entirely unrelated to IWF judging.

**3. Why appropriate for a markerless RGB-video system
[Implementation].** Markerless pose estimation is far more prone to
transient detection failures (occlusion by the bar/plates, motion blur at
high bar speed, multiple people in frame) than marker-based laboratory
capture. A confidence-weighting layer is a natural and necessary
mitigation specifically because this is a video-pose-based (not
marker-based) system — a laboratory marker-capture system would not
typically need an analogous mechanism, since marker tracking failures are
rarer and usually explicitly flagged/interpolated rather than silently
producing a low-confidence-but-still-numeric detection.

**4. Vision-based proxy vs. laboratory kinetic measurement.** Not
applicable in the usual sense — this mechanism does not claim to measure
any biomechanical quantity; it exists only to express uncertainty about
the metrics that do.

**5. Advantages.** Makes reliability visible rather than silently trusting
every prediction equally; grounds each rule's confidence in the specific
frames it actually depends on, rather than one blanket clip-level number;
reuses the same, already-transparent `RULE_WEIGHTS` importance weighting
used for the scores themselves, keeping the design internally consistent
and auditable.

**6. Limitations.** The `LOW_CONFIDENCE_THRESHOLD = 0.6` cutoff and the
`RULE_WEIGHTS` reuse are project-chosen constants, not derived from a
formal sensitivity or calibration study; YOLO11-pose's own per-keypoint
confidence is a model-internal detection estimate, not an independently
calibrated measurement-uncertainty value, so it may not linearly
correspond to true anatomical landmark accuracy; confidence weighting can
mask a systematically flawed rule computation (e.g. a formula-level
subtlety) — high pose confidence does not guarantee the downstream rule
formula captures the intended construct.

**7. Future improvements (not implemented).** A calibration study
comparing YOLO11-pose confidence against independently-assessed keypoint
accuracy (e.g. against a labelled ground-truth subset) could validate or
adjust the `0.6` threshold; reporting a calibrated per-metric uncertainty
interval, rather than a single blended confidence scalar, could give a
richer picture of reliability than the current scheme.

---

## Anthropometric Normalization

**1. What it measures biomechanically.** Not a biomechanical measurement
itself, but a **scaling/normalization procedure**: raw pixel-space
distances and velocities (hip/knee linear ROM, peak linear velocity) are
divided by a clip-level anthropometric reference length — the median
thigh+shank segment length (`leg_length`) across the clip
(`compute_reference_lengths`,
[hip_knee_biomechanics.py](../src/features/hip_knee_biomechanics.py#L143)),
via
$$\text{value}_{normalized} = \frac{\text{value}_{raw\_px}}{\text{leg\_length}_{px}}$$
Separately, the LSTM's raw *input* features are normalized by translating
to the hip-midpoint origin and scaling by **torso length**
(`normalize_by_body_proportion`,
[hip_knee_pose_utils.py](../src/features/hip_knee_pose_utils.py#L147)) —
a **different** reference length than the one used for the reported
linear-ROM/velocity features, a distinction worth stating explicitly in
the dissertation to avoid implying the two normalizations are on the same
scale.

**2. Why meaningful in Olympic Weightlifting [Biomechanics] /
[Implementation].** Athletes vary substantially in height and limb length,
and camera setups vary in distance/zoom across recording sessions — any
raw pixel-space distance/velocity metric is confounded by both factors
unless corrected. Normalizing by a body-proportional reference length is
a standard general technique in markerless-video biomechanics to make
measurements more comparable across subjects and recording conditions.
**[IWF]**: unrelated to IWF judging, which does not perform any such
normalization.

**3. Why appropriate for a markerless RGB-video system
[Implementation].** This is specifically necessary *because* the system
is monocular and uncalibrated — a laboratory system with calibrated
multi-camera 3-D reconstruction would already report metric (cm/m) units
and would not need a body-proportional pixel-space normalization; this
implementation is a deliberate, appropriate adaptation to the constraints
of single-camera, non-calibrated video.

**4. Vision-based proxy vs. laboratory kinetic measurement.** This is a
**normalization procedure for kinematic (distance/velocity) quantities**,
not a kinetic one. It should be described as a practical proxy for
metric-unit measurement (true cm/m distances, as a calibrated lab system
would report) rather than an equivalent substitute — the normalized
values are in "reference-lengths" (or "reference-lengths per second"),
not real-world units.

**5. Advantages.** Uses the **median** (not mean) across the clip for
robustness to single-frame jitter/occlusion; leg length (thigh+shank sum)
tracks true limb length better than a straight hip-to-ankle line, which
foreshortens as the knee bends; degenerate/near-zero reference lengths are
handled with a safe fallback (returns the raw value) rather than raising
or producing `inf`/`nan`.

**6. Limitations.** The reference length is measured from the same single
2-D view as the quantity it normalizes, so it is subject to the same
perspective distortion it is meant to help correct for; no independent
ground-truth (e.g. measured limb length in cm) validates that the
pixel-based reference is proportionally consistent across athletes; two
different reference lengths (leg length vs. torso length) are used in
different parts of the pipeline, so normalized values from different
parts of the system are not directly comparable to each other on the same
scale; no camera calibration means normalized values remain in
relative ("reference-lengths") units rather than true metric units.

**7. Future improvements (not implemented).** Validating the pixel-based
reference length against actual measured anthropometric data (if
available for a subset of athletes) could confirm or refine its use;
camera calibration (e.g. a known reference object in frame) could convert
normalized values into approximate real-world metric units, enabling
direct comparison with laboratory-reported measurements; a literature
review comparing reference-length choices (leg length vs. torso length
vs. estimated stature) could clarify whether the current choice is
optimal.

---

## Summary: Tier classification across all 11 metrics

| Metric | [IWF] rule? | [Biomechanics] principle invoked | [Implementation] specificity |
|---|---|---|---|
| Rule A — Sequential Extension | No | Proximal-to-distal segment sequencing (general concept) | Signed-velocity peak-timing offset; project-specific |
| Rule B — Hip Dominance | No | Hip- vs. knee-dominant lifting strategy (qualitative coaching concept) | Peak angular-velocity ratio; project-specific |
| Rule C — Synchronization | No | Movement coordination / synchrony (general concept) | Pearson r on angular velocity signals; project-specific |
| Rule D — RFD Approximation | No | Rate of Force Development (force-based, in strength-training literature) | Angular-acceleration ratio — **explicitly a kinematic proxy, not true RFD** |
| Hip ROM | No | Joint range of motion (standard goniometric concept) | Bilateral, smoothed angle excursion; project-specific |
| Knee ROM | No | Joint range of motion (standard goniometric concept) | Bilateral, smoothed angle excursion; project-specific |
| Peak Hip Velocity | No | Linear joint velocity / explosiveness (general concept) | Anthropometrically-normalized vertical pixel velocity; project-specific |
| Peak Knee Velocity | No | Linear joint velocity / explosiveness (general concept) | Anthropometrically-normalized vertical pixel velocity; project-specific |
| Correlation | No | Same as Rule C (Pearson synchrony) | Raw coefficient (unnormalized) exposed alongside the Rule C score |
| Confidence Weighting | No | N/A — measurement-reliability engineering, not a biomechanical construct | Per-rule, frame-grounded confidence blend; project-specific |
| Anthropometric Normalization | No | Body-proportional scaling (general markerless-video technique) | Leg-length (and separately, torso-length) reference scaling; project-specific |

**No metric in this system implements, references, or should be presented
as an IWF judging rule.** Every metric is a general biomechanics-inspired
concept operationalized through this project's specific engineering
choices, and every choice's numeric thresholds are learned solely from
this project's own dataset rather than from an externally validated
biomechanics or competition-judging standard. This document makes no
literature-validation claims beyond what is stated above, and identifies
Rule D (RFD Approximation) as the metric requiring the clearest
proxy-vs-laboratory-measurement caveat in the dissertation text.
