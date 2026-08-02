# Final Peer Review — Hip & Knee Lift-Quality Analysis Module

**Document type**: Simulated peer review, written as three anonymous
reviewers plus an editor decision, in the style of an IEEE/Elsevier
Computer Vision / Sports Biomechanics venue. This is a **documentation-only
artifact**. No source code, dataset, trained model, threshold file, or any
existing report was modified to produce this review — the module is
reviewed exactly as it stands, feature-complete.

**Citation policy**: No citations are invented anywhere in this review.
Where a reviewer comment references a "commonly used technique" or
"standard practice," this refers to broadly-known concepts in the
field, not a specific verifiable publication — this review does not
claim any specific literature validation that is not already
established, and does not fabricate references.

**Scope reviewed**: dataset construction and labeling, feature extraction,
YOLO11-pose estimation, anthropometric normalization, the LSTM classifier,
Captum-based explainability, Rule A–D biomechanical analysis, confidence
weighting, performance scoring, the REST API, robustness validation, and
the scientific-documentation reports (`reports/scientific_validation_of_rules.md`,
`reports/dissertation_biomechanics_justification.md`, and others).

---

## Reviewer 1 — Computer Vision & Machine Learning Methodology

### Summary

The submission presents a monocular RGB-video pipeline for classifying
weightlifting lift quality (binary: Good/Poor) from lower-body pose
sequences. Pose is extracted with off-the-shelf YOLO11-pose (COCO-17),
reduced to 8 lower-body keypoints, normalized by a body-proportional
scale, and classified with a small 2-layer LSTM. A recent iteration added
class-weighting, dropout, and reduced hidden size to address a
documented overfitting/class-imbalance problem, improving test accuracy
from 71.4% to 88.6% on a 35-sample held-out split. A Captum Integrated
Gradients explainability layer was added afterward. The core CV/ML
methodology is competent and pragmatic for the project's evident resource
constraints, but the evaluation protocol is not sufficiently rigorous to
support strong generalization claims.

### Major strengths

- **Honest, documented root-cause analysis of a real overfitting/bias
  problem.** The model-improvement report (`reports/model_improvement_report.md`)
  correctly diagnoses class imbalance (112 Good / 60 Poor, ~1.87:1) and
  over-parameterization (54,402 params for 137 training sequences) as the
  causes of a strongly majority-biased classifier (9/12 "Poor" samples
  misclassified), then applies well-targeted, minimal fixes (class-weighted
  loss, dropout=0.3, halved hidden size, slightly increased weight decay)
  rather than an unprincipled hyperparameter search. This is good ML
  engineering practice.
- **Off-the-shelf pose estimator is a defensible choice** given the
  project scope — using YOLO11-pose rather than training a custom
  keypoint detector is appropriate when the research question is about
  downstream lift-quality classification, not pose estimation itself.
- **Anthropometric, body-proportional input normalization**
  (`normalize_by_body_proportion`: hip-centered, torso-length-scaled)
  is a sound design choice that removes camera-distance and athlete-height
  confounds from the raw keypoint coordinates before they reach the LSTM.
- **Explainability is grounded in a real, working Captum Integrated
  Gradients computation** (not a placeholder or approximation) with a
  near-zero convergence delta (reported as −0.000000), which is the
  correct sanity check to report for IG and indicates the attribution
  computation is numerically sound on the completeness axiom.
- **One sequence per video** (verified: `hip_knee_build_dataset.py`
  iterates one sequence per video, no per-video augmentation into
  multiple sequences), which avoids the common and serious "leakage"
  failure mode of splitting augmented/duplicated sequences from the same
  source video across train and test.

### Major weaknesses

- **Small dataset for a deep sequence model.** 172 total sequences (137
  train / 35 test) is a small dataset for an LSTM, even a lightweight one.
  While the authors correctly identified and mitigated overfitting
  symptoms, the fundamental data-scarcity constraint remains and is
  likely to limit how far these accuracy numbers generalize to lifts,
  athletes, or camera setups not represented in this sample.
- **Single train/test split, no cross-validation.** All reported metrics
  (88.57% accuracy, 87.32% balanced accuracy) come from **one** stratified
  split (`random_state=42`). With only 35 test samples, the standard error
  on this accuracy estimate is large; no k-fold cross-validation,
  bootstrap confidence interval, or repeated-split variance estimate is
  reported anywhere in the codebase or reports. A single-split accuracy
  number on a 35-sample test set should not be treated as a stable
  generalization estimate.
- **No external/held-out dataset or cross-camera-setup validation.** All
  data appears to come from the same collection effort/environment
  (same "FYP research" video source, same folder-based Good/Poor
  labeling). There is no evidence of testing on footage from a different
  gym, camera, or labeler to assess distribution shift robustness.
- **YOLO11-pose is used purely off-the-shelf, with no domain-specific
  fine-tuning or accuracy validation against a labelled keypoint ground
  truth for this specific use case** (barbell-occluded, motion-blurred
  weightlifting footage differs from YOLO's general training
  distribution). Pose-estimation accuracy on this specific footage type
  is asserted only indirectly (via the downstream classification
  accuracy and the confidence-weighting mechanism), not measured directly.
- **The explainability analysis explains one sample at a time and one
  video was used for the demonstrated run (`13bad.MOV`).** No aggregate/
  average feature-importance analysis across multiple samples or classes
  is presented, so it is unclear whether the reported importance ranking
  (Confidence, Anthropometric Features, Knee ROM/Peak, etc.) is
  representative or specific to that one example.

### Minor comments

- The 8-keypoint reduction from YOLO's 17 COCO keypoints (dropping
  upper-body/face points) is reasonable for this task but is not
  explicitly justified against alternatives (e.g. including elbow/wrist
  for grip-width context) anywhere in the documentation.
- The Integrated Gradients baseline (a zero vector) is well-justified
  given the hip-centered normalization, but only one baseline choice is
  explored; a brief sensitivity check against an alternative baseline
  (e.g. the per-clip mean pose) would strengthen the explainability
  claims.
- `n_steps=50` for Integrated Gradients is a reasonable default but is not
  reported as having been tuned or validated for convergence beyond the
  single reported convergence-delta check.

### Questions likely to be asked during a viva

1. "Your test accuracy is 88.6% on 35 samples — what is the 95%
   confidence interval on that estimate, and have you run repeated
   splits or k-fold cross-validation to check its stability?"
2. "How do you know YOLO11-pose is accurately tracking hip/knee
   keypoints specifically on barbell-occluded weightlifting footage,
   rather than on the general scenes it was originally trained on?"
3. "Since your Good/Poor labels come from a single labeling process, how
   do you know the model isn't learning artifacts of that specific
   labeler's judgment rather than a generalizable notion of lift quality?"
4. "Your Captum analysis explains one video. Would the feature-importance
   ranking change meaningfully on a Poor-classified sample, or on a
   different athlete? Have you checked?"
5. "Why was hidden_size=32 (not some other value) chosen after the
   overfitting fix — was this tuned systematically or chosen as a
   reasonable halving?"

### Recommendation: **Minor Revision**

The core methodology is sound and the overfitting fix is genuinely
well-diagnosed engineering work, but the evaluation protocol (single
split, no cross-validation, no confidence intervals, no external test
data) needs to be either strengthened or, at minimum, explicitly and
prominently caveated as a limitation before the accuracy figures are
presented as a headline result.

---

## Reviewer 2 — Biomechanics Methodology

### Summary

The submission computes four named biomechanical "rules" (A–D:
Sequential Extension, Hip Dominance, Synchronization, Rate of Force
Development) plus range-of-motion, peak-angle, peak-velocity,
correlation, and anthropometric-normalization features, all derived from
2-D joint angles/positions estimated via markerless video. The underlying
mathematics (three-point angle via arccosine, finite-difference
derivatives, Pearson correlation) are standard and correctly implemented.
However, the biomechanical *interpretation* attached to several metrics —
most notably "Rate of Force Development" — extends beyond what the
underlying video-only kinematic data can actually support, and this is
only partially and inconsistently caveated across the project's own
documentation.

### Major strengths

- **The project's own supplementary documentation
  (`reports/scientific_validation_of_rules.md`,
  `reports/dissertation_biomechanics_justification.md`) is unusually
  candid** about the limitations of each metric, explicitly distinguishing
  IWF judging rules (confirmed absent from the codebase), general
  biomechanics concepts, and this project's specific engineering
  implementation. This level of self-scrutiny is not typical in student
  work and is a genuine strength for scientific defensibility.
- **Correct, standard joint-angle formula** (arccosine of the normalized
  dot product between two limb-segment vectors) with a sensible
  degenerate-vector guard, and **bilateral averaging** of left/right sides
  to reduce single-side occlusion noise — both defensible, standard
  choices.
- **Savitzky–Golay smoothing before differentiation** is an appropriate
  and standard technique to reduce the noise amplification inherent in
  computing velocity/acceleration from noisy pose-jitter data via finite
  differences.
- **Anthropometric normalization is well-motivated and correctly
  addresses a real confound** (camera distance / athlete height) for the
  linear (pixel-space) ROM/velocity features, using a robust
  median-across-clip reference length.

### Major weaknesses

- **"Rate of Force Development" is a materially misleading label for
  what is actually computed.** True RFD is $dF/dt$ of measured force
  (typically force-plate-derived); this implementation computes a ratio
  of peak **angular accelerations**, with no force, torque, or mass/
  inertia estimation anywhere in the pipeline. While the project's own
  supplementary reports do caveat this, the metric's user/API-facing name
  (`rate_of_force_development` field in `PredictionReport`, surfaced
  directly in the JSON API response) does not itself carry this caveat —
  a downstream consumer of the API reading only the field name and value
  could easily be misled into thinking true kinetic RFD is being reported.
- **No inverse-dynamics or ground-truth kinetic validation exists
  anywhere in the project** to assess how well any of the kinematic
  proxies (RFD-ratio, Hip Dominance ratio) actually track the biomechanical
  constructs their names invoke. This is an inherent limitation of a
  video-only system, honestly acknowledged in the documentation, but it
  means claims about "explosiveness" or "dominance" remain unverified
  hypotheses rather than validated measurements.
- **Sequential Delay (Rule A) uses signed `argmax` velocity rather than
  `argmax(|velocity|)`**, a subtle implementation choice that means the
  metric technically measures "timing of peak *positive* angular
  velocity" for each joint rather than "timing of peak-magnitude angular
  velocity" — this is flagged in the project's own validation reports but
  not evaluated for whether it changes the metric's practical behavior
  on real data (e.g., could the true peak for a given joint sometimes be
  negative-signed and thus be missed by this formula?).
- **All "good"/"poor" thresholds for every rule are learned solely from
  this project's own 172-sample, single-source dataset** — there is no
  comparison anywhere against an externally published normative range for
  hip/knee ROM, hip-dominance ratio, or synchronization correlation in
  weightlifting. The scientific documentation is honest about this, but
  it remains a fundamental constraint on how far any "Good" vs "Poor"
  classification can be biomechanically generalized beyond this dataset.
- **Correlation and Synchronization are reported as two separate named
  metrics but are mathematically the same Pearson coefficient** (raw vs.
  min-max-rescaled). This is documented in the project's own reports, but
  presenting them as two independent "biomechanical features" (e.g. in
  the explainability feature-importance breakdown) risks
  double-counting the same underlying signal when a reader interprets
  the results.

### Minor comments

- The two-tier "hip_rom"/"hip_peak" (angle-based) vs. "Peak Hip Velocity"
  (linear, pixel-based) terminology is used consistently within the code
  but could be confusing to a reader unfamiliar with the distinction
  between angular and linear kinematics — a clearer, unified glossary
  across all reports (rather than distinguishing it only in the
  dissertation-justification report) would help.
- `epsilon = 1e-6` divide-by-zero guards in the Hip Dominance and RFD
  ratios are a reasonable numerical safeguard but could, in principle,
  produce an artificially very large ratio for a genuinely near-zero
  knee velocity/acceleration clip — this edge case is not explicitly
  discussed.
- No discussion is given of whether/how the sampled-frame count (20
  frames/clip) was chosen relative to a typical Olympic-lift duration,
  which would help justify (or reveal a limitation of) the temporal
  resolution available to Rules A and D.

### Questions likely to be asked during a viva

1. "You call Rule D 'Rate of Force Development.' Can you clearly state,
   right now, what physical quantity it is actually computing, and why
   you chose to keep that name in the API response rather than a more
   accurate one?"
2. "If your labels aren't derived from IWF judging criteria, what
   *specifically* was the labeling protocol used to decide 'Good' vs
   'Poor' for these 172 clips, and how consistent/reliable was it?"
3. "Correlation and Synchronization are the same coefficient reported
   twice, in different forms. Doesn't reporting both as separate
   'features' in your explainability analysis risk over-representing
   that single signal's importance?"
4. "How would you validate the Hip Dominance ratio or the RFD-ratio
   against ground truth, if you had access to a force plate or motion
   capture lab? What would that experiment look like?"
5. "Your Sequential Delay uses signed velocity `argmax`. Can you show a
   concrete example from your dataset where this produces a different
   answer than `argmax(|velocity|)` would?"

### Recommendation: **Major Revision**

The mathematics is implemented correctly and the project's own supporting
documentation shows genuine scientific self-awareness, but the "Rate of
Force Development" naming issue is a substantive scientific-communication
problem that should be fixed (at minimum via renaming/relabeling in
user/API-facing output, and by more prominent in-line caveats) before this
work could be considered ready for a biomechanics-literate audience,
alongside stronger acknowledgement of the fully dataset-relative (not
externally validated) nature of every threshold used.

---

## Reviewer 3 — Software Engineering, Reproducibility, Robustness, and Deployment

### Summary

The codebase is organized into clearly-scoped, single-responsibility
modules (`hip_knee_biomechanics.py`, `hip_knee_scoring.py`,
`hip_knee_predict.py`, `hip_knee_pose_utils.py`, etc.), with a REST API
layer built on top of an already-validated CLI pipeline, and a
dedicated robustness-testing effort that found and fixed four genuine
defects (silent wrong-person tracking, unhandled ffmpeg failures, an
unguarded CLI entry point, and a silent video-writer failure). This is
solid, professional-grade engineering for a student project. The main
gaps are around formal reproducibility artifacts (environment pinning,
automated tests) and production-readiness hardening (auth, rate
limiting) that are explicitly and honestly scoped out rather than hidden.

### Major strengths

- **A dedicated, evidence-based robustness validation effort**
  (`reports/backend_validation.md`) that tested against both synthetic
  edge cases (short/long/low-fps/high-fps videos) and **real footage**,
  and found a genuinely important bug (frames could silently show a
  mirror reflection or bystander instead of the athlete in multi-person
  scenes) — critically, the fix was **empirically refined** after the
  first design (hard-reject on ambiguity) was found to break on real
  4K footage with mirrors, causing 100% frame rejection. This
  iterative, evidence-driven bug-fixing process is exactly the right
  engineering methodology and is well-documented.
- **The REST API is built as a thin wrapper around the existing,
  already-validated CLI pipeline** (`predict_video`) rather than
  duplicating logic — a sound architectural decision that avoids a
  common source of train/serve skew (the API using different code paths
  than the CLI/offline evaluation).
- **Concrete, sensible security-conscious choices in the API**: a 500MB
  upload cap (explicitly citing CWE-400 Uncontrolled Resource
  Consumption), an inference lock to serialize non-thread-safe
  YOLO/LSTM forward passes rather than risking undefined concurrent
  behavior, and separate scratch directories per component (CLI vs. API
  vs. explainability) to avoid concurrent-request collisions.
- **Confidence weighting and thresholds are persisted together as model
  artifacts** (not hardcoded), meaning a retrained model always ships
  with the exact normalization parameters it was trained with — a good
  practice that prevents a common "silently stale threshold" bug class.
- **Consistent, structured logging with execution-time instrumentation**
  (`log_execution_time`) across the pipeline, useful for both debugging
  and the kind of performance transparency (e.g. the ~60s ffmpeg
  extraction cost documented during the explainability work) that aids
  reproducibility discussions.

### Major weaknesses

- **No automated test suite is evident** (no `pytest`/`unittest` files
  referenced anywhere in this review's research) — the "robustness
  validation" was a manual, one-time investigative exercise
  (`reports/backend_validation.md`) rather than a regression-preventing
  automated test suite that would catch a future re-introduction of any
  of the four fixed bugs. This is the single largest reproducibility/
  maintainability gap in the project.
- **No environment/dependency pinning evidence beyond `requirements.txt`
  with mostly unpinned versions** (e.g. `captum` added with no version
  pin beyond what was installed at development time, `torch>=2.3`-style
  ranges rather than exact pins) — for a research artifact intended to
  be reproducible, exact dependency pinning (or a lockfile) would be
  expected to guarantee the reported accuracy numbers are reproducible
  on a different machine/date.
- **The REST API has no authentication, authorization, or rate
  limiting**, acknowledged explicitly in `reports/api_documentation.md`
  as a known limitation rather than hidden — appropriate for a
  research prototype, but this must be stated clearly as **not
  production-ready** if the dissertation or any accompanying materials
  ever describe it as "deployed" or "production."
- **Single-worker, lock-serialized inference** means the API cannot
  scale to concurrent load — acceptable for a research demo, but a
  scalability limitation that should be explicit in any deployment
  narrative.
- **No CI/CD pipeline evidence** (no GitHub Actions/other CI config
  referenced in this review) to automatically run linting, tests, or
  the robustness scenarios on every change — meaning regressions could
  be reintroduced by future edits without being caught automatically.

### Minor comments

- Logging is consistent, but no evidence of structured/centralized log
  aggregation (e.g. for the API in a real deployment) — acceptable for
  a research prototype, worth flagging as future work only.
- The API's in-process `threading.Lock` approach to serializing
  inference is a reasonable stop-gap but would not survive a
  multi-process/multi-replica deployment (e.g. behind a load balancer)
  without an external coordination mechanism — again, appropriately
  scoped out rather than hidden, but worth a one-line explicit caveat in
  the API docs.
- The 12 synthetic robustness test videos plus additional real-footage
  regression runs are a reasonable but still fairly small robustness
  test matrix; broader fuzz-style testing (malformed containers, extreme
  aspect ratios, rotated video, variable frame rates within one file)
  is not evidenced.

### Questions likely to be asked during a viva

1. "You have a detailed manual robustness report but no automated test
   suite — how do you know the four bugs you fixed won't silently
   regress the next time someone touches this code?"
2. "Your `requirements.txt` doesn't fully pin versions — can you
   guarantee someone re-running this in a year gets the same 88.57%
   accuracy number, or could a dependency update change results?"
3. "The API has no authentication. If this were to be handed to another
   team to integrate into a real product, what's the very first thing
   that would need to change before it could be exposed publicly?"
4. "Your multi-person tracking fix was refined after real-world testing
   revealed the first design broke on mirror footage. What other
   real-world edge cases do you suspect you haven't tested yet?"
5. "Walk me through what happens, end-to-end, if two users hit `/predict`
   at the exact same moment — where exactly does the second request
   wait, and for how long?"

### Recommendation: **Minor Revision**

The engineering practices demonstrated (modular design, evidence-driven
robustness fixing, thin API wrapper over validated logic, explicit
security/scaling caveats) are strong for a final-year project. Adding
even a minimal automated regression test suite covering the four
previously-found bugs, and pinning dependency versions, would meaningfully
strengthen the reproducibility claims without requiring any change to the
existing experiments or results.

---

## Overall Editor Decision

**Is this publishable as a university final-year research project?**

**Yes, with revisions.** The Hip & Knee module demonstrates a coherent,
end-to-end applied research pipeline — from raw video through pose
estimation, biomechanical feature engineering, a trained classifier,
explainability, a deployable API, and a genuinely evidence-driven
robustness validation exercise. The engineering quality and the project's
own scientific self-documentation (explicitly separating IWF rules,
general biomechanics concepts, and this project's specific engineering
choices; refusing to invent citations; flagging the RFD-naming issue
itself) are considerably more rigorous and self-aware than is typical for
work at this level. This is a strong candidate for a final-year
dissertation submission.

**However, three issues should be addressed, or at minimum very
prominently and consistently caveated, before final submission:**

1. **The "Rate of Force Development" naming issue** (Reviewer 2's primary
   concern) should be resolved by either renaming the field consistently
   across the API/report/UI-facing surfaces (e.g. to something like
   "Angular Acceleration Ratio (RFD proxy)"), or by adding an explicit,
   impossible-to-miss disclaimer at every point this value is surfaced to
   an end user — not only in the supplementary scientific-validation
   reports, which a typical reader/examiner may not reach.
2. **The evaluation protocol's statistical weakness** (Reviewer 1's
   primary concern) — a single 35-sample test split with no
   cross-validation or confidence interval — should be explicitly
   labeled as a limitation in the dissertation's main results section
   (not only in a supplementary report), and, time permitting, a k-fold
   cross-validation or repeated-split analysis would substantially
   strengthen the accuracy claims.
3. **Absence of automated regression tests** (Reviewer 3's primary
   concern) is the most actionable, lowest-effort fix: converting even a
   subset of the 12 manually-run robustness scenarios into an automated
   test suite would materially improve the project's engineering
   maturity and protect the bug fixes already found from silent
   regression.

**What should NOT be changed**: the underlying experiments, the trained
model, the dataset, and the Rule A–D formulas themselves do not need to
be altered to address any of the above — all three recommended actions
are documentation, labeling, and testing-infrastructure improvements that
can be made without re-running or invalidating any existing result.

**Recommendation: Minor-to-Major Revision (documentation and testing
infrastructure only; no re-experimentation required).** The project is
publishable at final-year dissertation standard once the RFD-naming
clarity issue and the single-split evaluation caveat are addressed
prominently in the main dissertation text, and is strengthened further
(though not blocked) by adding automated regression tests for the
previously-identified robustness bugs.
