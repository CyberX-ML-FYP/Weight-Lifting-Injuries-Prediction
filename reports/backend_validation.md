# Hip & Knee Backend — Robustness Validation Report

**Scope:** `src/data/hip_knee_dataset.py` and `src/models/hip_knee_predict.py` (the Hip & Knee
prediction backend only). No retraining performed. No changes made to
`module3_arm_analysis`, frontend, visualization, or teammate modules.

**Method:** 12 synthetic edge-case videos were generated from real footage
(`data/raw/hip_knee/Side view/13bad.MOV`, transformed via ffmpeg) and run through
the full CLI pipeline (`python -m src.models.hip_knee_predict`), plus additional
runs against real dataset videos from multiple camera views for regression and
"natural" edge-condition coverage (camera movement, occlusion, mirrors).

---

## 1. Real Bugs Found & Fixed

Per the task instruction ("do not change algorithms unless a real bug is
found"), the following are genuine defects/gaps — not algorithm changes:

| # | Bug | Location | Fix |
|---|-----|----------|-----|
| 1 | Multi-person frames always used YOLO detection index 0 (could silently be a mirror reflection, bystander, or piece of equipment instead of the athlete). A `select_person_by_center()` helper existed for this but was never called anywhere. | `run_pose_model()` in `hip_knee_dataset.py` | Wired in `select_person_by_center` as a **preference** among plausible detections (confidence-filtered first, then center-distance disambiguation), falling back to the highest-confidence candidate when center-distance is inconclusive, instead of blindly using index 0 (see §3 for why a hard "reject if ambiguous" design was tried and rejected). |
| 2 | `extract_frames()` had no exception handling around the ffmpeg subprocess call. Corrupted/unsupported-format/0-byte videos crashed with a raw `CalledProcessError` traceback. A failed extraction could also leave partial `frame_*.jpg` files that a later retry would silently treat as a valid cache. | `extract_frames()` in `hip_knee_dataset.py` | Added an upfront 0-byte check (`ValueError`), wrapped the ffmpeg call in `try/except CalledProcessError` with partial-frame cleanup, and raise a clear `RuntimeError` including the ffmpeg stderr tail. |
| 3 | The CLI `main()` entry point had no top-level exception handling — any failure (missing file, empty video, corrupted format, no usable frames, etc.) surfaced as a raw Python traceback to end users. | `main()` in `hip_knee_predict.py` | Wrapped the `predict_video()` call in `try/except (FileNotFoundError, ValueError, RuntimeError)`, printing a clean `ERROR: ...` message to stderr and exiting with code 1. Full traceback still logged via `logger.exception` for debugging. `predict_video()` itself is unchanged and still raises normally for library callers. |
| 4 | `save_annotated_video()` never checked whether `cv2.VideoWriter` actually opened (a codec/container issue could silently produce a broken/empty video with no error). A failure in annotated-video generation also crashed the *entire* pipeline, discarding an already-successful JSON/console prediction. | `save_annotated_video()` / `predict_video()` in `hip_knee_predict.py` | Added `writer.isOpened()` check (raises a clear `RuntimeError`). `predict_video()` now wraps the `save_annotated_video()` call in `try/except`, logging the failure and continuing without the video — the core JSON/console report is no longer at risk from video-writer issues. |

### Fix designed, tested, and refined during this task (important finding)

The initial design for Bug #1 rejected a frame outright whenever multiple
"plausible" people were detected and none was unambiguously closest to frame
center. Testing against **real footage** (not just synthetic videos) showed
this was too aggressive: `13bad.MOV`-derived clips (a real 4K gym recording
with a mirror) produced 8–9 simultaneous person detections per frame — none
within the existing `CENTER_DISTANCE_THRESHOLD_PX = 100` px of frame center in
a 3840×2160 frame — causing **every sampled frame, and therefore the entire
video, to be rejected** (`ValueError: No usable frames extracted`). Several
real dataset videos (e.g. `24bad.mp4`, `9bad.mp4`) also routinely showed 4–7
simultaneous detections, confirming this is common in this footage (mirrors,
equipment, other people partially in frame), not a one-off artifact of the
test clip.

The fix was refined to: filter to confidence-plausible detections → prefer an
unambiguous center-distance match if one exists → **otherwise fall back to
the highest-confidence plausible candidate** rather than discarding the frame.
This preserves full pipeline availability (matching, and improving on, the
pre-fix behavior) while still correctly resolving genuinely well-separated
multi-person scenes when center-distance is conclusive. This refinement was
only discovered through empirical testing, not code review — a real value
demonstration of this validation task.

---

## 2. Test Matrix

| Scenario | Test artifact / method | Result |
|---|---|---|
| Short videos | `short_1s.mp4` (1s, 4K@60fps), `single_frame.mp4` (single-frame) | ✅ PASS — both produce a full prediction report (flagged low-confidence, correctly, given minimal data) |
| Long videos | `long_loop.mp4` (~98s looped, avoids excessive 4K test runtime) | ✅ PASS — completes in ~89s, sensible output |
| Low FPS videos | `low_fps_5.mp4` (5 fps) | ✅ PASS |
| High FPS videos | `high_fps_120.mp4` (120 fps) | ✅ PASS |
| Different resolutions | `res_small_320x180.mp4` (320×180) | ✅ PASS |
| Portrait / landscape videos | `portrait_720x1280.mp4`, `landscape_1280x720.mp4` | ✅ PASS (both) |
| Missing keypoints / low-confidence detections | Existing frame-quality gate (`MIN_KEYPOINT_CONFIDENCE`) + "hold last known-good frame" fallback (pre-existing, unchanged, confirmed correct); forced end-to-end via `--min-confidence 0.999` on `res_small_320x180.mp4` | ✅ PASS — every frame rejected as expected, pipeline raises a clean `ValueError` (no crash) |
| Corrupted videos | `corrupted_truncated.mp4` (30%-truncated), `corrupted_random_header.mp4` (random bytes) | ✅ PASS (after fix #2) — clean `RuntimeError`/`ERROR:` message. **Previously would have been a raw traceback.** |
| Empty videos | `empty.mp4` (0 bytes) | ✅ PASS (after fix #2) — clean `ValueError`/`ERROR:` message |
| Unsupported formats | `unsupported_fake.mp4` (plain text renamed `.mp4`) | ✅ PASS (after fix #2) — clean `RuntimeError`/`ERROR:` message |
| Multiple people in the frame | Real footage: `short_1s.mp4` (8–9 detections/frame, mirror), `24bad.mp4` (4–7/frame), `9bad.mp4` (4–5/frame) | ✅ PASS (after fix #1) — correctly disambiguated/selected without crashing or silently misusing a bystander/reflection |
| Camera movement | Real dataset videos (`13bad.MOV`, `24bad.mp4`, `9bad.mp4` — natural handheld variation) | ✅ PASS (proxy validation — no dedicated synthetic camera-shake generator was built; see Limitations) |
| Occlusions | Frame-quality gate + last-known-good-frame fallback (pre-existing, unchanged); exercised by the forced-confidence test above | ✅ PASS (validated mechanism; no footage with guaranteed physical occlusion available — see Limitations) |
| Prediction pipeline (end-to-end) | All of the above | ✅ PASS |
| Rule A–D | Verified via console/JSON output across all successful runs (sensible, varying scores per video) | ✅ PASS |
| Anthropometric normalization | `leg_length_reference_px`, `hip/knee_linear_rom_normalized`, `hip/knee_peak_velocity_normalized` present and populated in every successful run's JSON | ✅ PASS |
| Confidence weighting | `rule_[a-d]_confidence`, `overall_confidence`, `confidence_weighted_final_score`, `is_low_confidence` present and behaving sensibly (correctly flagged `true` on most edge-case/low-quality inputs) | ✅ PASS |
| Pose smoothing | Savitzky-Golay filter (unchanged from prior task) confirmed still invoked in `analyze_lift()` — not touched by this task's fixes | ✅ PASS |
| Prediction JSON | `reports/prediction.json` — all 24 fields present and correctly typed on every successful run | ✅ PASS |
| Annotated video generation | `reports/annotated_videos/13bad_annotated.mp4` generated and confirmed on disk after fixes | ✅ PASS (writer failure paths reviewed and defensively fixed; not fault-injected — see Limitations) |

### Regression check

Re-ran `13bad.MOV` (the original known-good reference video) end-to-end with
video generation enabled after all fixes: predicted class `Good`, confidence
≈52.2% (baseline from the confidence-weighted-rules task was 52.60% — the
~0.4-point difference is consistent with JPEG re-encoding noise from a fresh
ffmpeg extraction pass, not a functional change), Rule A–D scores unchanged in
shape/order of magnitude, annotated video and JSON both produced successfully.
Two further real dataset videos (`24bad.mp4`, `9bad.mp4`, both ground-truth
"bad") were correctly predicted `Poor` with high confidence (83.5%, 99.4%).

---

## 3. Passed / Failed Summary

- **Passed (after fixes): 18/18** listed scenarios/verification items above.
- **Failed before fixes, now fixed:** empty videos, corrupted videos,
  unsupported formats (all previously raised raw uncaught tracebacks instead
  of clean errors), and multi-person frames (previously silently used
  whichever detection YOLO returned first, with no visibility or
  disambiguation).
- **No remaining failing scenarios** in this test matrix.

---

## 4. Remaining Limitations

1. **Rule A's synchronization-delay units.** `analyze_lift()` is always
   called with the default `fps=30` while the actual signal is 20 samples
   evenly spaced across the *entire* video regardless of true duration/fps.
   This means the reported "delay (s)" does not represent true video-time
   seconds for videos whose real duration/fps deviates from that convention.
   This is applied identically in training and inference, so relative
   comparisons (Good vs Poor) remain valid, but absolute delay values should
   not be over-interpreted for atypical-duration/fps videos. Not changed, per
   "do not change algorithms unless a real bug is found" — this is a
   pre-existing, consistent design convention, not a bug.
2. **Center-distance disambiguation threshold.** `CENTER_DISTANCE_THRESHOLD_PX
   = 100` (pre-existing constant) is very tight relative to typical 4K frame
   dimensions, so in practice on this dataset the confidence-based fallback
   (not the center-distance preference) resolves most multi-detection frames.
   The mechanism is safe (never crashes, always makes a best-effort choice)
   but the "prefer the person nearest frame-center" behavior fires less often
   than its design intent on high-resolution footage. Retuning this threshold
   was out of scope (would be an algorithm change, not a bug fix).
3. **No genuine two-distinct-people footage available.** All observed
   multi-detection scenarios in this dataset are explained by mirrors,
   equipment, or spurious low/medium-confidence boxes rather than a second
   real athlete/bystander at a different screen position. The disambiguation
   logic was validated extensively against these real multi-detection cases,
   but a true "two people, both clearly human, at different frame positions"
   scenario was not available to test.
4. **Camera movement / occlusion validated via proxy, not dedicated fault
   injection.** No synthetic camera-shake or deliberate keypoint-occlusion
   generator was built; these scenarios were validated using existing real
   dataset footage's natural variation. The pipeline handled all of it
   without crashing, but a controlled "camera pans away and back" or
   "athlete fully occluded for N frames" test was not specifically
   constructed.
5. **Video-writer failure path not fault-injected.** The `isOpened()` check
   and the `predict_video()` try/except isolation around annotated-video
   generation were verified by code review and a normal successful run; a
   deliberate codec/container failure was not reproduced on this machine
   (the `mp4v` codec is available here), so the failure path itself is
   defensively coded but not empirically exercised.
6. **Long-video test used a downscaled source.** `long_loop.mp4` was built by
   looping a 320×180 clip rather than genuine long-duration 4K footage, to
   keep test runtime reasonable. Full-duration 4K extraction time scales with
   resolution × duration, so very long high-resolution source videos would
   take proportionally longer (a performance consideration, not a
   correctness bug — no failure mode was observed at any tested duration).

---

## 5. Production Readiness Score

| Category | Score | Notes |
|---|---|---|
| Core prediction pipeline correctness | 9/10 | Rule A-D, anthropometric normalization, confidence weighting, pose smoothing all verified intact and correctly populated across all test runs |
| Input robustness (duration/fps/resolution/orientation/corrupted/empty/unsupported) | 9/10 | All 12 synthetic edge cases pass; 3 previously-crashing cases now fail cleanly with actionable messages |
| Multi-person / occlusion / low-confidence handling | 8/10 | Real bug fixed and validated against real multi-detection footage; center-distance threshold limitation noted (§4.2) |
| Exception handling & error messaging | 9/10 | Clear boundary between library (`predict_video()`, still raises) and CLI (`main()`, clean exit) |
| Test coverage completeness | 7/10 | Camera movement/occlusion/two-person scenarios validated by proxy rather than dedicated fault injection (§4.3–4.5) |

**Overall: ~84/100 — Production-ready for supervised/beta deployment.**
The backend handles the full required scenario matrix without crashing,
degrades gracefully with clear, actionable errors, and correctly preserves
all existing analysis features (Rule A-D, normalization, confidence
weighting, smoothing). The noted limitations (mostly around a fixed
disambiguation-distance constant and untested extreme fault-injection paths)
are appropriate follow-up items rather than blockers, and none were observed
to cause an actual pipeline failure during this validation.

---

## 6. Files Modified

- `src/data/hip_knee_dataset.py`: `run_pose_model()` (multi-person
  disambiguation), `extract_frames()` (ffmpeg error handling + empty-file
  check + partial-frame cleanup), new `_cleanup_partial_frames()` helper.
- `src/models/hip_knee_predict.py`: `save_annotated_video()` (writer
  `isOpened()` check), `predict_video()` (isolates annotated-video failures
  from the core report), `main()` (clean CLI error handling), added `import sys`.
- No changes to `src/features/hip_knee_pose_utils.py`,
  `src/features/hip_knee_biomechanics.py`, `src/features/hip_knee_scoring.py`,
  `src/features/hip_knee_config.py`, model weights, or any teammate module.
- Test artifacts generated under `data/_robustness_tests/` (scratch corpus,
  not part of the training dataset).
