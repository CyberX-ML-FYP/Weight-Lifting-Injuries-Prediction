# Weight-Lifting-Injuries-Prediction

**Multi-View Mathematical Analysis and AI-Based Performance & Risk Assessment System for Clean & Jerk Weightlifting**

Team CyberX · Faculty of Information Technology · University of Moratuwa · 2026

---

## 1. Overview

This project analyses Olympic weightlifting (the **Clean & Jerk**) from ordinary video recordings. It replaces expensive marker-based motion capture with a computer-vision pipeline that any coach can run on a laptop.

The system:

1. Records each lift attempt from **three camera views** (side, front, 45° diagonal).
2. Processes every video **frame by frame** with **MediaPipe BlazePose**, extracting 33 body landmarks per frame.
3. Runs **four independent analysis modules in parallel** on the same landmark stream — trunk/spine, hip/knee, arm/shoulder/elbow, and bar path.
4. Exports one **CSV per module per video**, all sharing the same frame index.
5. Merges the four CSVs into a single **master dataset**, joined with expert coach labels.
6. Trains **supervised machine learning models** (Random Forest, XGBoost, Attention-LSTM) to classify lift quality, produce a performance score, and estimate injury risk.

The key design principle: **each stage produces a well-defined, independently verifiable artifact** — raw video → landmark data → module CSV → master dataset → prediction report. Any stage can be tested on its own.

---

## 2. Team

| Index Number | Name | Module Responsibility |
|---|---|---|
| 214147B | Perera A.K.A.K.K. | Module 1 — Trunk and Spine Analysis |
| 214188B | Hemal Savindu | Module 2 — Hip and Knee Analysis |
| 214027H | Bandara H.G.P.M. | Module 3 — Arm, Shoulder and Elbow Analysis |
| 214189E | Senarathna G.G.P.C. | Module 4 — Bar Path, Multi-View Sync, ML Pipeline |

**IT Supervisor:** Dr. Lochandaka Ranathunga, Department of Information Technology

---

## 3. System Pipeline

```
                    Multi-view videos  (.mp4 / .mov,  >= 30 fps)
                                  |
                  Stage 1 - Frame extraction  (cv2.VideoCapture)
                                  |
                  Stage 2 - Pose estimation   (MediaPipe BlazePose)
                            33 landmarks per frame
                                  |
        +-----------------+-------+-------+-----------------+
        |                 |               |                 |
    Module 1          Module 2        Module 3          Module 4
  Trunk / Spine     Hip / Knee     Arm / Shoulder      Bar Path
        |                 |               |                 |
  trunk_angles.csv  knee_angles.csv  arm_analysis.csv  bar_path.csv
        |                 |               |                 |
        +-----------------+-------+-------+-----------------+
                                  |
              Stage 5 - Multi-view sync + merge on frame index
                        + expert labels + normalisation
                                  |
                          master_dataset.csv
                                  |
              Stage 6 - ML training  (RF / XGBoost / Attention-LSTM)
                                  |
    Lift quality  |  Performance score  |  Injury risk  |  Feedback
```

Every frame is treated as an **independent image processing event**. A 10-second video at 30 fps is 300 individual images, each producing one row in every module CSV.

---

## 4. Folder Structure

The repository follows the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) template.

```
Weight-Lifting-Injuries-Prediction/
├── .gitignore
├── README.md
├── requirements.txt
├── environment.yml
│
├── data/
│   ├── raw/                        # never edited, never committed
│   │   └── videos/
│   │       ├── side/               # 90° lateral camera
│   │       ├── front/              # facing the lifter
│   │       └── angle45/            # 45° diagonal camera
│   ├── interim/                    # per-frame module CSVs
│   │   ├── module1/
│   │   ├── module2/
│   │   ├── module3/
│   │   └── module4/
│   ├── processed/                  # master_dataset.csv, train/test splits
│   └── external/                   # expert label sheets, reference tables
│
├── models/                         # trained models (.pkl, .h5)
│
├── notebooks/                      # exploratory analysis
│
├── references/
│   └── folder_structure.txt
│
├── reports/
│   └── figures/                    # generated graphs (per module)
│
├── docs/                           # interim report, presentations
│
└── src/
    ├── __init__.py
    ├── config.py                   # ALL shared constants and paths
    │
    ├── data/
    │   ├── __init__.py
    │   ├── pose_extractor.py       # shared MediaPipe wrapper
    │   └── make_dataset.py         # batch runner: videos -> module CSVs
    │
    ├── features/
    │   ├── __init__.py
    │   ├── angles.py               # shared calculate_angle() utility
    │   ├── module1_trunk.py        # Trunk / Spine        (214147B)
    │   ├── module2_hip_knee.py     # Hip / Knee           (214188B)
    │   ├── module3_arm.py          # Arm / Shoulder       (214027H)
    │   ├── module4_bar_path.py     # Bar Path + sync      (214189E)
    │   └── build_features.py       # merge -> master_dataset.csv
    │
    ├── models/
    │   ├── __init__.py
    │   ├── train_model.py          # RF / XGBoost / Attention-LSTM
    │   └── predict_model.py        # new video -> prediction
    │
    └── visualization/
        ├── __init__.py
        ├── plot_settings.py        # shared chart theme
        └── visualize.py            # per-module research graphs
```

> **Important:** `data/raw/videos/` and `models/` are excluded by `.gitignore`. Video files and trained models are shared through the team drive, not through Git.

---

## 5. Setup

### 5.1 Requirements

- Python **3.11** (3.10 also works)
- Git
- ~4 GB free disk space for videos and intermediate CSVs

### 5.2 Clone and install

```bash
git clone https://github.com/CyberX-ML-FYP/Weight-Lifting-Injuries-Prediction.git
cd Weight-Lifting-Injuries-Prediction
```

**Option A — venv (recommended)**

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

**Option B — conda**

```bash
conda env create -f environment.yml
conda activate weightlifting
```

### 5.3 Verify

```bash
python -c "import cv2, mediapipe, numpy, pandas, sklearn; print('Environment OK')"
```

### 5.4 Pose model file

The MediaPipe pose model (`pose_landmarker_full.task`, ~9 MB) is downloaded automatically on first run into `models/`. It is git-ignored. To download it manually:

```
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

---

## 6. Data Preparation

### 6.1 Recording protocol

| Camera view | Position | Frame rate | Captures | Used by |
|---|---|---|---|---|
| Side | 90° lateral | ≥ 30 fps | Sagittal joint angles, bar path | All 4 modules |
| Front | Facing the lifter | ≥ 30 fps | Bilateral symmetry, alignment | M1, M2, M3 |
| 45° diagonal | Off-axis | ≥ 30 fps | Depth disambiguation | M4 multi-view |

All three cameras must record the **same attempt at the same time**. Ask the lifter to clap once before the lift — the clap gives every view a common temporal anchor for synchronisation.

### 6.2 File naming — this is a hard rule

```
data/raw/videos/side/1good.mp4
data/raw/videos/front/1good.mp4      <- SAME attempt, different camera
data/raw/videos/angle45/1good.mp4    <- SAME attempt, different camera
```

Rules:

- `<attempt_number><label>.<ext>` — for example `1good.mp4`, `7bad.mp4`
- The label is parsed from the filename: `good` → `0`, `bad` → `1`
- The **same attempt number must refer to the same lift in all three view folders**. If side `3good.mp4` and front `3good.mp4` are different lifts, the merge will silently produce corrupted training data.
- No spaces, no non-ASCII characters in filenames.

### 6.3 Expert labels

Quality labels are recorded by a coach in `data/external/labels.csv`:

```csv
attempt_id,quality_label,skill_level,notes
1,good,intermediate,clean lockout
2,bad,beginner,incomplete lockout on left arm
```

---

## 7. How to Run

Always run modules with `python -m` from the repository root, so package imports resolve correctly.

### Step 1 — Extract features from all videos

```bash
python -m src.data.make_dataset --view side
python -m src.data.make_dataset --view front
```

Reads every video in `data/raw/videos/<view>/`, runs pose estimation once per frame, passes the landmarks to all four modules, and writes per-frame CSVs into `data/interim/module*/`.

Useful flags:

| Flag | Effect |
|---|---|
| `--view side\|front\|angle45` | Which camera folder to process |
| `--module 1\|2\|3\|4` | Run one module only |
| `--show` | Display the annotated video while processing (slow) |
| `--save-video` | Write an annotated `.mp4` to `reports/figures/` |
| `--force` | Re-process videos that already have a CSV |

Processing is skipped for any video that already has an output CSV, so the command is safe to re-run.

### Step 2 — Build the master dataset

```bash
python -m src.features.build_features
```

Synchronises the views, merges the four module CSVs on `frame_index`, joins the expert labels, applies min-max normalisation, and writes `data/processed/master_dataset.csv`.

### Step 3 — Train the models

```bash
python -m src.models.train_model --model rf
python -m src.models.train_model --model xgb
python -m src.models.train_model --model lstm
```

Saves trained models to `models/` and evaluation charts to `reports/figures/`.

### Step 4 — Predict on a new video

```bash
python -m src.models.predict_model --video data/raw/videos/side/test_lift.mp4
```

Prints the predicted lift quality, performance score, injury risk level, and the features that most influenced the prediction.

### Step 5 — Generate report graphs

```bash
python -m src.visualization.visualize --module 3
```

---

## 8. Module Reference

All modules share one angle utility (`src/features/angles.py`):

```python
def calculate_angle(A, B, C):
    """Angle at vertex B, in degrees, using the vector dot product."""
    BA = np.array(A) - np.array(B)
    BC = np.array(C) - np.array(B)
    cos_t = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0)))
```

θ = cos⁻¹ ( (BA · BC) / (|BA| × |BC|) )

### 8.1 MediaPipe landmark indices

| Landmark | Index | Landmark | Index |
|---|---|---|---|
| Left shoulder | 11 | Right shoulder | 12 |
| Left elbow | 13 | Right elbow | 14 |
| Left wrist | 15 | Right wrist | 16 |
| Left hip | 23 | Right hip | 24 |
| Left knee | 25 | Right knee | 26 |
| Left ankle | 27 | Right ankle | 28 |

### 8.2 Module 1 — Trunk and Spine Analysis

**Owner:** 214147B · **File:** `src/features/module1_trunk.py` · **Output:** `trunk_angles.csv`

Landmarks 11, 12, 23, 24. The trunk vector runs from the hip midpoint to the shoulder midpoint; the spine angle is measured against a vertical reference.

| Reference range | Phase |
|---|---|
| 40°–55° forward lean | First pull |
| −10° to +10° | Clean catch |
| 0°–10° | Jerk dip |
| < 5° from vertical | Jerk drive / overhead |

Shoulder asymmetry flag fires when the left/right shoulder y-coordinates differ by more than **0.03 normalised units**.

**Columns:** `frame_index, timestamp_ms, shoulder_mid_x, shoulder_mid_y, hip_mid_x, hip_mid_y, spine_angle, lean_deviation, postural_deviation, shoulder_asymmetry_flag, lift_phase, low_visibility`

### 8.3 Module 2 — Hip and Knee Analysis

**Owner:** 214188B · **File:** `src/features/module2_hip_knee.py` · **Output:** `knee_angles.csv`

Knee angle from the hip–knee–ankle triplet; hip angle from the shoulder–hip–knee triplet, both sides independently.

- Angular velocity: `w(t) = (theta(t) - theta(t-1)) x FPS`, smoothed
- Range of motion: `ROM = theta_max - theta_min`
- Symmetry: `|theta_left - theta_right|`; values consistently above **10°** indicate bilateral imbalance
- Dip depth: minimum knee angle inside the jerk dip window
- Module score: `0.25 x ROM + 0.20 x Symmetry + 0.20 x Depth + 0.20 x Extension + 0.15 x Velocity`

**Columns:** `frame_index, timestamp_ms, L_knee_angle, R_knee_angle, L_hip_angle, R_hip_angle, knee_symmetry_diff, hip_symmetry_diff, L_knee_velocity, R_knee_velocity, lift_phase, dip_depth_flag, low_visibility, performance_score_m2`

### 8.4 Module 3 — Arm, Shoulder and Elbow Analysis

**Owner:** 214027H · **File:** `src/features/module3_arm.py` · **Output:** `arm_analysis.csv`

- Elbow angle: shoulder → **elbow** → wrist
- Shoulder angle: hip → **shoulder** → elbow
- Symmetry difference: `|theta_left - theta_right|`
- **Asymmetry flag:** difference > **15°**
- **Lockout flag:** either elbow < **160°** during the overhead reception phase (International Weightlifting Federation extension rule)
- Stage label assigned per frame from the wrist-height trajectory

**Columns:** `frame_index, timestamp_ms, L_elbow_angle, R_elbow_angle, L_shoulder_angle, R_shoulder_angle, elbow_symmetry_diff, shoulder_symmetry_diff, asym_flag, lockout_flag, jerk_stability_score, lift_stage, low_visibility`

> **View note:** symmetry must be read from the **front** view. In the side view the far arm is occluded by the body, so MediaPipe estimates its position and the symmetry difference is unreliable (often 30°+ even on clean lifts). Elbow extension and lockout are read from the **side** view, where the bend is clearly visible. Frames with `low_visibility = 1` should be filtered before aggregation.

### 8.5 Module 4 — Bar Path and Multi-View Synchronisation

**Owner:** 214189E · **File:** `src/features/module4_bar_path.py` · **Output:** `bar_path.csv`

Bar position is tracked as the wrist midpoint (landmarks 15, 16); direct barbell detection is planned.

- Deviation: `Bar_dev = sqrt( (1/N) * sum( (x_bar(i) - x_bar_mean)^2 ) )`
- Quality bands: **< 0.03** excellent · **0.03–0.07** moderate · **> 0.07** inefficient
- Sync anchor: the frame of **maximum upward wrist velocity** during the second pull, located independently in each view; the frame offset between anchors aligns the views

**Columns:** `frame_index, timestamp_ms, bar_x, bar_y, displacement, bar_deviation, bar_velocity, sync_offset, normalized_flag`

---

## 9. Master Dataset

`src/features/build_features.py` produces `data/processed/master_dataset.csv`:

1. Detect the sync anchor in each view and compute the frame offset.
2. Merge the four module CSVs on `frame_index` (pandas outer join).
3. Join expert labels from `data/external/labels.csv` by attempt id.
4. Apply min-max normalisation column by column: `x_norm = (x - x_min) / (x_max - x_min)`.

| Module | Key columns contributed |
|---|---|
| M1 | `spine_angle, lean_deviation, asymmetry_flag, lift_phase` |
| M2 | `L_knee, R_knee, L_hip, R_hip, symmetry, velocity, dip_depth` |
| M3 | `L_elbow, R_elbow, L_shoulder, R_shoulder, asym_flag, lockout_flag` |
| M4 | `bar_x, bar_y, displacement, deviation, normalized` |

---

## 10. Machine Learning Pipeline

**Partitioning.** Split at **lift-attempt level, never at frame level** — all frames from one attempt must stay in the same partition, otherwise near-identical frames leak between train and test and accuracy is inflated. Ratios are 70 / 15 / 15, stratified by quality label and skill level.

| Model | Configuration | Explainability |
|---|---|---|
| Random Forest | 200 trees, `max_depth=15`, `min_samples_leaf=5`, 5-fold stratified CV | `feature_importances_` — top 5 predictive features |
| XGBoost | Gradient boosting, native missing-value handling | SHAP values per prediction |
| Attention-LSTM | `T = 150` frames (~5 s at 30 fps); LSTM 128 → 64, dropout 0.3; attention layer; softmax head (3 classes) + sigmoid regression head; Adam `lr = 1e-4`; loss weights 0.8 classification / 0.2 regression | Attention weights show which lift phases drove the decision |

**Outputs**

| Output | Format | Produced by |
|---|---|---|
| Lift quality | Good / Moderate / Bad | Classifier |
| Performance score | 0–100 % | Regression head |
| Injury risk level | Low / Moderate / High | Rule + ML hybrid |
| Technique feedback | Text statements | Flag aggregation |
| Analysis graphs | PNG | All modules |

---

## 11. Configuration

All tunable values live in `src/config.py`. Do not hard-code paths or thresholds anywhere else.

```python
# Video processing
FRAME_SKIP = 3  # process every Nth frame
MIN_DETECTION_CONF = 0.5
MIN_TRACKING_CONF = 0.5
NUM_POSES = 5  # detect all people, then select the lifter

# Module 1
SHOULDER_ASYM_THRESHOLD = 0.03  # normalised units

# Module 2
KNEE_SYMMETRY_THRESHOLD = 10.0  # degrees

# Module 3
ARM_SYMMETRY_THRESHOLD = 15.0  # degrees
LOCKOUT_THRESHOLD = 160.0  # degrees
RACK_THRESHOLD = 100.0  # degrees

# Module 4
BAR_DEV_EXCELLENT = 0.03
BAR_DEV_MODERATE = 0.07
```

> `FRAME_SKIP`, `NUM_POSES` and the frame-index convention **must be identical across all four modules**, otherwise the CSVs cannot be merged.

**Selecting the lifter.** Training videos contain coaches and spectators. The pipeline detects up to 5 poses per frame and keeps the one whose torso centre (mean x of landmarks 11, 12, 23, 24) is closest to the horizontal centre of the frame, since the lifter is always centred in the recording protocol.

---

## 12. Team Git Workflow

Each member works on their own branch and opens a pull request into `main`.

| Member | Branch |
|---|---|
| 214147B | `member1-trunk-analysis` |
| 214188B | `member2-hip-knee-analysis` |
| 214027H | `member3-arm-analysis` |
| 214189E | `member4-barpath-ml` |

```bash
git checkout main
git pull origin main
git checkout -b member3-arm-analysis

# ... make changes ...

git add .
git commit -m "Add front-view symmetry filtering to Module 3"
git push origin member3-arm-analysis
```

Then open a pull request on GitHub.

**Never commit:** video files, per-frame CSVs, trained models, `.venv/`, `__pycache__/`, `pose_landmarker_full.task`.

**Do commit:** source code, `master_dataset.csv` (if small enough), report figures, documentation.

---

## 13. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | Running the file directly | Run from the repo root with `python -m src.features.module3_arm` |
| Skeleton drawn on a spectator | `num_poses=1` picks the strongest detection | Set `NUM_POSES = 5` and select the pose closest to frame centre |
| Video looks squeezed or stretched | Display resize ignores the source aspect ratio | Scale to a fixed width and compute height as `int(h * new_w / w)` |
| Video appears rotated | Phone rotation metadata ignored by OpenCV | Rotate with `cv2.rotate()` only when `width > height` and the source is portrait |
| Symmetry difference always > 15° | Far arm occluded in the side view | Take symmetry from the **front** view; drop frames where `low_visibility = 1` |
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | Version mismatch | `pip install mediapipe==0.10.9` |
| CSVs will not merge | Different `FRAME_SKIP` between modules | Align `src/config.py` across the team and re-run `make_dataset` |
| Suspiciously high accuracy | Frame-level train/test split | Split at attempt level in `train_model.py` |
| Processing very slow | Live display enabled | Drop `--show`; raise `FRAME_SKIP` |

---

## 14. Project Status

| Component | Status | Progress | Outstanding work |
|---|---|---|---|
| Frame extraction + pose estimation | Complete | 100 % | — |
| Module 1 — Trunk / Spine | Partial | 70 % | Threshold validation |
| Module 2 — Hip / Knee | Partial | 70 % | Front-view valgus integration |
| Module 3 — Arm / Shoulder | Partial | 70 % | Stage labelling refinement |
| Module 4 — Bar Path | Partial | 70 % | Multi-view sync completion |
| Master dataset merge | Partial | 60 % | Awaiting M4 sync |
| ML pipeline | Skeleton | 40 % | Awaiting full dataset |
| Final system integration | Pending | 20 % | After ML training |

### Known limitations

- **Front-rack occlusion.** Wrist and elbow landmarks are often hidden behind the bar during the catch. Gaps under ~10 frames are linearly interpolated; longer gaps need multi-view landmark fusion.
- **Wrist proxy for the bar.** The wrist midpoint is offset from the true bar centre during the pull. Direct HSV/contour barbell detection is planned.
- **2D analysis.** MediaPipe depth estimates are less reliable than x/y, so computations are currently sagittal-plane 2D from the side view.

### Future work

- Lumbar curvature estimation for finer lower-back risk indicators (M1)
- Knee valgus/varus quantification from the front view for ACL risk (M2)
- Multi-trajectory stage labelling via HMM or finite state machine (M3)
- Full multi-view synchronisation, direct bar detection, complete ML training with SHAP analysis and attention ablation (M4)

---

## 15. Technology Stack

| Category | Technology |
|---|---|
| Language | Python 3.11 |
| Computer vision | OpenCV (`cv2`), MediaPipe BlazePose |
| Numerical / data | NumPy, pandas |
| Visualisation | Matplotlib |
| Classical ML | scikit-learn, XGBoost |
| Deep learning | TensorFlow / PyTorch (Attention-LSTM) |
| Version control | Git, GitHub |

---

## 16. Academic Context

Submitted for the Level 4 Comprehensive Group Project, Faculty of Information Technology, University of Moratuwa (2026). The full interim report is in `docs/`.

This project is academic work. Video recordings of participants are collected with informed consent and are not distributed publicly.