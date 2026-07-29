# Module 3 — Arm / Shoulder / Elbow Analysis

**Author:** Pasindu — Index 214027H  
**Faculty of Information Technology, University of Moratuwa**

## Overview

This module analyses upper-limb biomechanics during Clean & Jerk weightlifting
attempts. It processes video footage frame-by-frame using MediaPipe pose
estimation and outputs quantitative measurements that feed into the team's
master ML dataset.

## What It Detects

- Bilateral elbow angles (left + right)
- Bilateral shoulder angles
- Arm symmetry difference per frame
- Incomplete lockout (elbow < 160° during jerk overhead)
- Bilateral asymmetry (difference > 15°)

## How It Works

1. Reads the video frame by frame.
2. Uses MediaPipe BlazePose to extract 33 body landmarks per frame.
3. Computes joint angles using the vector dot-product formula:

   `θ = cos⁻¹( (BA · BC) / |BA||BC| )`

4. Compares measured angles against biomechanics thresholds.
5. Exports per-frame data to CSV for downstream ML training.

## Usage

```bash
pip install -r requirements.txt
python -m src.module3_arm_analysis.analyzer
```

Edit the `VIDEO_INPUT` and `CSV_OUTPUT` paths at the bottom of `analyzer.py`.

## Output CSV Schema

| Column | Description |
|---|---|
| frame | Frame index in source video |
| left_elbow_angle | Left elbow angle in degrees |
| right_elbow_angle | Right elbow angle in degrees |
| left_shoulder_angle | Left shoulder angle in degrees |
| right_shoulder_angle | Right shoulder angle in degrees |
| symmetry_diff | Absolute L-R elbow difference |
| asymmetry_flag | 1 if symmetry_diff > 15° |
| lockout_flag | 1 if either elbow < 160° |

## Tech Stack

- Python 3.10+
- OpenCV — video I/O
- MediaPipe — pose estimation
- NumPy — vector math
- pandas — CSV management