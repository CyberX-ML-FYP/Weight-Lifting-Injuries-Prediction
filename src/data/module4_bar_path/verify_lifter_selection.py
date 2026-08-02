"""
Module 4 — Lifter selection verification tool.

Standalone diagnostic script (not part of the pipeline). Processes front-view
videos with MediaPipe Tasks PoseLandmarker (num_poses=5), scores every
detected pose each frame to pick "the lifter", and renders visual proof that
the same person stays selected across frames rather than the selection
jumping between people.

Each video's output goes to its own subfolder named after the video
(e.g. reports/figures/module4/verify/2good/), with a matching montage at
reports/figures/module4/verify/2good_montage.jpg.

Run (all front videos):
    python -m src.data.module4_bar_path.verify_lifter_selection

Run (a single video):
    python -m src.data.module4_bar_path.verify_lifter_selection 2good.mp4
"""
import os
import sys
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ── Paths ─────────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))

FRONT_VIDEO_DIR = os.path.join(BASE_DIR, "data", "raw", "videos", "front")
VIDEO_EXTENSIONS = (".mp4", ".mov")

# Shared with module3_arm_analysis so the model is only downloaded once.
MODEL_PATH = os.path.join(
    BASE_DIR, "src", "data", "module3_arm_analysis", "pose_landmarker_full.task"
)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

VERIFY_DIR = os.path.join(BASE_DIR, "reports", "figures", "module4", "verify")

# ── Landmark indices ──────────────────────────────────────────────────────────
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
TORSO_INDICES = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]

POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

# ── Tunables ───────────────────────────────────────────────────────────────────
FRAME_SKIP = 3
NUM_POSES = 5
MIN_DETECTION_CONF = 0.5
MIN_TRACKING_CONF = 0.5
VISIBILITY_THRESHOLD = 0.5
OUTPUT_WIDTH = 1280
JUMP_THRESHOLD_FRACTION = 0.25  # 25% of frame width
MONTAGE_STRIDE = 20

WEIGHT_CENTRENESS = 0.3
WEIGHT_CLOSENESS = 0.2
WEIGHT_LOWNESS = 0.2
WEIGHT_CONTINUITY = 0.3
CONTINUITY_RADIUS_FRACTION = 0.25  # frame-width fraction at which continuity score hits 0


def download_model_if_needed():
    """Download the MediaPipe pose landmarker model if not already present."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading pose landmarker model...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")


def _torso_visible(landmarks):
    return all(landmarks[idx].visibility > VISIBILITY_THRESHOLD for idx in TORSO_INDICES)


def _torso_center(landmarks):
    xs = [landmarks[idx].x for idx in TORSO_INDICES]
    ys = [landmarks[idx].y for idx in TORSO_INDICES]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _bbox_height(landmarks):
    ys = [lm.y for lm in landmarks]
    return max(ys) - min(ys)


def _hip_y(landmarks):
    return (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2.0


def select_lifter(pose_list, w, h, prev_center_px=None):
    """
    Score each detected pose 0-1 and return (best_landmarks, best_score,
    candidate_info) where candidate_info is a list of
    (landmarks, is_selected, score_or_None) for every pose that passed the
    visibility gate, so callers can draw all detected people.

    Signals (normalised across the detected poses in this frame):
      - CENTRENESS (0.3): torso centre x close to 0.5
      - CLOSENESS  (0.2): pose bounding-box height (bigger = closer)
      - LOWNESS    (0.2): hip y (lower in frame = better)
      - CONTINUITY (0.3): torso centre close to the previous frame's
        selected torso centre (pixel space). Without this, a bystander
        standing tall at the frame edge can outscore a lifter who is
        crouched low at the bar, since a bent-over pose has a smaller
        bounding-box height and briefly loses on CLOSENESS alone.
        Skipped on the first frame (prev_center_px is None).
    """
    candidates = []
    for landmarks in pose_list:
        if landmarks and _torso_visible(landmarks):
            candidates.append(landmarks)

    if not candidates:
        return None, 0.0, [], 0

    centre_x = np.array([_torso_center(lm)[0] for lm in candidates])
    bbox_h = np.array([_bbox_height(lm) for lm in candidates])
    hip_y = np.array([_hip_y(lm) for lm in candidates])

    # Centreness: distance from 0.5, inverted and normalised so closer to
    # centre -> higher score. Guard against zero-range with a fallback of 1.0.
    dist_from_centre = np.abs(centre_x - 0.5)
    max_dist = dist_from_centre.max()
    centreness = 1.0 - (dist_from_centre / max_dist) if max_dist > 1e-9 else np.ones_like(dist_from_centre)

    def _normalise(values):
        lo, hi = values.min(), values.max()
        if hi - lo < 1e-9:
            return np.ones_like(values)
        return (values - lo) / (hi - lo)

    closeness = _normalise(bbox_h)
    lowness = _normalise(hip_y)

    weight_centreness, weight_closeness, weight_lowness = (
        WEIGHT_CENTRENESS, WEIGHT_CLOSENESS, WEIGHT_LOWNESS
    )

    if prev_center_px is None:
        continuity = np.zeros(len(candidates))
        weight_continuity = 0.0
        # Redistribute the unused continuity weight across the other three
        # signals (proportionally) so the first frame's scores still sum to 1.
        scale = 1.0 / (weight_centreness + weight_closeness + weight_lowness)
        weight_centreness *= scale
        weight_closeness *= scale
        weight_lowness *= scale
    else:
        centre_px = np.array([
            (_torso_center(lm)[0] * w, _torso_center(lm)[1] * h) for lm in candidates
        ])
        dist_px = np.linalg.norm(centre_px - np.array(prev_center_px), axis=1)
        radius = CONTINUITY_RADIUS_FRACTION * w
        continuity = np.clip(1.0 - dist_px / radius, 0.0, 1.0)
        weight_continuity = WEIGHT_CONTINUITY

    num_candidates = len(candidates)
    scores = (
        weight_centreness * centreness
        + weight_closeness * closeness
        + weight_lowness * lowness
        + weight_continuity * continuity
    )

    best_idx = int(np.argmax(scores))
    best_landmarks = candidates[best_idx]
    best_score = float(scores[best_idx])

    candidate_info = [
        (lm, i == best_idx, float(scores[i]))
        for i, lm in enumerate(candidates)
    ]
    return best_landmarks, best_score, candidate_info, num_candidates


def _draw_skeleton(frame, landmarks, w, h, color, thickness):
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in POSE_CONNECTIONS:
        if a < len(points) and b < len(points):
            cv2.line(frame, points[a], points[b], color, thickness)
    for x, y in points:
        cv2.circle(frame, (x, y), max(2, thickness), color, -1)


def _draw_bbox(frame, landmarks, w, h, color, thickness):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def _resize_landscape(frame, target_width):
    h, w = frame.shape[:2]
    scale = target_width / w
    return cv2.resize(frame, (target_width, int(h * scale)))


def _build_landmarker():
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=NUM_POSES,
        min_pose_detection_confidence=MIN_DETECTION_CONF,
        min_pose_presence_confidence=MIN_DETECTION_CONF,
        min_tracking_confidence=MIN_TRACKING_CONF,
    )
    return vision.PoseLandmarker.create_from_options(options)


def process_video(video_path):
    """Process one video, writing frames + montage to their own subfolder
    under VERIFY_DIR named after the video's stem. Returns
    (processed_count, jump_frames).

    Builds its own PoseLandmarker so each video's VIDEO-mode timestamp
    sequence starts fresh -- a landmarker reused across videos throws
    "Input timestamp must be monotonically increasing" once the next
    video's timestamps restart below where the previous one left off.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    stem = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(VERIFY_DIR, stem)
    montage_path = os.path.join(VERIFY_DIR, f"{stem}_montage.jpg")
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    frame_idx = 0
    processed_count = 0
    jump_frames = []
    prev_center = None
    montage_frames = []

    try:
        with _build_landmarker() as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % FRAME_SKIP != 0:
                    frame_idx += 1
                    continue

                orig_h, orig_w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((frame_idx / fps) * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                best_landmarks, best_score, candidates, num_candidates = select_lifter(
                    result.pose_landmarks, orig_w, orig_h, prev_center_px=prev_center
                )

                selection_jump = 0
                if best_landmarks is not None:
                    cx, cy = _torso_center(best_landmarks)
                    center_px = (cx * orig_w, cy * orig_h)
                    if prev_center is not None:
                        dx = abs(center_px[0] - prev_center[0])
                        if dx > JUMP_THRESHOLD_FRACTION * orig_w:
                            selection_jump = 1
                            jump_frames.append(frame_idx)
                    # Only anchor continuity from a contested frame (>=2
                    # candidates) -- a lone detection could be a bystander
                    # caught while the real lifter is briefly undetected.
                    if num_candidates >= 2:
                        prev_center = center_px

                # ── Draw ──────────────────────────────────────────────────
                for landmarks, is_selected, score in candidates:
                    if is_selected:
                        continue
                    _draw_bbox(frame, landmarks, orig_w, orig_h, (160, 160, 160), 1)

                if best_landmarks is not None:
                    _draw_bbox(frame, best_landmarks, orig_w, orig_h, (0, 220, 0), 4)
                    _draw_skeleton(frame, best_landmarks, orig_w, orig_h, (0, 220, 0), 3)

                frame = _resize_landscape(frame, OUTPUT_WIDTH)

                text = f"frame {frame_idx}  score={best_score:.2f}"
                cv2.putText(frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (255, 255, 255), 2, cv2.LINE_AA)
                if selection_jump:
                    cv2.putText(frame, "JUMP!", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                                (0, 0, 255), 3, cv2.LINE_AA)

                out_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
                cv2.imwrite(out_path, frame)

                if processed_count % MONTAGE_STRIDE == 0:
                    montage_frames.append(frame)

                processed_count += 1
                frame_idx += 1
    finally:
        cap.release()

    if montage_frames:
        _build_montage(montage_frames, montage_path)

    print(f"[{stem}] frames processed: {processed_count}")
    print(f"[{stem}] frames with selection_jump=1: {len(jump_frames)}")
    print(f"[{stem}] jump frame numbers: {jump_frames}")

    return processed_count, jump_frames


def main():
    download_model_if_needed()
    os.makedirs(VERIFY_DIR, exist_ok=True)

    if len(sys.argv) > 1:
        video_paths = [os.path.join(FRONT_VIDEO_DIR, sys.argv[1])]
    else:
        video_paths = sorted(
            os.path.join(FRONT_VIDEO_DIR, f)
            for f in os.listdir(FRONT_VIDEO_DIR)
            if f.lower().endswith(VIDEO_EXTENSIONS)
        )

    summary = {}
    for video_path in video_paths:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        try:
            processed_count, jump_frames = process_video(video_path)
            summary[stem] = (processed_count, jump_frames)
        except Exception as exc:
            print(f"[{stem}] FAILED: {exc}")
            summary[stem] = (0, [])

    print("\n=== Summary ===")
    for stem, (processed_count, jump_frames) in summary.items():
        print(f"{stem}: {processed_count} frames, {len(jump_frames)} jumps {jump_frames}")


def _build_montage(frames, out_path, tile_width=320):
    tiles = []
    for f in frames:
        h, w = f.shape[:2]
        scale = tile_width / w
        tiles.append(cv2.resize(f, (tile_width, int(h * scale))))

    tile_h = tiles[0].shape[0]
    tiles = [t if t.shape[0] == tile_h else cv2.resize(t, (tile_width, tile_h)) for t in tiles]

    cols = min(6, len(tiles))
    rows = int(np.ceil(len(tiles) / cols))

    grid = np.zeros((rows * tile_h, cols * tile_width, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * tile_h:(r + 1) * tile_h, c * tile_width:(c + 1) * tile_width] = tile

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, grid)


if __name__ == "__main__":
    main()
