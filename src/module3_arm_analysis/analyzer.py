"""
Module 3 — Arm / Shoulder / Elbow Analysis
Analyzes upper-limb biomechanics in Clean & Jerk video footage.

Author: Pasindu (214027H)
Faculty of Information Technology, University of Moratuwa
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import urllib.request
import numpy as np
import pandas as pd
import os

from src.module3_arm_analysis.config import (
    MODEL_PATH, MODEL_URL,
    LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST,
    RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    SYMMETRY_THRESHOLD, LOCKOUT_THRESHOLD,
    FRAME_SKIP, MIN_DETECTION_CONF, MIN_TRACKING_CONF,
    ANNOTATED_VIDEO_DIR,
)

# Colors are BGR (OpenCV convention).
LEFT_COLOR  = (255, 255, 0)   # cyan
RIGHT_COLOR = (255, 0, 255)   # magenta
MIDLINE_COLOR = (200, 200, 200)  # light gray, for shoulder-shoulder / hip-hip context lines


def calculate_angle(A, B, C):
    """
    Calculate the angle at vertex B (in degrees) given three points A, B, C.
    Uses the vector dot product formula:
        θ = cos⁻¹( (BA · BC) / (|BA| · |BC|) )
    """
    A, B, C = np.array(A), np.array(B), np.array(C)
    BA = A - B
    BC = C - B
    cosine = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-6)
    return round(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))), 1)


def download_model_if_needed():
    """Download MediaPipe pose landmarker model if not already present."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading pose landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")


def draw_arm_landmarks(frame, landmarks):
    """
    Draw only the points/segments Module 3 actually measures: shoulders,
    elbows, wrists, hips. Left and right are color-coded separately so a
    left/right landmark swap is visible at a glance on playback -- unlike
    a full 33-point skeleton, this is meant for auditing tracking quality
    on the specific joints this module's angles/symmetry depend on.
    """
    h, w = frame.shape[:2]

    def pt(idx):
        lm = landmarks[idx]
        return (int(lm.x * w), int(lm.y * h))

    left_chain  = [LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST]
    right_chain = [RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST]

    for chain, color in ((left_chain, LEFT_COLOR), (right_chain, RIGHT_COLOR)):
        pts = [pt(idx) for idx in chain]
        for a, b in zip(pts, pts[1:]):
            cv2.line(frame, a, b, color, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, p, 5, color, -1, cv2.LINE_AA)
            cv2.circle(frame, p, 5, (255, 255, 255), 1, cv2.LINE_AA)

    # Context lines: shoulder-to-shoulder and hip-to-hip, relevant to the
    # symmetry measurement but not part of either arm's angle calculation.
    cv2.line(frame, pt(LEFT_SHOULDER), pt(RIGHT_SHOULDER), MIDLINE_COLOR, 1, cv2.LINE_AA)
    cv2.line(frame, pt(LEFT_HIP), pt(RIGHT_HIP), MIDLINE_COLOR, 1, cv2.LINE_AA)


class PersonTracker:
    """
    Selects which detected pose (there can be up to 5, since gym footage
    often has coaches/other lifters/bystanders/cameramen in frame) is the
    actual lifter.

    Confirmed necessary, not theoretical: on real footage (side_21bad.MOV)
    a pure "closest to frame center" heuristic tracked a person walking
    through the background, then the cameraman, before finally landing on
    the actual lifter only once they stood upright and centered late in
    the lift -- during the crouched/off-center setup and pull phase, a
    bystander was often more centered than the lifter.

    A first version of this class made continuity a hard lock (always
    follow whoever's nearest to the last-tracked position, if within
    range). Tested against the same video: it stopped the tracker
    flickering between different wrong bystanders, but it also got
    permanently stuck on the first wrong pick, since that bystander never
    moved far enough away to break the lock -- worse than the original
    heuristic, which at least self-corrected once the real lifter became
    clearly the most centered person later in the lift.

    This version makes continuity a soft tiebreaker instead: each frame,
    find the single most-centered candidate as the reference point. Keep
    following the previously-tracked person ONLY if they're still within
    STICKINESS_MARGIN of that reference -- close enough that switching
    would just be reacting to frame-to-frame jitter. If someone else is
    now clearly more centered (beyond the margin), switch to them. This
    keeps the old heuristic's ability to self-correct once the real
    lifter is obviously more centered, while damping the minor jitter
    that made the original version pick a *different* wrong bystander
    from one moment to the next.

    Per frame:
      1. Confidence filter: drop candidates whose mean torso-landmark
         visibility is too low (partial detections, distant bystanders).
      2. Find the most-centered candidate this frame (the reference).
      3. If we're tracking someone and they're still findable nearby and
         within STICKINESS_MARGIN of the reference's center-distance,
         keep following them.
      4. Otherwise, switch to the most-centered candidate.
    """

    TORSO_INDICES = (11, 12, 23, 24)  # left/right shoulder, left/right hip
    MIN_TRACK_CONFIDENCE = 0.5
    STICKINESS_MARGIN = 0.05    # normalized x-distance of slack before we abandon the current track
    MAX_MATCH_DISTANCE = 0.3    # normalized 2D distance beyond which we can't find the tracked person this frame

    def __init__(self):
        self.last_position = None  # (x, y) normalized torso center of the last-selected person

    def _torso_center_and_confidence(self, landmarks):
        points = [landmarks[idx] for idx in self.TORSO_INDICES if idx < len(landmarks)]
        if len(points) != len(self.TORSO_INDICES):
            return None
        x = sum(p.x for p in points) / len(points)
        y = sum(p.y for p in points) / len(points)
        confidence = sum(float(getattr(p, "visibility", 1.0)) for p in points) / len(points)
        return (x, y), confidence

    @staticmethod
    def _distance(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def select(self, pose_landmarks_list):
        if not pose_landmarks_list:
            return None

        candidates = []
        for landmarks in pose_landmarks_list:
            if not landmarks:
                continue
            info = self._torso_center_and_confidence(landmarks)
            if info is None:
                continue
            position, confidence = info
            if confidence < self.MIN_TRACK_CONFIDENCE:
                continue
            candidates.append((landmarks, position))

        if not candidates:
            # Nobody passed the confidence filter this frame -- leave
            # last_position as-is rather than losing the track on one bad frame.
            return None

        best_center = min(candidates, key=lambda c: abs(c[1][0] - 0.5))
        best_center_distance = abs(best_center[1][0] - 0.5)

        if self.last_position is not None:
            nearest = min(candidates, key=lambda c: self._distance(c[1], self.last_position))
            if self._distance(nearest[1], self.last_position) <= self.MAX_MATCH_DISTANCE:
                nearest_center_distance = abs(nearest[1][0] - 0.5)
                if nearest_center_distance <= best_center_distance + self.STICKINESS_MARGIN:
                    self.last_position = nearest[1]
                    return nearest[0]
                # Someone else is now clearly more centered -- re-anchor below.

        self.last_position = best_center[1]
        return best_center[0]


def analyze_video(video_path, output_csv_path, show_display=True,
                   save_annotated_video=False, annotated_video_path=None):
    """
    Process a Clean & Jerk video and produce a CSV of upper-limb features.

    Args:
        video_path:     Input video file (.mp4 / .mov).
        output_csv_path: Where to write the per-frame analysis CSV.
        show_display:   If True, opens a window showing real-time overlay.
        save_annotated_video: If True, saves the arm/shoulder/elbow overlay
            to an .mp4 file so it can be reviewed after the run, instead of
            only a transient live window. Off by default since batch runs
            process many videos and shouldn't pay this cost unless asked.
        annotated_video_path: Where to save it. Defaults to
            ANNOTATED_VIDEO_DIR/{video_stem}_module3_annotated.mp4.

    Returns:
        pandas.DataFrame of the extracted features.
    """
    download_model_if_needed()

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=5,
        min_pose_detection_confidence=MIN_DETECTION_CONF,
        min_pose_presence_confidence=MIN_DETECTION_CONF,
        min_tracking_confidence=MIN_TRACKING_CONF,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps       = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_idx = 0
    data      = []
    tracker   = PersonTracker()

    writer = None
    if save_annotated_video:
        if annotated_video_path is None:
            stem = os.path.splitext(os.path.basename(video_path))[0].strip()
            # Videos with the same filename can exist under different view
            # folders (e.g. "22good.mp4" under both angle/ and front/) --
            # include the parent folder name so defaults never collide.
            parent = os.path.basename(os.path.dirname(video_path)).strip() or "video"
            annotated_video_path = os.path.join(
                ANNOTATED_VIDEO_DIR, f"{parent}_{stem}_module3_annotated.mp4"
            )
        try:
            os.makedirs(os.path.dirname(annotated_video_path), exist_ok=True)
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # Write at fps/FRAME_SKIP since we only process every FRAME_SKIP-th
            # frame -- this keeps playback speed matching real time.
            writer = cv2.VideoWriter(
                annotated_video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                max(fps / FRAME_SKIP, 1), (width, height),
            )
            if not writer.isOpened():
                print(f"Warning: could not open video writer for {annotated_video_path} "
                      f"(missing/unsupported mp4v codec) -- continuing without saving annotated video.")
                writer = None
        except Exception as exc:
            print(f"Warning: failed to set up annotated video writer ({exc}) -- "
                  f"continuing without saving annotated video.")
            writer = None

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_SKIP != 0:
                frame_idx += 1
                continue

            h, w = frame.shape[:2]
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((frame_idx / fps) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            frame_idx += 1

            if result.pose_landmarks:
                landmark_list = tracker.select(result.pose_landmarks)
                if landmark_list is not None:
                    draw_arm_landmarks(frame, landmark_list)

                    def get_xy(idx):
                        lm = landmark_list[idx]
                        return [lm.x * w, lm.y * h]

                    def get_visibility(idx):
                        lm = landmark_list[idx]
                        return float(getattr(lm, "visibility", 1.0))

                    L_shoulder = get_xy(LEFT_SHOULDER)
                    L_elbow    = get_xy(LEFT_ELBOW)
                    L_wrist    = get_xy(LEFT_WRIST)
                    L_hip      = get_xy(LEFT_HIP)
                    R_shoulder = get_xy(RIGHT_SHOULDER)
                    R_elbow    = get_xy(RIGHT_ELBOW)
                    R_wrist    = get_xy(RIGHT_WRIST)
                    R_hip      = get_xy(RIGHT_HIP)
                    L_elbow_vis = get_visibility(LEFT_ELBOW)
                    R_elbow_vis = get_visibility(RIGHT_ELBOW)
                    L_wrist_vis = get_visibility(LEFT_WRIST)
                    R_wrist_vis = get_visibility(RIGHT_WRIST)

                    left_elbow_angle     = calculate_angle(L_shoulder, L_elbow, L_wrist)
                    right_elbow_angle    = calculate_angle(R_shoulder, R_elbow, R_wrist)
                    left_shoulder_angle  = calculate_angle(L_hip, L_shoulder, L_elbow)
                    right_shoulder_angle = calculate_angle(R_hip, R_shoulder, R_elbow)
                    symmetry_diff        = round(abs(left_elbow_angle - right_elbow_angle), 1)
                    low_visibility_flag  = int(
                        min(L_elbow_vis, R_elbow_vis, L_wrist_vis, R_wrist_vis) < 0.5
                    )

                    symmetry_flag = symmetry_diff > SYMMETRY_THRESHOLD
                    lockout_flag  = (left_elbow_angle  < LOCKOUT_THRESHOLD or
                                     right_elbow_angle < LOCKOUT_THRESHOLD)

                    # Wrist y-pixel coordinates (landmarks 15 / 16). In image
                    # space a smaller y = higher position = bar overhead, which
                    # lets us locate the jerk lockout moment downstream.
                    left_wrist_y  = round(L_wrist[1], 1)
                    right_wrist_y = round(R_wrist[1], 1)
                    avg_wrist_y   = round((left_wrist_y + right_wrist_y) / 2, 1)

                    data.append({
                        "frame"                : frame_idx,
                        "left_elbow_angle"     : left_elbow_angle,
                        "right_elbow_angle"    : right_elbow_angle,
                        "left_shoulder_angle"  : left_shoulder_angle,
                        "right_shoulder_angle" : right_shoulder_angle,
                        "symmetry_diff"        : symmetry_diff,
                        "left_wrist_y"         : left_wrist_y,
                        "right_wrist_y"        : right_wrist_y,
                        "avg_wrist_y"          : avg_wrist_y,
                        "asymmetry_flag"       : int(symmetry_flag),
                        "lockout_flag"         : int(lockout_flag),
                        "low_visibility_flag"  : low_visibility_flag,
                    })

                    if show_display or writer is not None:
                        sym_color = (0, 0, 255) if symmetry_flag else (0, 255, 0)
                        cv2.putText(frame, f"L Elbow: {left_elbow_angle}",
                                    (int(L_elbow[0]) - 70, int(L_elbow[1]) - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                        cv2.putText(frame, f"R Elbow: {right_elbow_angle}",
                                    (int(R_elbow[0]) - 70, int(R_elbow[1]) - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                        cv2.putText(frame, f"Symmetry Diff: {symmetry_diff} deg",
                                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, sym_color, 2)
                        if symmetry_flag:
                            cv2.putText(frame, "ASYMMETRY DETECTED",
                                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                        if lockout_flag:
                            cv2.putText(frame, "INCOMPLETE LOCKOUT",
                                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 140, 255), 2)
                        cv2.putText(frame, f"Frame: {frame_idx}",
                                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            if writer is not None:
                try:
                    writer.write(frame)
                except Exception as exc:
                    print(f"Warning: failed to write annotated video frame ({exc}) -- "
                          f"disabling annotated video for the rest of this run.")
                    writer.release()
                    writer = None

            if show_display:
                display_height = max(1, int(h * 1280 / w))
                display_frame = cv2.resize(frame, (1280, display_height))
                cv2.imshow("Module 3 — Upper Limb Analysis", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    if show_display:
        cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    print("\n===== ANALYSIS SUMMARY =====")
    print(f"Total frames analysed  : {len(df)}")
    print(f"Avg Left Elbow Angle   : {df['left_elbow_angle'].mean():.1f} deg")
    print(f"Avg Right Elbow Angle  : {df['right_elbow_angle'].mean():.1f} deg")
    print(f"Avg Symmetry Diff      : {df['symmetry_diff'].mean():.1f} deg")
    print(f"Asymmetry flags        : {df['asymmetry_flag'].sum()} frames")
    print(f"Incomplete lockout     : {df['lockout_flag'].sum()} frames")
    print(f"CSV saved -> {output_csv_path}")
    if save_annotated_video:
        if os.path.exists(annotated_video_path):
            print(f"Annotated video saved -> {annotated_video_path}")
        else:
            print("Annotated video was NOT saved (see warning above).")

    return df


if __name__ == "__main__":
    # Example usage
    # VIDEO_INPUT = "data/raw_videos/cleanjerk2.mov"
    # CSV_OUTPUT  = "data/processed/arm_analysis_sample.csv"
    VIDEO_INPUT = "data/raw_videos/side/1good.MOV"
    CSV_OUTPUT  = "data/processed/1good_side.csv"
    analyze_video(VIDEO_INPUT, CSV_OUTPUT, show_display=True)
