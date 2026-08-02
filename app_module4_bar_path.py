"""
Weight Lifting Injuries Prediction — Module 4 Demo
Streamlit frontend for the bar-path / injury-risk module.

Standalone app kept separate from the repo-root app.py (Module 3's
Streamlit frontend) to avoid filename collision and cross-module merge
risk. Proposing a single unified multi-module app is future work once
every module has a stable predict() entry point.

Run with: streamlit run app_module4_bar_path.py

Author: Senarathna G.G.P.C. (214189E) — Module 4
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.data.module4_bar_path.bar_path_predict import (
    clean_step,
    extract_step,
    predict_from_features,
    summarize_step,
)
from src.data.module4_bar_path.bar_path_train import FEATURE_COLUMNS
from src.data.module4_bar_path.config import BarPathConfig
from src.data.module4_bar_path.render_demo_skeletons import DEMO_OUTPUT_DIRNAME
from src.data.module4_bar_path.skeleton_video import render_skeleton_video
from src.data.module4_bar_path.utils import setup_logger

logger = setup_logger(__name__)

st.set_page_config(page_title="Module 4 — Bar Path & Injury Risk", page_icon="🏋️", layout="wide")

FEATURE_LABELS = {
    "max_deviation": "Max horizontal bar deviation",
    "avg_deviation": "Average horizontal bar deviation",
    "path_smoothness": "Path smoothness (lower = smoother)",
    "peak_vertical_velocity": "Peak vertical bar speed",
    "time_to_peak_velocity": "Timing of peak speed (fraction of lift)",
    "total_displacement": "Total bar path length",
    "jerk_like_movements": "Sudden direction/speed changes",
}

FEATURE_EXPLANATIONS = {
    "max_deviation": "How far the bar strayed from a straight vertical line at its worst point. Large values mean the lifter had to muscle the bar back under control, straining the back/shoulders to compensate.",
    "avg_deviation": "The bar's typical distance from a straight vertical line across the whole lift — the single strongest signal in this model for separating good and bad technique.",
    "path_smoothness": "How jerky the bar's speed changes were, frame to frame. A rough, unsmooth path suggests unstable bar control or poor timing between pull and catch.",
    "peak_vertical_velocity": "The fastest the bar moved upward. Not risky by itself — fast bar speed is good technique — but combined with high deviation it suggests losing control at speed.",
    "time_to_peak_velocity": "When in the lift the bar was moving fastest, as a fraction of the whole lift (0 = start, 1 = end).",
    "total_displacement": "How far the bar travelled in total across the lift. A longer, more roundabout path means more wasted motion and more opportunity for the bar to drift off-line.",
    "jerk_like_movements": "How many sudden accelerations/corrections happened — a proxy for the body absorbing a bad position it wasn't ready for.",
}

# A handful of pre-processed lifts (cleaned coordinates + features already
# on disk) so the demo tab gives an instant result with no video
# processing wait, matching the pattern in the Module 3 app.
DEMO_LIFTS = {
    "Good lift #2": "2good",
    "Good lift #10": "10good",
    "Good lift #46": "46good",
    "Bad lift #13": "13bad",
    "Bad lift #19": "19bad",
    "Bad lift #62": "62bad",
}


@st.cache_data
def load_features_table():
    config = BarPathConfig()
    return pd.read_csv(config.features_output_path)


@st.cache_data
def load_cleaned_coords(video_id: str):
    config = BarPathConfig()
    path = config.processed_output_dir / f"{video_id}_cleaned.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def find_demo_skeleton_video_path(video_id: str) -> Path | None:
    """Pre-rendered skeleton-overlay video, built by
    render_demo_skeletons.py so the demo tab doesn't have to run MediaPipe
    live. Falls back to the plain source video (find_demo_video_path) if
    the batch script hasn't been (re-)run for this lift yet."""
    config = BarPathConfig()
    path = config.root_dir / "reports" / "figures" / "module4" / DEMO_OUTPUT_DIRNAME / f"{video_id}_skeleton.mp4"
    return path if path.exists() else None


def find_demo_video_path(video_id: str) -> Path | None:
    config = BarPathConfig()
    for ext in config.video_extensions:
        candidate = config.raw_video_dir / f"{video_id}{ext}"
        if candidate.exists():
            return candidate
        candidate = config.raw_video_dir / f"{video_id}{ext.upper()}"
        if candidate.exists():
            return candidate
    return None


def render_risk_banner(report: dict) -> None:
    band = report["injury_risk_band"]
    score_pct = report["injury_risk_score"] * 100

    if band == "High":
        st.error(f"⚠️ **HIGH injury risk** — {score_pct:.0f}% risk score. Predicted technique: **{report['predicted_quality'].upper()}**")
    elif band == "Moderate":
        st.warning(f"⚠️ **MODERATE injury risk** — {score_pct:.0f}% risk score. Predicted technique: **{report['predicted_quality'].upper()}**")
    else:
        st.success(f"✅ **LOW injury risk** — {score_pct:.0f}% risk score. Predicted technique: **{report['predicted_quality'].upper()}**")

    st.caption(
        "Risk score = probability of 'bad' bar-path technique from the trained model ensemble. "
        "This is a mechanically-motivated proxy (poor bar control -> more compensation from the "
        "back/shoulders), not a clinically validated injury prediction."
    )


def render_model_breakdown(report: dict) -> None:
    st.subheader("Model breakdown")
    cols = st.columns(len(report["per_model_risk_scores"]) + 1)
    for col, (model_name, score) in zip(cols, report["per_model_risk_scores"].items()):
        col.metric(model_name.upper(), f"{score * 100:.0f}%")
    cols[-1].metric("Ensemble", f"{report['injury_risk_score'] * 100:.0f}%")


def render_top_features(report: dict) -> None:
    st.subheader("What drove this result")
    for feature in report["top_contributing_features"]:
        label = FEATURE_LABELS.get(feature, feature)
        value = report["features"].get(feature)
        explanation = FEATURE_EXPLANATIONS.get(feature, "")
        st.markdown(f"**{label}** — `{value:.4f}`  \n{explanation}")


def render_bar_path_chart(cleaned_df: pd.DataFrame | None) -> None:
    """Plot the bar's actual spatial trajectory (x vs y), not x/y against
    time -- this is the shape a coach actually reads: does the bar travel
    in a tight vertical line, or does it loop/drift out during the pull?
    A time-series line chart of x and y separately doesn't show that shape
    at all, which is why this uses a real x-vs-y plot instead of Streamlit's
    built-in line_chart (which can only plot columns against a shared index,
    not against each other)."""
    if cleaned_df is None or cleaned_df.empty:
        st.info("Bar path trajectory not available for this lift.")
        return

    st.subheader("Bar path trace")
    st.caption(
        "The barbell's actual path through space, front-view, trimmed to the lifting phase only "
        "(walk-in/walk-out removed). A straight vertical line is ideal technique; loops or sideways "
        "drift mean the lifter had to muscle the bar back under control. Arrows show direction of "
        "travel — the bar can legitimately double back on itself near the top (lockout, then starting "
        "to lower), so the path alone isn't always enough to tell start from finish."
    )

    x = cleaned_df["x"].to_numpy(dtype=float)
    y = cleaned_df["y"].to_numpy(dtype=float)
    progress = np.linspace(0.0, 1.0, len(x))

    fig, ax = plt.subplots(figsize=(4.5, 6))

    # Faint vertical reference line at the bar's starting x -- the "ideal"
    # path a lifter is trying to stay close to.
    ax.axvline(x[0], color="#9ca3af", linewidth=1, linestyle="--", zorder=1)

    # Color-graded path segments encode direction of travel (start -> finish)
    # without needing an animation -- a single flat color would lose that.
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    from matplotlib.collections import LineCollection
    lc = LineCollection(segments, cmap="viridis", linewidth=2.5, zorder=2)
    lc.set_array(progress)
    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax, label="Progress through lift", shrink=0.6, pad=0.02)

    # Direction arrows at regular intervals -- the path can legitimately
    # double back near lockout (bar rises, then starts lowering again), so
    # color alone can be ambiguous about which way is "forward." Each arrow
    # spans several frames (not just one) so it's actually visible at the
    # path's own scale, with a white outline so it reads against any
    # colour the viridis gradient happens to be at that point.
    n_arrows = min(7, max(1, len(x) // 12))
    span = max(2, len(x) // (n_arrows + 1) // 3)
    if len(x) > span:
        arrow_idx = np.linspace(0, len(x) - 1 - span, n_arrows, dtype=int)
        for i in arrow_idx:
            j = i + span
            ax.annotate(
                "", xy=(x[j], y[j]), xytext=(x[i], y[i]),
                arrowprops=dict(
                    arrowstyle="-|>", color="#1f2937", lw=1.5,
                    mutation_scale=18, shrinkA=0, shrinkB=0,
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                ),
                zorder=4,
            )

    # Direct-labeled start/end points, colored distinctly from the path's
    # own viridis gradient so they don't blend into a similarly-colored
    # segment of the trace itself.
    ax.scatter([x[0]], [y[0]], color="#16a34a", s=110, zorder=5, edgecolor="white", linewidth=2)
    ax.annotate("Start", (x[0], y[0]), textcoords="offset points", xytext=(10, 8), fontsize=11, fontweight="bold", color="#166534")
    ax.scatter([x[-1]], [y[-1]], color="#dc2626", s=110, zorder=5, edgecolor="white", linewidth=2, marker="s")
    ax.annotate("Finish", (x[-1], y[-1]), textcoords="offset points", xytext=(10, 8), fontsize=11, fontweight="bold", color="#991b1b")

    ax.set_xlim(max(0.0, x.min() - 0.08), min(1.0, x.max() + 0.08))
    ax.set_ylim(1.02, -0.02)  # y=0 is top of frame, y=1 is ground -- flip so "up" reads as up
    ax.set_xlabel("Horizontal position (0.5 = frame centre)")
    ax.set_ylabel("Height (ground → overhead)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#d1d5db")
    ax.tick_params(colors="#6b7280")
    ax.set_facecolor("#fafafa")
    fig.tight_layout()

    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


def render_wrist_deviation_chart(cleaned_df: pd.DataFrame | None) -> None:
    """Left vs right wrist horizontal deviation from each wrist's own
    starting position, over time. This is a different question than the
    spatial bar-path trace above: that chart shows the combined bar
    (wrist-midpoint) shape, this one isolates each wrist and lets a coach
    read off WHEN each arm drifted and whether the two arms drifted the
    same amount (bilateral symmetry) or not."""
    if cleaned_df is None or cleaned_df.empty:
        st.info("Wrist trajectory not available for this lift.")
        return
    required = {"left_wrist_x", "right_wrist_x", "frame_width", "frame_id"}
    if not required.issubset(cleaned_df.columns):
        st.info("Wrist-level columns not available for this lift.")
        return

    st.subheader("Left vs right wrist deviation")
    st.caption(
        "Horizontal drift of each wrist from its own starting position, as a fraction of frame "
        "width, across the lift. Large or diverging left/right values indicate the bar drifted "
        "off-line or the lift was asymmetric between arms."
    )

    df = cleaned_df.sort_values("frame_id").reset_index(drop=True)
    left_x = df["left_wrist_x"].to_numpy(dtype=float) / df["frame_width"].to_numpy(dtype=float)
    right_x = df["right_wrist_x"].to_numpy(dtype=float) / df["frame_width"].to_numpy(dtype=float)

    left_dev = left_x - left_x[0]
    right_dev = right_x - right_x[0]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.axhline(0.0, color="#9ca3af", linewidth=1, linestyle="--", zorder=1)
    ax.plot(df["frame_id"], left_dev, color="#2563eb", linewidth=2, label="Left wrist", zorder=2)
    ax.plot(df["frame_id"], right_dev, color="#ea580c", linewidth=2, label="Right wrist", zorder=2)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Horizontal deviation\n(fraction of frame width)")
    ax.legend(loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#d1d5db")
    ax.tick_params(colors="#6b7280")
    ax.set_facecolor("#fafafa")
    fig.tight_layout()

    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


st.title("🏋️ Clean & Jerk — Bar Path & Injury Risk Analysis")
st.write(
    "**Module 4** of the Weight Lifting Injuries Prediction project. "
    "Tracks the barbell's path from a **front-view** video and estimates injury risk from "
    "how far and how erratically the bar strayed from an ideal vertical line, using an "
    "ensemble of Random Forest and XGBoost models trained on 58 labelled lifts."
)
st.caption(
    "This module analyses the **front view only** — side and 45° views are used by the "
    "other three modules (trunk, hip/knee, arm) in this project, not by bar-path tracking."
)

tab_demo, tab_upload, tab_about = st.tabs(["🎬 Try a demo lift", "📤 Upload your own video", "ℹ️ About the model"])

with tab_demo:
    st.write("Pick one of 6 pre-processed example lifts for an instant result.")
    choice = st.selectbox("Demo lift", list(DEMO_LIFTS.keys()))
    video_id = DEMO_LIFTS[choice]

    if st.button("Run prediction", key="demo_run"):
        features_table = load_features_table()
        row = features_table[features_table["video_id"] == video_id]
        if row.empty:
            st.error("Demo lift not found in bar_path_features.csv. Was the feature table regenerated?")
        else:
            lift_features = row[["video_id"] + FEATURE_COLUMNS].reset_index(drop=True)
            true_label = "good" if row.iloc[0]["label"] == 0 else "bad"
            st.caption(f"Ground truth label for this demo lift (from filename): **{true_label}**")

            report = predict_from_features(lift_features)
            render_risk_banner(report)
            render_model_breakdown(report)

            video_col, chart_col = st.columns([1, 1])
            with video_col:
                show_skeleton = st.checkbox("Show tracked landmarks", value=True, key="demo_skeleton_toggle")
                skeleton_path = find_demo_skeleton_video_path(video_id) if show_skeleton else None
                if skeleton_path:
                    st.video(str(skeleton_path))
                    st.caption("Green skeleton = the lifter MediaPipe selected and tracked, trimmed to the lifting phase.")
                else:
                    demo_video_path = find_demo_video_path(video_id)
                    if demo_video_path:
                        st.video(str(demo_video_path))
                        if show_skeleton:
                            st.caption("No pre-rendered skeleton video for this lift — showing the source video instead.")
                    else:
                        st.info("Source video not found on disk for this demo lift.")
            with chart_col:
                render_top_features(report)

            cleaned_df = load_cleaned_coords(video_id)
            render_bar_path_chart(cleaned_df)
            render_wrist_deviation_chart(cleaned_df)

with tab_upload:
    st.write(
        "Upload a **front-view** video of a single clean & jerk attempt. "
        "Processing runs pose estimation frame by frame and typically takes 10-30 seconds."
    )
    front_file = st.file_uploader("Front view", type=["mp4", "mov"], key="front_up")

    if st.button("Run prediction", key="upload_run"):
        if front_file is None:
            st.warning("Upload a front-view video first.")
        else:
            # Write to a PERSISTENT scratch dir, not a TemporaryDirectory --
            # st.video() is only reliably playable in the browser when given
            # a file PATH (it streams it with proper HTTP range-request
            # support, matching what the demo tab already does successfully).
            # Passing raw bytes instead worked in principle but was the
            # actual cause of "video won't play" here: by the time
            # st.video(bytes(...)) ran, either the source had already been
            # deleted (TemporaryDirectory closes as soon as the `with` block
            # exits) or the browser couldn't seek/range-request a bytes blob
            # the way it can a served file. Old uploads are cleared each run
            # so this directory doesn't grow unbounded.
            config = BarPathConfig()
            upload_dir = config.interim_output_dir / "app_uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            for old_file in upload_dir.glob("*"):
                old_file.unlink(missing_ok=True)

            video_path = upload_dir / front_file.name
            video_path.write_bytes(front_file.getbuffer())

            skeleton_path = None
            try:
                with st.status("Analysing bar path...", expanded=True) as status:
                    st.write("Extracting pose landmarks frame by frame...")
                    raw_df = extract_step(video_path, config)

                    st.write("Detecting lift phase and cleaning the trajectory...")
                    cleaned_df = clean_step(raw_df, config, video_path.stem)

                    st.write("Computing bar-path features...")
                    lift_features = summarize_step(cleaned_df, video_path.stem)

                    st.write("Scoring with the model ensemble...")
                    report = predict_from_features(lift_features, config)

                    st.write("Rendering tracked-landmark video...")
                    candidate_skeleton_path = upload_dir / f"{video_path.stem}_skeleton.mp4"
                    try:
                        render_skeleton_video(video_path, candidate_skeleton_path, config)
                        skeleton_path = candidate_skeleton_path
                    except Exception:
                        logger.exception("Skeleton video rendering failed for %s", video_path.name)

                    status.update(label="Done", state="complete", expanded=False)
            except Exception as exc:
                st.error(f"Processing failed: {exc}")
                report, cleaned_df = None, None

            if report:
                render_risk_banner(report)
                render_model_breakdown(report)

                video_col, chart_col = st.columns([1, 1])
                with video_col:
                    show_skeleton = st.checkbox("Show tracked landmarks", value=True, key="upload_skeleton_toggle")
                    if show_skeleton and skeleton_path:
                        st.video(str(skeleton_path))
                        st.caption("Green skeleton = the lifter MediaPipe selected and tracked, trimmed to the lifting phase.")
                    else:
                        st.video(str(video_path))
                        if show_skeleton:
                            st.caption("Landmark tracking failed for this video — showing the original upload instead.")
                with chart_col:
                    render_top_features(report)

                render_bar_path_chart(cleaned_df)
                render_wrist_deviation_chart(cleaned_df)

with tab_about:
    st.write(
        "**Models:** Random Forest + XGBoost ensemble (simple average of predicted probabilities), "
        "trained on 58 labelled lifts (38 good / 20 bad)."
    )
    st.write("**Cross-validated accuracy:** ~81% (RF), ~79% (XGBoost), 5-fold stratified.")
    st.write(
        "An Attention-LSTM was also built and evaluated on the raw per-frame bar trajectory, but "
        "excluded from the ensemble: with only 58 training sequences its predictions were unreliable "
        "(low-confidence, all crammed into a narrow probability band). It will be revisited once more "
        "labelled videos are available."
    )

    st.subheader("Features used")
    for feature in FEATURE_COLUMNS:
        st.markdown(f"**{FEATURE_LABELS.get(feature, feature)}** — {FEATURE_EXPLANATIONS.get(feature, '')}")

    st.subheader("Risk bands")
    st.markdown(
        "- **Low** — risk score < 33%\n"
        "- **Moderate** — 33% – 66%\n"
        "- **High** — risk score ≥ 66%"
    )

    st.caption(
        "There is no clinical injury-outcome dataset behind this project — only coach-assigned "
        "good/bad technique labels. Injury risk here is a mechanically-motivated proxy (poor bar "
        "control implies more compensation from the back/shoulders), not a validated medical prediction."
    )
