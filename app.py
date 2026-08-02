"""
Weight Lifting Injuries Prediction — Module 3 Demo
Streamlit frontend for the arm/shoulder/elbow analysis module.

Run with: streamlit run app.py

Author: Pasindu (214027H)
"""
import os
import tempfile

import pandas as pd
import streamlit as st

from src.data.module3_arm_analysis.config import BASE_DIR, INTERIM_DIR, MASTER_DATASET_PATH
from src.data.module3_arm_analysis.predict import load_model, predict_from_features, predict_lift

st.set_page_config(page_title="Module 3 — Arm/Shoulder/Elbow Analysis", page_icon="🏋️", layout="wide")

# A handful of pre-processed lifts (all 3 camera views available) so the
# demo tab gives an instant result with no video processing wait.
DEMO_LIFTS = {
    "Good lift #43": "43good",
    "Good lift #48": "48good",
    "Good lift #55": "55good",
    "Bad lift #13": "13bad",
    "Bad lift #18": "18bad",
    "Bad lift #27": "27bad",
}


@st.cache_resource
def get_model_bundle():
    return load_model()


@st.cache_data
def load_master_dataset():
    return pd.read_csv(MASTER_DATASET_PATH)


@st.cache_data
def load_frame_csv(lift_id, view):
    path = os.path.join(INTERIM_DIR, f"{lift_id}_{view}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def render_result(prediction, features_row, frame_dfs=None):
    label = prediction["prediction"]
    proba_bad = prediction["probability_bad"]

    if label == "bad":
        st.error(f"⚠️  Predicted: **BAD** technique — confidence {proba_bad * 100:.0f}%")
    else:
        st.success(f"✅  Predicted: **GOOD** technique — confidence {(1 - proba_bad) * 100:.0f}%")

    st.subheader("Key measurements at jerk lockout")
    st.caption(
        "Lockout = the frame where the wrist is highest overhead "
        "(automatically detected). A fully extended arm reads close to 180°."
    )
    metrics = [
        ("Angle-view left elbow", features_row.get("angle_lockout_left_elbow")),
        ("Angle-view right elbow", features_row.get("angle_lockout_right_elbow")),
        ("Side-view left elbow", features_row.get("side_lockout_left_elbow")),
        ("Side-view right elbow", features_row.get("side_lockout_right_elbow")),
    ]
    cols = st.columns(len(metrics))
    for col, (label_text, val) in zip(cols, metrics):
        col.metric(label_text, f"{val:.0f}°" if val is not None and pd.notna(val) else "N/A")

    if prediction.get("imputed_features"):
        st.caption(
            f"Note: {len(prediction['imputed_features'])} feature(s) estimated from the "
            f"training average because that camera view wasn't provided: "
            f"{', '.join(prediction['imputed_features'])}"
        )

    if frame_dfs:
        available = {v: df for v, df in frame_dfs.items() if df is not None and "left_elbow_angle" in df.columns}
        if available:
            st.subheader("Elbow angle across the lift")
            view_tabs = st.tabs([v.capitalize() for v in available])
            for tab, (view, df) in zip(view_tabs, available.items()):
                with tab:
                    chart_df = df[["frame", "left_elbow_angle", "right_elbow_angle"]].set_index("frame")
                    st.line_chart(chart_df)


st.title("🏋️ Clean & Jerk — Arm / Shoulder / Elbow Analysis")
st.write(
    "**Module 3** of the Weight Lifting Injuries Prediction project. "
    "Predicts whether a lift shows good or bad upper-limb technique from "
    "elbow angle, symmetry, and jerk-lockout timing, extracted via pose "
    "estimation (MediaPipe) across up to three camera views."
)

tab_demo, tab_upload, tab_about = st.tabs(["🎬 Try a demo lift", "📤 Upload your own video", "ℹ️ About the model"])

with tab_demo:
    st.write("Pick one of 6 pre-processed example lifts for an instant result.")
    choice = st.selectbox("Demo lift", list(DEMO_LIFTS.keys()))
    lift_id = DEMO_LIFTS[choice]

    if st.button("Run prediction", key="demo_run"):
        master = load_master_dataset()
        row = master[master["lift_id"] == lift_id]
        if row.empty:
            st.error("Demo lift not found in the dataset. Was module3_master_dataset.csv regenerated?")
        else:
            features_row = row.iloc[0].to_dict()
            prediction = predict_from_features(features_row)
            frame_dfs = {view: load_frame_csv(lift_id, view) for view in ("side", "front", "angle45")}
            true_label = "good" if row.iloc[0]["label"] == 0 else "bad"
            st.caption(f"Ground truth label for this demo lift (from filename): **{true_label}**")
            render_result(prediction, features_row, frame_dfs)

with tab_upload:
    st.write(
        "Upload video(s) of a single lift. **Side and/or angle (3/4) view give the model "
        "the most useful signal** — those two views alone drive the top-8 features it "
        "relies on. Front view adds symmetry info but isn't required. "
        "Processing takes roughly 1-2 minutes per video."
    )
    side_file = st.file_uploader("Side view", type=["mp4", "mov", "avi", "mkv"], key="side_up")
    front_file = st.file_uploader("Front view", type=["mp4", "mov", "avi", "mkv"], key="front_up")
    angle_file = st.file_uploader("Angle (3/4) view", type=["mp4", "mov", "avi", "mkv"], key="angle_up")

    if st.button("Run prediction", key="upload_run"):
        uploads = {"side": side_file, "front": front_file, "angle45": angle_file}
        provided = {view: file for view, file in uploads.items() if file is not None}
        if not provided:
            st.warning("Upload at least one video first.")
        else:
            with tempfile.TemporaryDirectory() as tmp_dir:
                video_paths = {}
                for view, file in provided.items():
                    path = os.path.join(tmp_dir, f"{view}_{file.name}")
                    with open(path, "wb") as f:
                        f.write(file.getbuffer())
                    video_paths[view] = path

                with st.spinner("Running pose estimation and scoring the lift — this can take a couple of minutes..."):
                    try:
                        result = predict_lift(video_paths, show_display=False)
                    except Exception as exc:
                        st.error(f"Processing failed: {exc}")
                        result = None

            if result:
                render_result(result, result["features"], result["frame_dfs"])

with tab_about:
    bundle = get_model_bundle()
    st.write(f"**Model:** {type(bundle['model']).__name__}, trained on 68 labeled lifts (43 good / 25 bad).")
    st.write("**Cross-validated accuracy:** ~90% (5-fold stratified).")
    st.write(f"**Features the model uses:** {len(bundle['features'])} (selected from 59 engineered features)")
    with st.expander("Show selected feature names"):
        st.write(bundle["features"])

    fi_path = os.path.join(BASE_DIR, "reports", "figures", "module3", "feature_importance.png")
    if os.path.exists(fi_path):
        st.subheader("Feature importance")
        st.image(fi_path)

    st.write(
        "Features come from pose-estimated elbow/shoulder angles across three camera "
        "views (side, front, oblique 3/4 'angle' view), split into early/middle/final "
        "phases of the lift, plus the exact jerk-lockout moment (the frame where the "
        "wrist is highest overhead)."
    )
