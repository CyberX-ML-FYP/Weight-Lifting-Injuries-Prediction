"""
Weight Lifting Injuries Prediction — Streamlit Frontend

Run with: streamlit run app.py
"""
import os
import tempfile

import pandas as pd
import streamlit as st

from src.models.predict_model import (
    VALID_VIEWS as M1_VIEWS,
    load_model_bundle as load_module1_bundle,
    predict_video as predict_module1_video,
    score_feature_row as score_module1_feature_row,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")

st.set_page_config(page_title="Weight Lifting Injuries Prediction", page_icon="🏋️", layout="wide")

# ─────────────────────────── Module 1 — Trunk / Spine ───────────────────────────

M1_DEMO_LIFTS = {
    "Good lift #10": "10good",
    "Good lift #11": "11good",
    "Good lift #12": "12good",
    "Good lift #14": "14good",
    "Bad lift #13": "13bad",
    "Bad lift #18": "18bad",
    "Bad lift #19": "19bad",
    "Bad lift #21": "21bad",
}


@st.cache_data
def load_module1_features():
    path = os.path.join(FEATURES_DIR, "module1", "module1_features.csv")
    return pd.read_csv(path)


@st.cache_resource
def get_module1_bundle():
    return load_module1_bundle()


def render_module1_result(result):
    label = result["predicted_class"]
    confidence = result["confidence"]

    if label == "bad":
        st.error(f"⚠️  Predicted: **BAD** technique — confidence {confidence * 100:.0f}%")
    else:
        st.success(f"✅  Predicted: **GOOD** technique — confidence {confidence * 100:.0f}%")

    st.caption(f"View: {result['view']}")

    st.subheader("Top contributing features")
    fi_df = pd.DataFrame(result["top_features"])
    if not fi_df.empty:
        fi_df = fi_df.rename(columns={"feature": "Feature", "value": "This video's value", "importance": "Model importance"})
        st.dataframe(fi_df.set_index("Feature"), width="stretch")

    st.info(
        "Module 1 (trunk/spine) acting alone is ~66% cross-validated accuracy — "
        "barely above the 65.5% majority-class baseline. Treat this as one "
        "component signal, not a final verdict, until Modules 2-4 are merged "
        "into a full master dataset."
    )


def render_module1_tab():
    st.write(
        "**Module 1** analyses trunk/spine posture during the lift — spine angle, "
        "forward lean, postural deviation, and shoulder asymmetry — extracted via "
        "MediaPipe pose estimation and summarised over the *dynamic lift-phase* "
        "portion of the clip (pull/catch/dip/drive), excluding walk-in, walk-out, "
        "and idle standing time."
    )

    m1_demo, m1_upload, m1_about = st.tabs(["🎬 Try a demo lift", "📤 Upload your own video", "ℹ️ About the model"])

    with m1_demo:
        st.write("Pick one of the pre-processed example lifts for an instant result (no video processing wait).")
        choice = st.selectbox("Demo lift", list(M1_DEMO_LIFTS.keys()), key="m1_demo_lift")
        view_choice = st.selectbox("View", M1_VIEWS, key="m1_demo_view")
        lift_id = M1_DEMO_LIFTS[choice]

        if st.button("Run prediction", key="m1_demo_run"):
            features = load_module1_features()
            row = features[(features["video_id"] == lift_id) & (features["view"] == view_choice)]
            if row.empty:
                st.error(f"No precomputed features for {lift_id} ({view_choice}).")
            else:
                feature_row = row.iloc[0].to_dict()
                bundle = get_module1_bundle()
                result = score_module1_feature_row(feature_row, bundle=bundle)
                true_label = "good" if feature_row["label"] == 0 else "bad"
                st.caption(f"Ground truth label for this demo lift (from filename): **{true_label}**")
                render_module1_result(result)

    with m1_upload:
        st.write(
            "Upload video(s) for one lift attempt — front, side, and/or 45° angle. "
            "**Each view is scored independently**: Module 1's model does not yet "
            "combine multiple views into a single prediction, so you'll get one "
            "result per video you upload. Processing takes roughly 1-2 minutes "
            "per video (pose estimation runs on every frame)."
        )
        front_file = st.file_uploader("Front view", type=["mp4", "mov", "avi", "mkv"], key="m1_front_up")
        side_file = st.file_uploader("Side view", type=["mp4", "mov", "avi", "mkv"], key="m1_side_up")
        angle_file = st.file_uploader("45° angle view", type=["mp4", "mov", "avi", "mkv"], key="m1_angle_up")

        if st.button("Run prediction", key="m1_upload_run"):
            uploads = {"front": front_file, "side": side_file, "angle45": angle_file}
            provided = {view: file for view, file in uploads.items() if file is not None}
            if not provided:
                st.warning("Upload at least one video first.")
            else:
                bundle = get_module1_bundle()
                with tempfile.TemporaryDirectory() as tmp_dir:
                    for view, file in provided.items():
                        video_dir = os.path.join(tmp_dir, view)
                        os.makedirs(video_dir, exist_ok=True)
                        video_path = os.path.join(video_dir, file.name)
                        with open(video_path, "wb") as f:
                            f.write(file.getbuffer())

                        st.markdown(f"### {view.capitalize()} view result")
                        with st.spinner(f"Processing {view} view — this can take a minute or two..."):
                            try:
                                result = predict_module1_video(video_path, view=view, bundle=bundle)
                            except Exception as exc:
                                st.error(f"{view} processing failed: {exc}")
                                continue
                        render_module1_result(result)

    with m1_about:
        try:
            bundle = get_module1_bundle()
        except FileNotFoundError as exc:
            st.error(str(exc))
        else:
            st.write(f"**Model:** {type(bundle['model']).__name__}")
            st.write(
                "**Cross-validated accuracy (trunk/spine features only):** ~66% — "
                "barely above the 65.5% majority-class baseline, grouped by lift "
                "so no view of a given lift leaks between train/test folds."
            )
            st.write(f"**Features the model uses:** {len(bundle['features'])}")
            with st.expander("Show selected feature names"):
                st.write(bundle["features"])

            fi_path = os.path.join(BASE_DIR, "reports", "figures", "module1", "feature_importance.png")
            if os.path.exists(fi_path):
                st.subheader("Feature importance")
                st.image(fi_path)

            st.warning(
                "Trunk/spine posture alone doesn't reliably separate good vs. bad "
                "lifts in this dataset — likely because lift-quality labels here are "
                "driven more by hip/knee, arm/shoulder, or bar-path mechanics than by "
                "trunk lean. Real predictive power will need the full multi-module "
                "master dataset once Modules 2 and 4 are built and merged in."
            )


# ─────────────────────────── Module 3 — Arm / Shoulder / Elbow ───────────────────────────

def render_module3_tab():
    try:
        from src.module3_arm_analysis.config import BASE_DIR as M3_BASE_DIR, DATA_DIR as M3_DATA_DIR
        from src.module3_arm_analysis.predict import load_model, predict_from_features, predict_lift
    except ImportError as exc:
        st.error(
            "Module 3's frontend isn't wired up in this checkout yet "
            f"(`{exc}`). Its predict.py hasn't landed on this branch — nothing "
            "to do here on the Module 1 side, this tab just isn't ready."
        )
        return

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
        path = os.path.join(M3_DATA_DIR, "processed", "module3_master_dataset.csv")
        return pd.read_csv(path)

    @st.cache_data
    def load_frame_csv(lift_id, view):
        path = os.path.join(M3_DATA_DIR, "processed", f"{lift_id}_{view}.csv")
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
                frame_dfs = {view: load_frame_csv(lift_id, view) for view in ("side", "front", "angle")}
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
            uploads = {"side": side_file, "front": front_file, "angle": angle_file}
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

        fi_path = os.path.join(M3_BASE_DIR, "reports", "figures", "module3", "feature_importance.png")
        if os.path.exists(fi_path):
            st.subheader("Feature importance")
            st.image(fi_path)

        st.write(
            "Features come from pose-estimated elbow/shoulder angles across three camera "
            "views (side, front, oblique 3/4 'angle' view), split into early/middle/final "
            "phases of the lift, plus the exact jerk-lockout moment (the frame where the "
            "wrist is highest overhead)."
        )


# ─────────────────────────── Top-level layout ───────────────────────────

st.title("🏋️ Weight Lifting Injuries Prediction")
st.write("Multi-View Mathematical Analysis and AI-Based Performance & Risk Assessment for Clean & Jerk Weightlifting.")

tab_module1, tab_module3 = st.tabs(["Module 1 — Trunk & Spine", "Module 3 — Arm / Shoulder / Elbow"])

with tab_module1:
    render_module1_tab()

with tab_module3:
    render_module3_tab()
