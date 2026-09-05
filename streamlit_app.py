"""
GymSense AI - Streamlit Competition Dashboard
Student: Ranjitha B K
USN: 1SB24AI041
Branch: Artificial Intelligence and Machine Learning
Event: MACHINE SPECTRA 1.0

IMPORTANT:
This Streamlit app is a competition/ML dashboard for the existing GymSense AI project.
The full phone-sensor PWA (DeviceMotion + FastAPI + installable app) should remain on
an HTTPS FastAPI deployment such as Render.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------
# PAGE
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="GymSense AI",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

MODEL_PATH = MODELS_DIR / "exercise_classifier.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
DATASET_PATH = DATA_DIR / "sensor_data.csv"


# ---------------------------------------------------------------------
# STYLE
# ---------------------------------------------------------------------

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at 10% 0%, rgba(101, 72, 255, .16), transparent 28rem),
                radial-gradient(circle at 90% 10%, rgba(255, 55, 148, .12), transparent 25rem),
                #07101f;
            color: #f5f7fb;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            background: #081426;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .gs-hero {
            border: 1px solid rgba(255,255,255,.09);
            border-radius: 26px;
            padding: 28px;
            background:
                linear-gradient(135deg, rgba(17,39,74,.94), rgba(12,20,46,.94));
            margin-bottom: 18px;
        }

        .gs-kicker {
            color: #ff6eb4;
            font-size: .78rem;
            font-weight: 800;
            letter-spacing: .15em;
            text-transform: uppercase;
        }

        .gs-title {
            font-size: clamp(2.2rem, 6vw, 4.1rem);
            line-height: .98;
            font-weight: 900;
            margin-top: 8px;
            margin-bottom: 10px;
        }

        .gs-sub {
            color: #a9b7ce;
            max-width: 760px;
            line-height: 1.7;
        }

        .gs-pill {
            display: inline-block;
            margin-top: 14px;
            margin-right: 8px;
            border: 1px solid rgba(255,255,255,.10);
            background: rgba(255,255,255,.05);
            border-radius: 999px;
            padding: 7px 12px;
            color: #dbe4f4;
            font-size: .82rem;
        }

        .gs-card {
            border: 1px solid rgba(255,255,255,.08);
            border-radius: 20px;
            padding: 20px;
            background: rgba(255,255,255,.035);
            min-height: 100%;
        }

        .gs-card h3 {
            margin-top: 0;
            margin-bottom: 6px;
        }

        .gs-muted {
            color: #97a9c2;
        }

        .gs-flow {
            text-align: center;
            font-weight: 800;
            line-height: 2.05;
            border: 1px solid rgba(255,255,255,.08);
            border-radius: 22px;
            padding: 22px 12px;
            background: rgba(7, 18, 39, .72);
        }

        .gs-arrow {
            color: #ff67ad;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(255,255,255,.08);
            background: rgba(255,255,255,.035);
            border-radius: 18px;
            padding: 15px;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 44px;
            border-radius: 12px;
            padding-left: 15px;
            padding-right: 15px;
            background: rgba(255,255,255,.035);
        }

        .stTabs [aria-selected="true"] {
            background: rgba(255, 73, 162, .12) !important;
        }

        .gs-warning {
            border: 1px solid rgba(255, 192, 83, .25);
            background: rgba(255, 192, 83, .07);
            border-radius: 16px;
            padding: 14px 16px;
            color: #f4d59a;
            margin: 10px 0 18px 0;
        }

        .gs-success {
            border: 1px solid rgba(73, 222, 153, .24);
            background: rgba(73, 222, 153, .07);
            border-radius: 16px;
            padding: 14px 16px;
            color: #a8f1cc;
            margin: 10px 0 18px 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# LOADERS
# ---------------------------------------------------------------------

@st.cache_resource
def load_model_bundle() -> dict[str, Any]:
    bundle: dict[str, Any] = {
        "model": None,
        "scaler": None,
        "encoder": None,
        "metadata": {},
        "error": None,
    }

    try:
        if MODEL_PATH.exists():
            bundle["model"] = joblib.load(MODEL_PATH)

        if SCALER_PATH.exists():
            bundle["scaler"] = joblib.load(SCALER_PATH)

        if ENCODER_PATH.exists():
            bundle["encoder"] = joblib.load(ENCODER_PATH)

        if METADATA_PATH.exists():
            bundle["metadata"] = json.loads(
                METADATA_PATH.read_text(encoding="utf-8")
            )

    except Exception as exc:
        bundle["error"] = str(exc)

    return bundle


@st.cache_data
def load_dataset_summary() -> dict[str, Any]:
    if not DATASET_PATH.exists():
        return {
            "exists": False,
            "rows": 0,
            "columns": 0,
            "activities": [],
            "counts": pd.DataFrame(),
        }

    try:
        df = pd.read_csv(DATASET_PATH)

        activity_col = None
        for candidate in ("activity", "label", "exercise", "class"):
            if candidate in df.columns:
                activity_col = candidate
                break

        activities: list[str] = []
        counts = pd.DataFrame()

        if activity_col:
            activities = sorted(
                df[activity_col].dropna().astype(str).unique().tolist()
            )
            counts = (
                df[activity_col]
                .astype(str)
                .value_counts()
                .rename_axis("Activity")
                .reset_index(name="Samples")
            )

        return {
            "exists": True,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "activities": activities,
            "counts": counts,
        }

    except Exception as exc:
        return {
            "exists": True,
            "rows": 0,
            "columns": 0,
            "activities": [],
            "counts": pd.DataFrame(),
            "error": str(exc),
        }


bundle = load_model_bundle()
metadata = bundle["metadata"] or {}
dataset = load_dataset_summary()


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def percentage(value: Any) -> str:
    if value is None:
        return "—"

    try:
        number = float(value)
        if 0 <= number <= 1:
            number *= 100
        return f"{number:.2f}%"
    except (TypeError, ValueError):
        return "—"


def get_live_app_url() -> str:
    # Recommended: add GYMSENSE_APP_URL in Streamlit Cloud Secrets.
    try:
        secret_value = st.secrets.get("GYMSENSE_APP_URL", "")
    except Exception:
        secret_value = ""

    return (
        str(secret_value).strip()
        or os.getenv("GYMSENSE_APP_URL", "").strip()
    )


# ---------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------

st.markdown(
    """
    <div class="gs-hero">
        <div class="gs-kicker">MACHINE SPECTRA 1.0 • AIML MINI PROJECT</div>
        <div class="gs-title">GymSense <span style="color:#ff63ad;">AI</span></div>
        <div class="gs-sub">
            AI-powered workout activity recognition using smartphone
            accelerometer and gyroscope data. Sensor windows are converted
            into engineered features and classified by a trained machine-learning model.
        </div>
        <span class="gs-pill">📱 Phone Motion Sensors</span>
        <span class="gs-pill">🧠 Machine Learning</span>
        <span class="gs-pill">⚡ Real-Time Recognition</span>
        <span class="gs-pill">📊 Activity Analytics</span>
    </div>
    """,
    unsafe_allow_html=True,
)

student = metadata.get("student", {})
c1, c2, c3, c4 = st.columns(4)
c1.metric("Student", student.get("name", "Ranjitha B K"))
c2.metric("USN", student.get("usn", "1SB24AI041"))
c3.metric("Selected Model", metadata.get("selected_model", "Random Forest"))
c4.metric("Exercise Classes", len(metadata.get("classes", [])) or "—")


# ---------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------

tab_home, tab_model, tab_data, tab_flow, tab_live = st.tabs(
    [
        "🏠 Overview",
        "🧠 ML Model",
        "📊 Dataset",
        "⚙️ How It Works",
        "📱 Live Phone App",
    ]
)


# ---------------------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------------------

with tab_home:
    left, right = st.columns([1.35, 1])

    with left:
        st.markdown("### Project objective")
        st.write(
            "GymSense AI recognizes the activity being performed by analysing "
            "motion patterns from a phone's accelerometer and gyroscope. "
            "Known activities are classified by ML; low-confidence movement "
            "can fall back to **WORKOUT**, while inactivity is treated as **REST**."
        )

        classes = metadata.get("classes", [])
        if classes:
            st.markdown("### Trained activity classes")
            st.write(" • ".join(classes))

    with right:
        st.markdown(
            """
            <div class="gs-card">
                <h3>Competition Demo</h3>
                <div class="gs-muted">
                    Open app → choose avatar → enable sensors → calibrate →
                    start workout → perform exercises → watch prediction,
                    animation, reps and duration → end workout → show history
                    and ML evaluation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Saved model configuration")

    a, b, c, d = st.columns(4)
    a.metric(
        "Sampling rate",
        f"{metadata.get('sampling_rate_hz', '—')} Hz",
    )
    b.metric(
        "Window duration",
        f"{metadata.get('window_duration_seconds', '—')} s",
    )
    c.metric(
        "Features",
        metadata.get("total_features", "—"),
    )
    d.metric(
        "Training windows",
        metadata.get("total_training_windows", "—"),
    )

    if bundle["error"]:
        st.error(f"Model loading issue: {bundle['error']}")
    elif bundle["model"] is not None:
        st.markdown(
            '<div class="gs-success">✓ Trained ML model files are available in this deployment.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="gs-warning">Model file is not available. Ensure the models/ folder is pushed to GitHub.</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------

with tab_model:
    metrics = metadata.get("evaluation_metrics", {})

    st.markdown("### Saved evaluation results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", percentage(metrics.get("accuracy")))
    m2.metric("Precision", percentage(metrics.get("precision")))
    m3.metric("Recall", percentage(metrics.get("recall")))
    m4.metric("F1 Score", percentage(metrics.get("f1_score")))

    values = [
        metrics.get("accuracy"),
        metrics.get("precision"),
        metrics.get("recall"),
        metrics.get("f1_score"),
    ]

    try:
        numeric_values = [float(v) for v in values if v is not None]
    except Exception:
        numeric_values = []

    if numeric_values and min(numeric_values) >= 99.9:
        st.markdown(
            """
            <div class="gs-warning">
                Competition note: the currently saved evaluation is extremely high.
                If the training data is synthetic or highly controlled, explain that
                clearly and collect real phone-sensor recordings before claiming
                real-world performance.
            </div>
            """,
            unsafe_allow_html=True,
        )

    comparison = metadata.get("model_comparison", {})
    if comparison:
        rows = []
        for name, result in comparison.items():
            rows.append(
                {
                    "Model": name,
                    "Accuracy (%)": result.get("accuracy"),
                    "Precision (%)": result.get("precision"),
                    "Recall (%)": result.get("recall"),
                    "F1 (%)": result.get("f1_score"),
                }
            )

        comparison_df = pd.DataFrame(rows)
        st.markdown("### Model comparison")
        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
        )

    feature_importance = metadata.get("top_feature_importances", [])
    if feature_importance:
        st.markdown("### Top feature importance")
        fi_df = pd.DataFrame(feature_importance)

        if {"feature", "importance"}.issubset(fi_df.columns):
            chart_df = (
                fi_df[["feature", "importance"]]
                .set_index("feature")
                .sort_values("importance", ascending=False)
            )
            st.bar_chart(chart_df, height=360)

    cm = metadata.get("confusion_matrix", {})
    matrix = cm.get("matrix", [])
    labels = cm.get("labels", [])

    if matrix and labels:
        st.markdown("### Confusion matrix")
        cm_df = pd.DataFrame(
            matrix,
            index=labels,
            columns=labels,
        )
        st.dataframe(cm_df, use_container_width=True)


# ---------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------

with tab_data:
    if not dataset["exists"]:
        st.warning("`data/sensor_data.csv` was not found.")
    elif dataset.get("error"):
        st.error(f"Could not read dataset: {dataset['error']}")
    else:
        d1, d2, d3 = st.columns(3)
        d1.metric("Raw sensor rows", f"{dataset['rows']:,}")
        d2.metric("Columns", dataset["columns"])
        d3.metric("Activity labels", len(dataset["activities"]) or "—")

        st.markdown("### Dataset role")
        st.write(
            "Raw phone motion samples are grouped into short time windows. "
            "Statistical and motion features are extracted from each window "
            "before training the classifiers."
        )

        if not dataset["counts"].empty:
            st.markdown("### Samples by activity")
            counts_chart = dataset["counts"].set_index("Activity")
            st.bar_chart(counts_chart, height=380)
            st.dataframe(
                dataset["counts"],
                use_container_width=True,
                hide_index=True,
            )


# ---------------------------------------------------------------------
# HOW IT WORKS
# ---------------------------------------------------------------------

with tab_flow:
    st.markdown("### End-to-end ML pipeline")

    st.markdown(
        """
        <div class="gs-flow">
            📱 SMARTPHONE<br>
            <span class="gs-arrow">↓</span><br>
            ACCELEROMETER + GYROSCOPE<br>
            <span class="gs-arrow">↓</span><br>
            SENSOR WINDOWING<br>
            <span class="gs-arrow">↓</span><br>
            FEATURE EXTRACTION<br>
            <span class="gs-arrow">↓</span><br>
            🧠 TRAINED ML CLASSIFIER<br>
            <span class="gs-arrow">↓</span><br>
            PREDICTION + CONFIDENCE<br>
            <span class="gs-arrow">↓</span><br>
            PREDICTION SMOOTHING<br>
            <span class="gs-arrow">↓</span><br>
            EXERCISE / REST / WORKOUT<br>
            <span class="gs-arrow">↓</span><br>
            ANIMATION + TIMER + REPS + SETS<br>
            <span class="gs-arrow">↓</span><br>
            HISTORY + ANALYTICS
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Why both sensors?")
    x, y = st.columns(2)

    with x:
        st.markdown(
            """
            <div class="gs-card">
                <h3>Accelerometer</h3>
                <div class="gs-muted">
                    Measures linear motion and changes along X, Y and Z.
                    It helps distinguish stillness, walking, running and
                    repeated exercise movement.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with y:
        st.markdown(
            """
            <div class="gs-card">
                <h3>Gyroscope / Rotation</h3>
                <div class="gs-muted">
                    Adds rotational movement information. This helps separate
                    exercises that may have similar acceleration but different
                    orientation or arm/body rotation patterns.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------
# LIVE PHONE APP
# ---------------------------------------------------------------------

with tab_live:
    st.markdown("### Live smartphone workout app")

    st.info(
        "Your full phone-sensor interface is a FastAPI + HTML/CSS/JavaScript PWA. "
        "That version should be deployed on an HTTPS FastAPI host such as Render. "
        "Streamlit is best used here as the ML/project dashboard."
    )

    live_url = get_live_app_url()

    if live_url:
        st.link_button(
            "📱 Open GymSense AI Live Workout",
            live_url,
            use_container_width=True,
        )
        st.caption(
            "Open this HTTPS link on your phone to use the real accelerometer "
            "and gyroscope workflow."
        )
    else:
        st.markdown(
            """
            <div class="gs-card">
                <h3>Connect your live app</h3>
                <div class="gs-muted">
                    After deploying the FastAPI/PWA version, add its HTTPS URL
                    to Streamlit Cloud Secrets as:
                    <br><br>
                    <code>GYMSENSE_APP_URL = "https://your-app.onrender.com"</code>
                    <br><br>
                    This tab will then show a direct button to your real phone app.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Why not run the full sensor PWA inside Streamlit?")
    st.write(
        "Your production workout app depends on custom browser motion APIs, "
        "service-worker/PWA behavior, and FastAPI prediction routes. Keeping that "
        "part on its native HTTPS deployment preserves the sensor and installable-app "
        "experience, while this Streamlit page gives judges a clean ML dashboard."
    )


# ---------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------

st.divider()
st.caption(
    "GymSense AI • Ranjitha B K • 1SB24AI041 • "
    "Artificial Intelligence and Machine Learning • MACHINE SPECTRA 1.0"
)
