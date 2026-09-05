from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from feature_extraction import extract_features_from_records, get_feature_names


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="GymSense AI",
    page_icon="💪",
    layout="centered",
    initial_sidebar_state="collapsed",
)

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"

SMOOTH_WINDOW = 2

# Let walking/running survive at lower confidence,
# but require stronger confidence for gym exercises.
WALK_RUN_MIN_CONF = 0.12
EXERCISE_MIN_CONF = 0.45
GENERAL_MIN_CONF = 0.50

REPETITIVE = {
    "BICEP CURL",
    "HAMMER CURL",
    "SQUAT",
    "LUNGE",
    "JUMPING JACK",
    "SHOULDER PRESS",
    "FRONT RAISE",
    "LATERAL RAISE",
}

KNOWN_ACTIVITIES = {
    "REST",
    "WALKING",
    "RUNNING",
    "BICEP CURL",
    "HAMMER CURL",
    "SQUAT",
    "LUNGE",
    "JUMPING JACK",
    "SHOULDER PRESS",
    "FRONT RAISE",
    "LATERAL RAISE",
}

ACTIVITY_EMOJI = {
    "READY": "⚡",
    "REST": "🧘",
    "WALKING": "🚶",
    "RUNNING": "🏃",
    "BICEP CURL": "💪",
    "HAMMER CURL": "🏋️",
    "SQUAT": "🏋️",
    "LUNGE": "🦵",
    "JUMPING JACK": "⭐",
    "SHOULDER PRESS": "🏋️",
    "FRONT RAISE": "💪",
    "LATERAL RAISE": "💪",
    "WORKOUT": "🔥",
}


# ============================================================
# GLOBAL UI
# ============================================================

st.markdown(
    """
<style>
html, body, [class*="css"] {
    font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 8% 0%, rgba(86, 72, 255, .18), transparent 24rem),
        radial-gradient(circle at 95% 0%, rgba(255, 62, 158, .14), transparent 24rem),
        #06101f;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 620px !important;
    padding-top: .55rem !important;
    padding-left: .55rem !important;
    padding-right: .55rem !important;
    padding-bottom: 2rem !important;
}

.gs-brand {
    font-size: 1.26rem;
    font-weight: 950;
    letter-spacing: -.045em;
    line-height: 1;
}

.gs-brand span {
    color: #ff58aa;
}

.gs-sub {
    color: #8397b5;
    font-size: .65rem;
    margin-top: 5px;
}

.gs-status {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border: 1px solid rgba(60, 232, 165, .28);
    background: rgba(60, 232, 165, .07);
    color: #53e2a6;
    border-radius: 999px;
    padding: 4px 8px;
    font-size: .60rem;
    font-weight: 900;
}

.gs-card {
    border: 1px solid rgba(255,255,255,.08);
    background: linear-gradient(145deg, rgba(15,31,57,.96), rgba(7,17,33,.98));
    border-radius: 17px;
    padding: 10px;
    margin-bottom: 8px;
}

.gs-kicker {
    color: #8ea5c5;
    font-size: .60rem;
    letter-spacing: .14em;
    font-weight: 900;
    margin-bottom: 4px;
}

.gs-title {
    color: #fff;
    font-size: 1.10rem;
    font-weight: 950;
    letter-spacing: -.03em;
}

.gs-note {
    color: #8297b6;
    font-size: .64rem;
    margin-top: 4px;
}

div.stButton > button {
    min-height: 36px !important;
    padding: 4px 7px !important;
    border-radius: 11px !important;
    font-size: .70rem !important;
    font-weight: 850 !important;
}

[data-testid="stDataFrame"] {
    font-size: .68rem !important;
}

/* ---------------- AVATAR CHOOSER ---------------- */

.avatar-choice {
    height: 174px;
    max-width: 165px;
    margin: 0 auto 5px auto;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,.08);
    background:
        radial-gradient(circle at 50% 45%, rgba(115,77,255,.18), transparent 56%),
        #09182d;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}

.avatar-choice svg {
    width: 118px;
    height: 160px;
}

/* ---------------- READY / WAITING ---------------- */

.ready-stage {
    width: 100%;
    max-width: 235px;
    height: 220px;
    margin: 0 auto;
    border-radius: 22px;
    border: 1px solid rgba(255,255,255,.08);
    background:
        radial-gradient(circle at 50% 44%, rgba(112,78,255,.20), transparent 56%),
        #09172b;
    display: flex;
    align-items: center;
    justify-content: center;
}

.ready-ring {
    width: 138px;
    height: 138px;
    border-radius: 999px;
    border: 14px solid rgba(122,96,255,.19);
    background: #070d1b;
    box-shadow: 0 12px 34px rgba(0,0,0,.25);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    animation: readyPulse 1.9s ease-in-out infinite;
}

.ready-icon {
    font-size: 2.8rem;
    line-height: 1;
}

.ready-text {
    margin-top: 6px;
    font-size: .66rem;
    color: #ccd4e7;
    letter-spacing: .10em;
}

@keyframes readyPulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.035); }
}

/* ---------------- LIVE AVATAR STAGE ---------------- */

.live-stage {
    width: 100%;
    max-width: 235px;
    height: 220px;
    margin: 0 auto;
    border-radius: 22px;
    border: 1px solid rgba(255,255,255,.08);
    background:
        radial-gradient(circle at 50% 48%, rgba(104,76,255,.20), transparent 55%),
        #09172b;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}

.live-pill {
    position: absolute;
    top: 8px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 8;
    min-width: 118px;
    text-align: center;
    padding: 5px 10px;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,.12);
    background: rgba(5,14,29,.86);
    color: white;
    font-size: .70rem;
    font-weight: 950;
    backdrop-filter: blur(8px);
}

.live-stage svg {
    width: 150px;
    height: 190px;
    margin-top: 15px;
    overflow: visible;
}

/* SVG movement */
.avatar-svg .body-group,
.avatar-svg .head-group,
.avatar-svg .left-arm,
.avatar-svg .right-arm,
.avatar-svg .left-leg,
.avatar-svg .right-leg,
.avatar-svg .left-forearm,
.avatar-svg .right-forearm {
    transform-box: fill-box;
}

.avatar-svg .left-arm,
.avatar-svg .right-arm,
.avatar-svg .left-leg,
.avatar-svg .right-leg {
    transform-origin: top center;
}

.avatar-svg .left-forearm,
.avatar-svg .right-forearm {
    transform-origin: top center;
}

.avatar-svg.activity-walking .left-arm {
    animation: walkArmA .62s ease-in-out infinite alternate;
}
.avatar-svg.activity-walking .right-arm {
    animation: walkArmB .62s ease-in-out infinite alternate;
}
.avatar-svg.activity-walking .left-leg {
    animation: walkLegA .62s ease-in-out infinite alternate;
}
.avatar-svg.activity-walking .right-leg {
    animation: walkLegB .62s ease-in-out infinite alternate;
}
.avatar-svg.activity-walking .body-group,
.avatar-svg.activity-walking .head-group {
    animation: walkBob .31s ease-in-out infinite alternate;
}

.avatar-svg.activity-running .left-arm {
    animation: runArmA .38s ease-in-out infinite alternate;
}
.avatar-svg.activity-running .right-arm {
    animation: runArmB .38s ease-in-out infinite alternate;
}
.avatar-svg.activity-running .left-leg {
    animation: runLegA .38s ease-in-out infinite alternate;
}
.avatar-svg.activity-running .right-leg {
    animation: runLegB .38s ease-in-out infinite alternate;
}
.avatar-svg.activity-running .body-group,
.avatar-svg.activity-running .head-group {
    animation: runBob .19s ease-in-out infinite alternate;
}

.avatar-svg.activity-rest .body-group,
.avatar-svg.activity-rest .head-group {
    animation: breathe .95s ease-in-out infinite alternate;
}

.avatar-svg.activity-squat .body-group,
.avatar-svg.activity-squat .head-group {
    animation: squatBody .95s ease-in-out infinite;
}
.avatar-svg.activity-squat .left-leg {
    animation: squatLegA .95s ease-in-out infinite;
}
.avatar-svg.activity-squat .right-leg {
    animation: squatLegB .95s ease-in-out infinite;
}

.avatar-svg.activity-lunge .body-group,
.avatar-svg.activity-lunge .head-group {
    animation: lungeBody .95s ease-in-out infinite alternate;
}
.avatar-svg.activity-lunge .left-leg {
    animation: lungeLegA .95s ease-in-out infinite alternate;
}
.avatar-svg.activity-lunge .right-leg {
    animation: lungeLegB .95s ease-in-out infinite alternate;
}

.avatar-svg.activity-jumping-jack .body-group,
.avatar-svg.activity-jumping-jack .head-group {
    animation: jackBody .68s ease-in-out infinite;
}
.avatar-svg.activity-jumping-jack .left-arm {
    animation: jackArmA .68s ease-in-out infinite;
}
.avatar-svg.activity-jumping-jack .right-arm {
    animation: jackArmB .68s ease-in-out infinite;
}
.avatar-svg.activity-jumping-jack .left-leg {
    animation: jackLegA .68s ease-in-out infinite;
}
.avatar-svg.activity-jumping-jack .right-leg {
    animation: jackLegB .68s ease-in-out infinite;
}

.avatar-svg.activity-bicep-curl .left-forearm,
.avatar-svg.activity-hammer-curl .left-forearm {
    animation: curlForearmA .75s ease-in-out infinite alternate;
}
.avatar-svg.activity-bicep-curl .right-forearm,
.avatar-svg.activity-hammer-curl .right-forearm {
    animation: curlForearmB .75s ease-in-out infinite alternate;
}

.avatar-svg.activity-shoulder-press .left-arm,
.avatar-svg.activity-front-raise .left-arm,
.avatar-svg.activity-lateral-raise .left-arm {
    animation: raiseArmA .82s ease-in-out infinite alternate;
}
.avatar-svg.activity-shoulder-press .right-arm,
.avatar-svg.activity-front-raise .right-arm,
.avatar-svg.activity-lateral-raise .right-arm {
    animation: raiseArmB .82s ease-in-out infinite alternate;
}

.avatar-svg.activity-workout .body-group,
.avatar-svg.activity-workout .head-group {
    animation: workoutBounce .58s ease-in-out infinite alternate;
}

@keyframes walkArmA { from { transform: rotate(24deg); } to { transform: rotate(-24deg); } }
@keyframes walkArmB { from { transform: rotate(-24deg); } to { transform: rotate(24deg); } }
@keyframes walkLegA { from { transform: rotate(-18deg); } to { transform: rotate(18deg); } }
@keyframes walkLegB { from { transform: rotate(18deg); } to { transform: rotate(-18deg); } }
@keyframes walkBob { from { transform: translateY(1px); } to { transform: translateY(-5px); } }

@keyframes runArmA { from { transform: rotate(45deg); } to { transform: rotate(-42deg); } }
@keyframes runArmB { from { transform: rotate(-42deg); } to { transform: rotate(45deg); } }
@keyframes runLegA { from { transform: rotate(-36deg); } to { transform: rotate(32deg); } }
@keyframes runLegB { from { transform: rotate(32deg); } to { transform: rotate(-36deg); } }
@keyframes runBob { from { transform: translateY(2px); } to { transform: translateY(-8px); } }

@keyframes breathe { from { transform: translateY(1px) scale(1); } to { transform: translateY(-2px) scale(1.015); } }

@keyframes squatBody {
    0%,100% { transform: translateY(0); }
    50% { transform: translateY(26px); }
}
@keyframes squatLegA {
    0%,100% { transform: rotate(0deg); }
    50% { transform: rotate(24deg); }
}
@keyframes squatLegB {
    0%,100% { transform: rotate(0deg); }
    50% { transform: rotate(-24deg); }
}

@keyframes lungeBody { from { transform: translate(-2px, 0); } to { transform: translate(7px, 16px); } }
@keyframes lungeLegA { from { transform: rotate(-8deg); } to { transform: rotate(28deg); } }
@keyframes lungeLegB { from { transform: rotate(8deg); } to { transform: rotate(-16deg); } }

@keyframes jackBody {
    0%,100% { transform: translateY(2px); }
    50% { transform: translateY(-18px); }
}
@keyframes jackArmA {
    0%,100% { transform: rotate(8deg); }
    50% { transform: rotate(-105deg); }
}
@keyframes jackArmB {
    0%,100% { transform: rotate(-8deg); }
    50% { transform: rotate(105deg); }
}
@keyframes jackLegA {
    0%,100% { transform: rotate(0deg); }
    50% { transform: rotate(28deg); }
}
@keyframes jackLegB {
    0%,100% { transform: rotate(0deg); }
    50% { transform: rotate(-28deg); }
}

@keyframes curlForearmA { from { transform: rotate(0deg); } to { transform: rotate(-105deg); } }
@keyframes curlForearmB { from { transform: rotate(0deg); } to { transform: rotate(105deg); } }

@keyframes raiseArmA { from { transform: rotate(8deg); } to { transform: rotate(-88deg); } }
@keyframes raiseArmB { from { transform: rotate(-8deg); } to { transform: rotate(88deg); } }

@keyframes workoutBounce { from { transform: translateY(2px); } to { transform: translateY(-6px); } }

.live-activity-name {
    text-align: center;
    margin-top: 5px;
    font-size: 1.15rem;
    font-weight: 950;
    letter-spacing: -.04em;
}

.live-confidence {
    text-align: center;
    color: #8ea3c1;
    font-size: .64rem;
    margin-top: 3px;
}

.metric-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
    margin-top: 7px;
}

.metric-box {
    background: rgba(255,255,255,.035);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 12px;
    padding: 6px 4px;
    text-align: center;
}

.metric-label {
    color: #8397b4;
    font-size: .53rem;
    font-weight: 850;
}

.metric-value {
    margin-top: 2px;
    color: white;
    font-size: .90rem;
    font-weight: 950;
}

@media(max-width: 480px) {
    .block-container {
        padding-left: .42rem !important;
        padding-right: .42rem !important;
    }

    .avatar-choice {
        max-width: 145px;
        height: 158px;
    }

    .avatar-choice svg {
        width: 108px;
        height: 145px;
    }

    .ready-stage,
    .live-stage {
        max-width: 205px;
        height: 192px;
    }

    .ready-ring {
        width: 122px;
        height: 122px;
        border-width: 12px;
    }

    .live-stage svg {
        width: 136px;
        height: 172px;
    }

    .metric-value {
        font-size: .82rem;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "avatar": None,
    "sensor_status": "OFF",
    "calibrated": False,
    "calibration": {
        "acc_std": 0.05,
        "jerk": 0.04,
        "gyro_mean": 1.0,
    },
    "workout_running": False,
    "workout_start": None,
    "current_activity": "READY",
    "current_confidence": None,
    "confidence_label": "",
    "raw_prediction": None,
    "motion_prediction": None,
    "recent": [],
    "activity_started": None,
    "history": [],
    "rep_totals": {},
    "set_totals": {},
    "last_rep_ts": {},
    "segment_rep_start": 0,
    "last_non_rest": None,
    "summary": None,
}

for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_ml():
    model = joblib.load(MODELS / "exercise_classifier.pkl")
    scaler = joblib.load(MODELS / "scaler.pkl")
    encoder = joblib.load(MODELS / "label_encoder.pkl")

    metadata = {}
    metadata_path = MODELS / "model_metadata.json"

    if metadata_path.exists():
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

    feature_names = (
        metadata.get("feature_names")
        or get_feature_names()
    )

    return model, scaler, encoder, feature_names


# ============================================================
# MOTION HEURISTIC
# ============================================================

def motion_features(records: list[dict[str, Any]]) -> dict[str, float]:
    ax = np.array([float(r.get("accel_x", 0.0)) for r in records])
    ay = np.array([float(r.get("accel_y", 0.0)) for r in records])
    az = np.array([float(r.get("accel_z", 9.81)) for r in records])

    ga = np.array([float(r.get("gyro_alpha", 0.0)) for r in records])
    gb = np.array([float(r.get("gyro_beta", 0.0)) for r in records])
    gg = np.array([float(r.get("gyro_gamma", 0.0)) for r in records])

    acc_mag = np.sqrt(ax * ax + ay * ay + az * az)
    gyro_mag = np.sqrt(ga * ga + gb * gb + gg * gg)

    acc_std = float(np.std(acc_mag))
    jerk = (
        float(np.mean(np.abs(np.diff(acc_mag))))
        if len(acc_mag) > 1
        else 0.0
    )
    gyro_mean = float(np.mean(np.abs(gyro_mag)))
    axis_std = float(np.std(ax) + np.std(ay) + np.std(az))

    return {
        "acc_std": acc_std,
        "jerk": jerk,
        "gyro_mean": gyro_mean,
        "axis_std": axis_std,
    }


def infer_motion_activity(
    records: list[dict[str, Any]],
) -> str:
    f = motion_features(records)
    baseline = st.session_state.calibration or {}

    base_acc = float(baseline.get("acc_std", 0.05))
    base_jerk = float(baseline.get("jerk", 0.04))
    base_gyro = float(baseline.get("gyro_mean", 1.0))

    rest_acc_limit = max(0.17, base_acc * 3.2 + 0.06)
    rest_jerk_limit = max(0.12, base_jerk * 3.2 + 0.05)
    rest_gyro_limit = max(9.0, base_gyro * 3.0 + 3.0)

    if (
        f["acc_std"] <= rest_acc_limit
        and f["jerk"] <= rest_jerk_limit
        and f["gyro_mean"] <= rest_gyro_limit
    ):
        return "REST"

    if (
        f["acc_std"] >= 1.55
        or f["jerk"] >= 1.15
        or f["gyro_mean"] >= 55.0
        or f["axis_std"] >= 5.0
    ):
        return "RUNNING"

    return "WALKING"


# ============================================================
# ML + MOTION
# ============================================================

def predict_activity(
    records: list[dict[str, Any]],
):
    model, scaler, encoder, feature_names = load_ml()

    features = extract_features_from_records(records)

    x = np.array(
        [[float(features.get(name, 0.0)) for name in feature_names]],
        dtype=float,
    )

    x = scaler.transform(x)

    confidence = None
    probabilities: dict[str, float] = {}
    raw_prediction = "WORKOUT"

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)[0]

        class_ids = getattr(
            model,
            "classes_",
            np.arange(len(probs)),
        )

        labels = []

        for class_id in class_ids:
            label = str(
                encoder.inverse_transform([int(class_id)])[0]
            )
            label = label.upper().replace("_", " ")
            labels.append(label)

        best_index = int(np.argmax(probs))
        raw_prediction = labels[best_index]
        confidence = float(probs[best_index])

        probabilities = {
            label: float(probability)
            for label, probability in zip(labels, probs)
        }

    else:
        encoded = int(model.predict(x)[0])
        raw_prediction = str(
            encoder.inverse_transform([encoded])[0]
        )
        raw_prediction = raw_prediction.upper().replace("_", " ")

    motion_prediction = infer_motion_activity(records)

    # Still phone -> REST.
    if motion_prediction == "REST":
        return (
            "REST",
            probabilities.get("REST", confidence),
            "Motion / REST",
            raw_prediction,
            motion_prediction,
        )

    # Let model walking/running pass at modest confidence.
    if (
        raw_prediction in {"WALKING", "RUNNING"}
        and confidence is not None
        and confidence >= WALK_RUN_MIN_CONF
    ):
        return (
            raw_prediction,
            confidence,
            "ML + motion",
            raw_prediction,
            motion_prediction,
        )

    # Gym exercises need stronger confidence.
    if (
        raw_prediction in REPETITIVE
        and confidence is not None
        and confidence >= EXERCISE_MIN_CONF
    ):
        return (
            raw_prediction,
            confidence,
            "ML confidence",
            raw_prediction,
            motion_prediction,
        )

    if (
        raw_prediction in KNOWN_ACTIVITIES
        and confidence is not None
        and confidence >= GENERAL_MIN_CONF
    ):
        return (
            raw_prediction,
            confidence,
            "ML confidence",
            raw_prediction,
            motion_prediction,
        )

    # If ML is unsure, never force WORKOUT.
    # Use the real phone motion instead.
    return (
        motion_prediction,
        confidence,
        "Motion fallback",
        raw_prediction,
        motion_prediction,
    )


# ============================================================
# SMOOTHING
# ============================================================

def smooth_prediction(label: str) -> str:
    recent = list(st.session_state.recent)
    recent.append(label)
    recent = recent[-SMOOTH_WINDOW:]
    st.session_state.recent = recent

    if len(recent) < SMOOTH_WINDOW:
        return label

    winner, count = Counter(recent).most_common(1)[0]

    if count >= 2:
        return winner

    return recent[-1]


# ============================================================
# REP COUNTER
# ============================================================

def find_rep_peaks(records):
    if len(records) < 12:
        return []

    ax = np.array([float(r.get("accel_x", 0.0)) for r in records])
    ay = np.array([float(r.get("accel_y", 0.0)) for r in records])
    az = np.array([float(r.get("accel_z", 9.81)) for r in records])

    magnitude = np.sqrt(ax * ax + ay * ay + az * az)
    dynamic = np.abs(magnitude - np.median(magnitude))

    threshold = max(
        0.65,
        float(
            np.mean(dynamic)
            + 0.75 * np.std(dynamic)
        ),
    )

    peaks = []
    last_index = -999
    min_distance = max(6, len(records) // 10)

    for index in range(1, len(dynamic) - 1):
        if (
            dynamic[index] > threshold
            and dynamic[index] >= dynamic[index - 1]
            and dynamic[index] > dynamic[index + 1]
            and index - last_index >= min_distance
        ):
            timestamp = int(
                records[index].get("timestamp_ms", 0) or 0
            )
            peaks.append(timestamp)
            last_index = index

    return peaks


def update_reps(records, activity):
    if activity not in REPETITIVE:
        return

    st.session_state.rep_totals.setdefault(activity, 0)

    last_timestamp = int(
        st.session_state.last_rep_ts.get(activity, 0)
    )

    for timestamp in find_rep_peaks(records):
        if (
            timestamp > 0
            and timestamp - last_timestamp >= 550
        ):
            st.session_state.rep_totals[activity] += 1
            last_timestamp = timestamp

    st.session_state.last_rep_ts[activity] = last_timestamp


# ============================================================
# HISTORY
# ============================================================

def start_activity(activity):
    st.session_state.current_activity = activity
    st.session_state.activity_started = time.time()

    if activity in REPETITIVE:
        if st.session_state.last_non_rest != activity:
            st.session_state.set_totals[activity] = (
                st.session_state.set_totals.get(activity, 0)
                + 1
            )

        st.session_state.segment_rep_start = (
            st.session_state.rep_totals.get(activity, 0)
        )
    else:
        st.session_state.segment_rep_start = 0


def close_activity():
    activity = st.session_state.current_activity
    started = st.session_state.activity_started

    if (
        activity in (None, "READY")
        or started is None
    ):
        return

    end_time = time.time()
    reps = 0

    if activity in REPETITIVE:
        reps = max(
            0,
            st.session_state.rep_totals.get(activity, 0)
            - st.session_state.segment_rep_start,
        )

    st.session_state.history.append(
        {
            "activity": activity,
            "start": started,
            "end": end_time,
            "duration": max(0, end_time - started),
            "reps": reps,
            "confidence": st.session_state.current_confidence,
        }
    )

    if activity != "REST":
        st.session_state.last_non_rest = activity


def change_activity(new_activity):
    old_activity = st.session_state.current_activity

    if old_activity == "READY":
        start_activity(new_activity)
        return

    if old_activity == new_activity:
        return

    close_activity()

    if new_activity == "REST":
        st.session_state.last_non_rest = None

    start_activity(new_activity)


# ============================================================
# WORKOUT SESSION
# ============================================================

def start_workout():
    st.session_state.workout_running = True
    st.session_state.workout_start = time.time()
    st.session_state.current_activity = "READY"
    st.session_state.current_confidence = None
    st.session_state.confidence_label = ""
    st.session_state.raw_prediction = None
    st.session_state.motion_prediction = None
    st.session_state.activity_started = None
    st.session_state.history = []
    st.session_state.rep_totals = {}
    st.session_state.set_totals = {}
    st.session_state.last_rep_ts = {}
    st.session_state.segment_rep_start = 0
    st.session_state.recent = []
    st.session_state.last_non_rest = None
    st.session_state.summary = None


def end_workout():
    if not st.session_state.workout_running:
        return

    close_activity()

    end_time = time.time()
    start_time = st.session_state.workout_start or end_time

    active_time = sum(
        item["duration"]
        for item in st.session_state.history
        if item["activity"] != "REST"
    )

    activities = sorted(
        {
            item["activity"]
            for item in st.session_state.history
            if item["activity"] != "REST"
        }
    )

    st.session_state.summary = {
        "duration": max(0, end_time - start_time),
        "active": active_time,
        "reps": sum(st.session_state.rep_totals.values()),
        "sets": sum(st.session_state.set_totals.values()),
        "activities": activities,
    }

    st.session_state.workout_running = False
    st.session_state.current_activity = "READY"
    st.session_state.activity_started = None
    st.session_state.recent = []


# ============================================================
# DISPLAY HELPERS
# ============================================================

def format_duration(seconds):
    seconds = int(max(0, seconds))
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


def current_elapsed():
    if (
        st.session_state.workout_running
        and st.session_state.activity_started
    ):
        return max(
            0,
            int(
                time.time()
                - st.session_state.activity_started
            ),
        )
    return 0


def avatar_svg(gender: str, activity: str = "READY") -> str:
    female = gender == "female"

    skin = "#f2b58a"
    hair = "#6a3b27" if female else "#3a261e"
    shirt = "#f04f9a" if female else "#2f8bff"
    shorts = "#202a3c"
    shoes = "#ff5da8" if female else "#45a6ff"

    activity_class = (
        "activity-"
        + activity.lower().replace(" ", "-")
    )

    # Dumbbells visible for strength exercises.
    strength = activity in {
        "BICEP CURL",
        "HAMMER CURL",
        "SHOULDER PRESS",
        "FRONT RAISE",
        "LATERAL RAISE",
    }

    dumbbell_display = "inline" if strength else "none"

    # A slightly different hairstyle/torso for female/male.
    hair_extra = (
        '<path d="M80 47 C62 54 59 91 70 108 C76 90 84 72 93 58 Z" fill="#6a3b27"/>'
        if female
        else ""
    )

    waist_shape = (
        '<path d="M88 112 L112 112 L116 150 L84 150 Z" fill="#f04f9a"/>'
        if female
        else '<rect x="84" y="108" width="32" height="44" rx="8" fill="#2f8bff"/>'
    )

    return f"""
<svg
    class="avatar-svg {activity_class}"
    viewBox="0 0 200 280"
    xmlns="http://www.w3.org/2000/svg"
>
    <defs>
        <filter id="shadow">
            <feDropShadow dx="0" dy="5" stdDeviation="5" flood-opacity=".28"/>
        </filter>
    </defs>

    <g filter="url(#shadow)">
        <g class="head-group">
            {hair_extra}
            <circle cx="100" cy="50" r="24" fill="{skin}"/>
            <path
                d="M76 49
                   C78 18 124 15 126 50
                   C118 37 110 32 98 32
                   C88 32 81 38 76 49 Z"
                fill="{hair}"
            />
            <circle cx="92" cy="51" r="2" fill="#35251f"/>
            <circle cx="108" cy="51" r="2" fill="#35251f"/>
            <path d="M94 61 Q100 65 106 61" fill="none" stroke="#9f5d50" stroke-width="2"/>
        </g>

        <g class="body-group">
            {waist_shape}
            <rect x="88" y="148" width="24" height="14" rx="5" fill="{shorts}"/>

            <g class="left-arm">
                <rect x="79" y="111" width="12" height="46" rx="6" fill="{skin}"/>
                <g class="left-forearm">
                    <rect x="79" y="150" width="12" height="42" rx="6" fill="{skin}"/>
                    <g style="display:{dumbbell_display}">
                        <rect x="70" y="184" width="30" height="7" rx="3" fill="#242b37"/>
                        <circle cx="72" cy="187.5" r="8" fill="#111722"/>
                        <circle cx="98" cy="187.5" r="8" fill="#111722"/>
                    </g>
                </g>
            </g>

            <g class="right-arm">
                <rect x="109" y="111" width="12" height="46" rx="6" fill="{skin}"/>
                <g class="right-forearm">
                    <rect x="109" y="150" width="12" height="42" rx="6" fill="{skin}"/>
                    <g style="display:{dumbbell_display}">
                        <rect x="100" y="184" width="30" height="7" rx="3" fill="#242b37"/>
                        <circle cx="102" cy="187.5" r="8" fill="#111722"/>
                        <circle cx="128" cy="187.5" r="8" fill="#111722"/>
                    </g>
                </g>
            </g>

            <g class="left-leg">
                <rect x="87" y="159" width="11" height="66" rx="6" fill="{skin}"/>
                <rect x="84" y="216" width="17" height="34" rx="7" fill="#171d28"/>
                <ellipse cx="91" cy="254" rx="14" ry="6" fill="{shoes}"/>
            </g>

            <g class="right-leg">
                <rect x="102" y="159" width="11" height="66" rx="6" fill="{skin}"/>
                <rect x="99" y="216" width="17" height="34" rx="7" fill="#171d28"/>
                <ellipse cx="109" cy="254" rx="14" ry="6" fill="{shoes}"/>
            </g>
        </g>
    </g>
</svg>
"""


def show_avatar_choice(gender: str):
    st.markdown(
        f"""
<div class="avatar-choice">
    {avatar_svg(gender, "REST")}
</div>
""",
        unsafe_allow_html=True,
    )


def show_ready_stage():
    st.markdown(
        """
<div class="ready-stage">
    <div class="ready-ring">
        <div class="ready-icon">⌛</div>
        <div class="ready-text">AI DETECTED</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def show_live_stage():
    activity = st.session_state.current_activity
    gender = st.session_state.avatar or "female"

    if activity == "READY":
        show_ready_stage()
        return

    st.markdown(
        f"""
<div class="live-stage">
    <div class="live-pill">
        {ACTIVITY_EMOJI.get(activity, "🔥")} {activity}
    </div>

    {avatar_svg(gender, activity)}
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# SENSOR COMPONENT
# ============================================================

SENSOR_HTML = r"""
<div class="sensor-card">

<div class="sensor-title-row">
    <div>
        <b>MOTION SENSOR</b>
        <span id="message">Tap Enable Sensor</span>
    </div>

    <span id="badge">● OFF</span>
</div>

<div class="button-grid">
    <button id="enable">Enable Sensor</button>
    <button id="start" disabled>Start Workout</button>
    <button id="end" disabled>End Workout</button>
</div>

<div class="sensor-grid">

    <div class="sensor-box">
        <div class="sensor-box-title">
            ACCELEROMETER
            <span id="acc-status">OFF</span>
        </div>

        <canvas id="acc-canvas"></canvas>

        <div class="sensor-values">
            <span>X <b id="ax">0.00</b></span>
            <span>Y <b id="ay">0.00</b></span>
            <span>Z <b id="az">0.00</b></span>
        </div>
    </div>

    <div class="sensor-box">
        <div class="sensor-box-title">
            GYROSCOPE
            <span id="gyro-status">OFF</span>
        </div>

        <canvas id="gyro-canvas"></canvas>

        <div class="sensor-values">
            <span>α <b id="ga">0.00</b></span>
            <span>β <b id="gb">0.00</b></span>
            <span>γ <b id="gg">0.00</b></span>
        </div>
    </div>

</div>

<div class="sensor-note">
    Phone motion only • Not ECG / heart-rate data
</div>

</div>
"""

SENSOR_CSS = r"""
.sensor-card {
    color: #fff;
    border: 1px solid rgba(255,255,255,.08);
    background: linear-gradient(145deg, #0b1930, #071426);
    border-radius: 16px;
    padding: 9px;
    font-family: Inter, system-ui;
}

.sensor-title-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.sensor-title-row b {
    display: block;
    font-size: 9px;
    letter-spacing: .13em;
    color: #8ba0bd;
}

.sensor-title-row div span {
    display: block;
    font-size: 9px;
    color: #9db0ca;
    margin-top: 2px;
}

#badge {
    font-size: 9px;
    font-weight: 900;
    border: 1px solid #35465e;
    border-radius: 999px;
    padding: 4px 7px;
}

.button-grid {
    display: grid;
    grid-template-columns: 1.15fr 1fr 1fr;
    gap: 5px;
    margin-top: 8px;
}

button {
    min-height: 34px;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,.10);
    background: #142540;
    color: #fff;
    font-size: 9px;
    font-weight: 850;
    padding: 4px;
}

#enable {
    background: linear-gradient(90deg, #ff4e9f, #8e48ff);
    border: none;
}

#start {
    background: linear-gradient(90deg, #22b879, #24d6a0);
    border: none;
}

#end {
    background: linear-gradient(90deg, #d43d68, #ff536d);
    border: none;
}

button:disabled {
    opacity: .34;
}

.sensor-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
    margin-top: 7px;
}

.sensor-box {
    border: 1px solid rgba(255,255,255,.06);
    background: #06111f;
    border-radius: 12px;
    padding: 6px;
}

.sensor-box-title {
    display: flex;
    justify-content: space-between;
    font-size: 8px;
    color: #9db0ca;
    font-weight: 850;
}

.sensor-box-title span {
    color: #50e2a4;
}

canvas {
    display: block;
    width: 100%;
    height: 52px;
    margin-top: 2px;
}

.sensor-values {
    display: flex;
    justify-content: space-between;
    gap: 3px;
    font-size: 8px;
    color: #8297b7;
}

.sensor-values b {
    color: #fff;
}

.sensor-note {
    font-size: 8px;
    color: #657c9b;
    text-align: center;
    margin-top: 4px;
}

@media(max-width:420px) {
    button {
        font-size: 8px;
        min-height: 32px;
    }

    canvas {
        height: 46px;
    }
}
"""

SENSOR_JS = r"""
export default function(component) {

const {
    parentElement,
    setTriggerValue,
    data
} = component;

if (!parentElement.__gymSensePreviewState) {

    const S = {
        enabled: false,
        calibrated: false,
        calibrating: false,
        workout: false,
        buffer: [],
        calibrationSamples: [],
        lastEmit: 0,
        accHistory: [],
        gyroHistory: [],
        listener: null
    };

    parentElement.__gymSensePreviewState = S;

    const q = selector =>
        parentElement.querySelector(selector);

    const enableButton = q('#enable');
    const startButton = q('#start');
    const endButton = q('#end');

    const message = q('#message');
    const badge = q('#badge');

    const accStatus = q('#acc-status');
    const gyroStatus = q('#gyro-status');

    const ax = q('#ax');
    const ay = q('#ay');
    const az = q('#az');

    const ga = q('#ga');
    const gb = q('#gb');
    const gg = q('#gg');

    function setMessage(text) {
        message.textContent = text;
    }

    function setLive(live) {
        badge.textContent =
            live ? '● LIVE' : '● OFF';

        badge.style.color =
            live ? '#56e5a8' : '#ffffff';

        accStatus.textContent =
            live ? 'LIVE' : 'OFF';

        gyroStatus.textContent =
            live ? 'LIVE' : 'OFF';
    }

    function addHistory(list, value) {
        list.push(value);

        if (list.length > 55) {
            list.splice(
                0,
                list.length - 55
            );
        }
    }

    function std(values) {
        if (!values.length) return 0;

        const mean =
            values.reduce(
                (a, b) => a + b,
                0
            ) / values.length;

        const variance =
            values.reduce(
                (sum, value) =>
                    sum +
                    Math.pow(
                        value - mean,
                        2
                    ),
                0
            ) / values.length;

        return Math.sqrt(variance);
    }

    function calibrationStats(samples) {

        if (!samples.length) {
            return {
                acc_std: 0.05,
                jerk: 0.04,
                gyro_mean: 1.0
            };
        }

        const accMag =
            samples.map(sample =>
                Math.sqrt(
                    sample.accel_x * sample.accel_x +
                    sample.accel_y * sample.accel_y +
                    sample.accel_z * sample.accel_z
                )
            );

        const gyroMag =
            samples.map(sample =>
                Math.sqrt(
                    sample.gyro_alpha * sample.gyro_alpha +
                    sample.gyro_beta * sample.gyro_beta +
                    sample.gyro_gamma * sample.gyro_gamma
                )
            );

        const jerkValues = [];

        for (
            let index = 1;
            index < accMag.length;
            index++
        ) {
            jerkValues.push(
                Math.abs(
                    accMag[index]
                    -
                    accMag[index - 1]
                )
            );
        }

        const gyroMean =
            gyroMag.length
            ?
            gyroMag.reduce(
                (a, b) => a + Math.abs(b),
                0
            ) / gyroMag.length
            :
            0;

        const jerkMean =
            jerkValues.length
            ?
            jerkValues.reduce(
                (a, b) => a + b,
                0
            ) / jerkValues.length
            :
            0;

        return {
            acc_std: std(accMag),
            jerk: jerkMean,
            gyro_mean: gyroMean
        };
    }

    function drawGraph(
        canvas,
        rows,
        colors
    ) {

        const rect =
            canvas.getBoundingClientRect();

        const ratio =
            window.devicePixelRatio || 1;

        const width =
            Math.max(
                120,
                rect.width
            );

        const height =
            Math.max(
                44,
                rect.height
            );

        canvas.width =
            width * ratio;

        canvas.height =
            height * ratio;

        const ctx =
            canvas.getContext('2d');

        ctx.scale(
            ratio,
            ratio
        );

        ctx.clearRect(
            0,
            0,
            width,
            height
        );

        ctx.strokeStyle =
            'rgba(120,150,190,.12)';

        for (
            let index = 0;
            index <= 3;
            index++
        ) {
            const y =
                height * index / 3;

            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }

        if (rows.length < 2) {
            return;
        }

        const flattened =
            rows
            .flat()
            .filter(Number.isFinite);

        const maxValue =
            Math.max(
                1,
                ...flattened.map(
                    value =>
                        Math.abs(value)
                )
            );

        const middle =
            height / 2;

        for (
            let axis = 0;
            axis < 3;
            axis++
        ) {

            ctx.strokeStyle =
                colors[axis];

            ctx.lineWidth = 1.4;
            ctx.beginPath();

            rows.forEach(
                (row, index) => {

                    const x =
                        index
                        *
                        width
                        /
                        (
                            rows.length - 1
                        );

                    const y =
                        middle
                        -
                        (
                            row[axis]
                            /
                            maxValue
                        )
                        *
                        (
                            height * .40
                        );

                    if (index === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                }
            );

            ctx.stroke();
        }
    }

    function animationLoop() {

        drawGraph(
            q('#acc-canvas'),
            S.accHistory,
            [
                '#ff5d79',
                '#42b5ff',
                '#42e39c'
            ]
        );

        drawGraph(
            q('#gyro-canvas'),
            S.gyroHistory,
            [
                '#ff5bc3',
                '#42e5ef',
                '#ffd45a'
            ]
        );

        requestAnimationFrame(
            animationLoop
        );
    }

    animationLoop();

    async function enableSensors() {

        try {

            if (
                !(
                    'DeviceMotionEvent'
                    in window
                )
            ) {

                setMessage(
                    'Motion sensor unavailable'
                );

                setTriggerValue(
                    'sensor_status',
                    {
                        status: 'UNSUPPORTED',
                        ts: Date.now()
                    }
                );

                return;
            }

            if (
                typeof
                DeviceMotionEvent.requestPermission
                ===
                'function'
            ) {

                const permission =
                    await
                    DeviceMotionEvent
                    .requestPermission();

                if (
                    permission !==
                    'granted'
                ) {

                    setMessage(
                        'Sensor permission denied'
                    );

                    setTriggerValue(
                        'sensor_status',
                        {
                            status: 'DENIED',
                            ts: Date.now()
                        }
                    );

                    return;
                }
            }

            if (!S.listener) {

                S.listener =
                    event => {

                        const accel =
                            event.accelerationIncludingGravity
                            ||
                            event.acceleration
                            ||
                            {};

                        const rotation =
                            event.rotationRate
                            ||
                            {};

                        const sample = {
                            accel_x:
                                Number(
                                    accel.x || 0
                                ),
                            accel_y:
                                Number(
                                    accel.y || 0
                                ),
                            accel_z:
                                Number(
                                    accel.z || 0
                                ),
                            gyro_alpha:
                                Number(
                                    rotation.alpha || 0
                                ),
                            gyro_beta:
                                Number(
                                    rotation.beta || 0
                                ),
                            gyro_gamma:
                                Number(
                                    rotation.gamma || 0
                                ),
                            timestamp_ms:
                                Date.now()
                        };

                        ax.textContent =
                            sample.accel_x.toFixed(2);

                        ay.textContent =
                            sample.accel_y.toFixed(2);

                        az.textContent =
                            sample.accel_z.toFixed(2);

                        ga.textContent =
                            sample.gyro_alpha.toFixed(2);

                        gb.textContent =
                            sample.gyro_beta.toFixed(2);

                        gg.textContent =
                            sample.gyro_gamma.toFixed(2);

                        addHistory(
                            S.accHistory,
                            [
                                sample.accel_x,
                                sample.accel_y,
                                sample.accel_z
                            ]
                        );

                        addHistory(
                            S.gyroHistory,
                            [
                                sample.gyro_alpha,
                                sample.gyro_beta,
                                sample.gyro_gamma
                            ]
                        );

                        if (S.calibrating) {

                            S.calibrationSamples.push(
                                sample
                            );

                            if (
                                S.calibrationSamples.length > 180
                            ) {
                                S.calibrationSamples =
                                    S.calibrationSamples.slice(
                                        -180
                                    );
                            }
                        }

                        if (S.workout) {

                            S.buffer.push(sample);

                            const now =
                                Date.now();

                            if (
                                S.buffer.length >= 24
                                &&
                                now - S.lastEmit >= 1450
                            ) {

                                setTriggerValue(
                                    'window',
                                    {
                                        samples:
                                            S.buffer.slice(-120),
                                        ts: now
                                    }
                                );

                                S.buffer =
                                    S.buffer.slice(-8);

                                S.lastEmit =
                                    now;
                            }
                        }
                    };

                window.addEventListener(
                    'devicemotion',
                    S.listener,
                    {
                        passive: true
                    }
                );
            }

            S.enabled = true;
            S.calibrated = false;
            S.calibrating = true;
            S.calibrationSamples = [];

            setLive(true);

            enableButton.disabled = true;
            enableButton.textContent =
                '✓ Sensor Enabled';

            setMessage(
                'Keep phone still... 3'
            );

            let seconds = 3;

            const timer =
                setInterval(
                    () => {

                        seconds--;

                        if (seconds > 0) {

                            setMessage(
                                'Keep phone still... '
                                +
                                seconds
                            );

                        } else {

                            clearInterval(timer);

                            S.calibrating = false;
                            S.calibrated = true;

                            const stats =
                                calibrationStats(
                                    S.calibrationSamples
                                );

                            startButton.disabled =
                                false;

                            setMessage(
                                'Sensor ready'
                            );

                            setTriggerValue(
                                'calibration',
                                {
                                    ...stats,
                                    ts: Date.now()
                                }
                            );

                            setTriggerValue(
                                'sensor_status',
                                {
                                    status: 'READY',
                                    ts: Date.now()
                                }
                            );
                        }

                    },
                    1000
                );

            setTriggerValue(
                'sensor_status',
                {
                    status: 'LIVE',
                    ts: Date.now()
                }
            );

        }

        catch(error) {

            setMessage(
                'Sensor error'
            );

            setTriggerValue(
                'sensor_status',
                {
                    status: 'ERROR',
                    ts: Date.now()
                }
            );
        }
    }

    function startWorkout() {

        if (
            !S.enabled
            ||
            !S.calibrated
        ) {
            return;
        }

        S.workout = true;
        S.buffer = [];
        S.lastEmit = 0;

        startButton.disabled = true;
        endButton.disabled = false;

        setMessage(
            'Workout running'
        );

        setTriggerValue(
            'session_event',
            {
                event: 'START',
                ts: Date.now()
            }
        );
    }

    function endWorkout() {

        if (!S.workout) {
            return;
        }

        S.workout = false;

        endButton.disabled = true;
        startButton.disabled = false;

        setMessage(
            'Workout completed'
        );

        setTriggerValue(
            'session_event',
            {
                event: 'END',
                ts: Date.now()
            }
        );
    }

    enableButton.addEventListener(
        'click',
        enableSensors
    );

    startButton.addEventListener(
        'click',
        startWorkout
    );

    endButton.addEventListener(
        'click',
        endWorkout
    );
}

const S =
    parentElement
    .__gymSensePreviewState;

if (
    data?.workoutRunning === false
    &&
    S.workout
) {
    S.workout = false;
}

return () => {};
}
"""

sensor_component = st.components.v2.component(
    "gymsense_preview_sensor",
    html=SENSOR_HTML,
    css=SENSOR_CSS,
    js=SENSOR_JS,
)


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns(
    [2.6, 1],
    vertical_alignment="center",
)

with header_left:
    st.markdown(
        f"""
<div class="gs-brand">
    GymSense <span>AI</span>
</div>

<div class="gs-sub">
    Phone Motion Workout Recognition
</div>

<div style="margin-top:6px;">
    <span class="gs-status">
        ● {st.session_state.sensor_status}
    </span>
</div>
""",
        unsafe_allow_html=True,
    )

with header_right:
    if st.session_state.avatar is not None:
        switch = st.button(
            "↔ Avatar",
            use_container_width=True,
            disabled=st.session_state.workout_running,
        )

        if switch:
            st.session_state.avatar = None
            st.session_state.current_activity = "READY"
            st.session_state.current_confidence = None
            st.session_state.confidence_label = ""
            st.session_state.raw_prediction = None
            st.session_state.motion_prediction = None
            st.session_state.recent = []
            st.rerun()


# ============================================================
# CHOOSE AVATAR
# ============================================================

if st.session_state.avatar is None:

    st.markdown(
        """
<div class="gs-card">
    <div class="gs-kicker">
        CHOOSE AVATAR
    </div>

    <div class="gs-title">
        Select Your Workout Avatar
    </div>

    <div class="gs-note">
        Full-body avatar selection.
        Avatar choice changes only the visual character.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    female_col, male_col = st.columns(
        2,
        gap="small",
    )

    with female_col:
        show_avatar_choice("female")

        if st.button(
            "Female",
            use_container_width=True,
            type="primary",
            key="female_avatar",
        ):
            st.session_state.avatar = "female"
            st.rerun()

    with male_col:
        show_avatar_choice("male")

        if st.button(
            "Male",
            use_container_width=True,
            key="male_avatar",
        ):
            st.session_state.avatar = "male"
            st.rerun()

    st.stop()


# ============================================================
# LIVE WORKOUT
# ============================================================

st.markdown(
    '<div class="gs-kicker">LIVE WORKOUT</div>',
    unsafe_allow_html=True,
)

show_live_stage()

activity = st.session_state.current_activity
confidence = st.session_state.current_confidence

st.markdown(
    f"""
<div class="live-activity-name">
    {ACTIVITY_EMOJI.get(activity, "🔥")}
    {activity}
</div>
""",
    unsafe_allow_html=True,
)

if confidence is not None:
    st.markdown(
        f"""
<div class="live-confidence">
    Confidence: {confidence * 100:.1f}%
</div>
""",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
<div class="live-confidence">
    Waiting for sensor prediction
</div>
""",
        unsafe_allow_html=True,
    )

elapsed = current_elapsed()

current_reps = (
    st.session_state.rep_totals.get(
        activity,
        0,
    )
    if activity in REPETITIVE
    else 0
)

current_sets = (
    st.session_state.set_totals.get(
        activity,
        0,
    )
    if activity in REPETITIVE
    else 0
)

st.markdown(
    f"""
<div class="metric-row">

    <div class="metric-box">
        <div class="metric-label">
            DURATION
        </div>
        <div class="metric-value">
            {format_duration(elapsed)}
        </div>
    </div>

    <div class="metric-box">
        <div class="metric-label">
            REPS
        </div>
        <div class="metric-value">
            {current_reps if activity in REPETITIVE else "—"}
        </div>
    </div>

    <div class="metric-box">
        <div class="metric-label">
            SETS
        </div>
        <div class="metric-value">
            {current_sets if activity in REPETITIVE else "—"}
        </div>
    </div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SENSOR PANEL
# ============================================================

sensor_result = sensor_component(
    data={
        "workoutRunning":
            bool(
                st.session_state.workout_running
            ),
        "prediction":
            st.session_state.current_activity,
    },
    key="gym-preview-sensor",
    on_sensor_status_change=lambda: None,
    on_calibration_change=lambda: None,
    on_session_event_change=lambda: None,
    on_window_change=lambda: None,
)


# ============================================================
# SENSOR STATUS
# ============================================================

sensor_status_event = getattr(
    sensor_result,
    "sensor_status",
    None,
)

if sensor_status_event:

    status = str(
        sensor_status_event.get(
            "status",
            "",
        )
    ).upper()

    if status == "LIVE":
        st.session_state.sensor_status = "LIVE"

    elif status == "READY":
        st.session_state.sensor_status = "READY"
        st.session_state.calibrated = True

    elif status:
        st.session_state.sensor_status = status


# ============================================================
# CALIBRATION RESULT
# ============================================================

calibration_event = getattr(
    sensor_result,
    "calibration",
    None,
)

if calibration_event:

    st.session_state.calibration = {
        "acc_std":
            float(
                calibration_event.get(
                    "acc_std",
                    0.05,
                )
            ),
        "jerk":
            float(
                calibration_event.get(
                    "jerk",
                    0.04,
                )
            ),
        "gyro_mean":
            float(
                calibration_event.get(
                    "gyro_mean",
                    1.0,
                )
            ),
    }


# ============================================================
# START / END EVENT
# ============================================================

session_event = getattr(
    sensor_result,
    "session_event",
    None,
)

if session_event:

    event = str(
        session_event.get(
            "event",
            "",
        )
    ).upper()

    if (
        event == "START"
        and
        not st.session_state.workout_running
    ):
        start_workout()
        st.rerun()

    elif (
        event == "END"
        and
        st.session_state.workout_running
    ):
        end_workout()
        st.rerun()


# ============================================================
# SENSOR WINDOW -> PREDICTION
# ============================================================

sensor_window = getattr(
    sensor_result,
    "window",
    None,
)

if (
    sensor_window
    and st.session_state.workout_running
    and isinstance(
        sensor_window.get("samples"),
        list,
    )
):

    records = sensor_window["samples"]

    if len(records) >= 12:

        (
            predicted_activity,
            predicted_confidence,
            confidence_label,
            raw_prediction,
            motion_prediction,
        ) = predict_activity(records)

        predicted_activity = smooth_prediction(
            predicted_activity
        )

        st.session_state.current_confidence = (
            predicted_confidence
        )

        st.session_state.confidence_label = (
            confidence_label
        )

        st.session_state.raw_prediction = (
            raw_prediction
        )

        st.session_state.motion_prediction = (
            motion_prediction
        )

        change_activity(
            predicted_activity
        )

        update_reps(
            records,
            predicted_activity,
        )

        st.rerun()


# ============================================================
# SMALL DEBUG LINE
# ============================================================

if (
    st.session_state.workout_running
    and st.session_state.raw_prediction
):
    st.caption(
        "ML: "
        f"{st.session_state.raw_prediction}"
        " • Motion: "
        f"{st.session_state.motion_prediction}"
        " • Showing: "
        f"{st.session_state.current_activity}"
    )


# ============================================================
# HISTORY
# ============================================================

st.markdown(
    '<div class="gs-kicker" style="margin-top:10px;">'
    'ACTIVITY HISTORY'
    '</div>',
    unsafe_allow_html=True,
)

history_rows = []

if (
    st.session_state.workout_running
    and st.session_state.activity_started
    and st.session_state.current_activity != "READY"
):

    start_time = st.session_state.activity_started
    current_activity = st.session_state.current_activity

    live_reps = (
        st.session_state.rep_totals.get(
            current_activity,
            0,
        )
        if current_activity in REPETITIVE
        else 0
    )

    history_rows.append(
        {
            "Exercise": current_activity,
            "Start":
                time.strftime(
                    "%I:%M:%S %p",
                    time.localtime(
                        start_time
                    ),
                ),
            "End": "LIVE",
            "Duration":
                format_duration(
                    time.time()
                    -
                    start_time
                ),
            "Reps":
                live_reps
                if current_activity in REPETITIVE
                else "—",
            "Confidence":
                (
                    f"{st.session_state.current_confidence * 100:.0f}%"
                    if st.session_state.current_confidence
                    is not None
                    else "—"
                ),
        }
    )

for item in reversed(
    st.session_state.history[-15:]
):

    history_rows.append(
        {
            "Exercise": item["activity"],
            "Start":
                time.strftime(
                    "%I:%M:%S %p",
                    time.localtime(
                        item["start"]
                    ),
                ),
            "End":
                time.strftime(
                    "%I:%M:%S %p",
                    time.localtime(
                        item["end"]
                    ),
                ),
            "Duration":
                format_duration(
                    item["duration"]
                ),
            "Reps":
                (
                    item["reps"]
                    if item["reps"]
                    else "—"
                ),
            "Confidence":
                (
                    f"{item['confidence'] * 100:.0f}%"
                    if item["confidence"]
                    is not None
                    else "—"
                ),
        }
    )

if history_rows:

    st.dataframe(
        pd.DataFrame(
            history_rows
        ),
        use_container_width=True,
        hide_index=True,
        height=min(
            225,
            38 + len(history_rows) * 34,
        ),
    )

else:

    st.caption(
        "Start workout. Detected activities will appear "
        "here with start time, end time and duration."
    )


# ============================================================
# SUMMARY
# ============================================================

if st.session_state.summary:

    summary = st.session_state.summary

    st.markdown(
        """
<div class="gs-card">
    <div class="gs-kicker">
        WORKOUT COMPLETE
    </div>

    <div class="gs-title">
        🏆 Session Summary
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Workout Time",
        format_duration(
            summary["duration"]
        ),
    )

    col2.metric(
        "Active Time",
        format_duration(
            summary["active"]
        ),
    )

    col3, col4 = st.columns(2)

    col3.metric(
        "Total Reps",
        summary["reps"],
    )

    col4.metric(
        "Total Sets",
        summary["sets"],
    )

    if summary["activities"]:
        st.caption(
            "Exercises: "
            +
            ", ".join(
                summary["activities"]
            )
        )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "GymSense AI • Ranjitha B K • "
    "1SB24AI041 • AIML"
)
