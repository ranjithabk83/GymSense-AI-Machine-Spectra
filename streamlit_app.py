from __future__ import annotations

import base64
import json
import mimetypes
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
AVATARS = BASE / "assets" / "avatars"

REST_THRESHOLD = 0.12
LOW_CONFIDENCE = 0.42
SMOOTH_WINDOW = 3

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

ACTIVITY_SLUG = {
    "READY": "rest",
    "REST": "rest",
    "WALKING": "walking",
    "RUNNING": "running",
    "BICEP CURL": "bicep_curl",
    "HAMMER CURL": "hammer_curl",
    "SQUAT": "squat",
    "LUNGE": "lunge",
    "JUMPING JACK": "jumping_jack",
    "SHOULDER PRESS": "shoulder_press",
    "FRONT RAISE": "front_raise",
    "LATERAL RAISE": "lateral_raise",
    "WORKOUT": "workout",
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
# COMPACT MOBILE CSS
# ============================================================

st.markdown(
    """
<style>

html, body, [class*="css"] {
    font-family: Inter, system-ui, sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 0%, rgba(67,87,255,.18), transparent 22rem),
        radial-gradient(circle at 95% 0%, rgba(255,58,151,.15), transparent 22rem),
        #07101f;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 620px !important;
    padding-top: .6rem !important;
    padding-left: .7rem !important;
    padding-right: .7rem !important;
    padding-bottom: 2rem !important;
}


/* HEADER */

.gs-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 4px;
}

.gs-brand {
    font-size: 1.28rem;
    font-weight: 950;
    letter-spacing: -.04em;
}

.gs-brand span {
    color: #ff59aa;
}

.gs-sub {
    color: #879bb9;
    font-size: .68rem;
    margin-top: -2px;
}

.status-pill {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 999px;
    background: rgba(52,224,156,.08);
    border: 1px solid rgba(52,224,156,.25);
    color: #52e2a5;
    font-size: .62rem;
    font-weight: 900;
}


/* SMALL CARDS */

.gs-card {
    border: 1px solid rgba(255,255,255,.08);
    background: linear-gradient(
        145deg,
        rgba(14,29,53,.96),
        rgba(7,17,33,.97)
    );
    border-radius: 17px;
    padding: 10px;
    margin-bottom: 8px;
}

.gs-section-title {
    color: #8ea4c3;
    font-size: .63rem;
    letter-spacing: .13em;
    font-weight: 900;
    margin-bottom: 3px;
}


/* AVATAR CHOOSER — SMALL */

.avatar-preview {
    width: 100%;
    max-width: 165px;
    height: 155px;
    margin: 0 auto 5px auto;
    display: flex;
    align-items: flex-end;
    justify-content: center;
    overflow: hidden;

    border-radius: 16px;

    border:
        1px solid
        rgba(255,255,255,.08);

    background:
        radial-gradient(
            circle at 50% 40%,
            rgba(255,80,170,.15),
            transparent 55%
        ),
        #09182d;
}

.avatar-preview img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    object-position: center bottom;
}


/* MAIN WORKOUT AVATAR — COMPACT */

.avatar-stage {
    width: 100%;
    max-width: 220px;
    height: 205px;

    margin:
        0 auto;

    position: relative;

    display: flex;
    align-items: flex-end;
    justify-content: center;

    overflow: hidden;

    border-radius: 18px;

    border:
        1px solid
        rgba(255,255,255,.08);

    background:
        radial-gradient(
            circle at 50% 35%,
            rgba(96,73,255,.20),
            transparent 55%
        ),
        #09172b;
}

.avatar-stage img {
    max-width: 95%;
    max-height: 92%;
    width: auto;
    height: auto;
    object-fit: contain;
    object-position: center bottom;
    transform-origin: 50% 92%;
    will-change: transform;
}

.activity-overlay {
    position: absolute;
    top: 7px;
    left: 50%;
    transform: translateX(-50%);

    z-index: 4;

    min-width: 110px;

    padding: 5px 9px;

    border-radius: 999px;

    text-align: center;

    font-size: .75rem;
    font-weight: 950;

    color: white;

    background:
        rgba(5,14,29,.82);

    border:
        1px solid
        rgba(255,255,255,.12);

    backdrop-filter:
        blur(8px);
}


/* CURRENT ACTIVITY */

.activity-name {
    text-align: center;
    margin-top: 5px;
    font-size: 1.35rem;
    line-height: 1;
    font-weight: 950;
    letter-spacing: -.04em;
}

.confidence {
    text-align: center;
    color: #8fa4c2;
    font-size: .67rem;
    margin-top: 3px;
}


/* MINI METRICS */

.metric-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
    margin-top: 7px;
}

.mini-metric {
    text-align: center;
    background: rgba(255,255,255,.035);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 12px;
    padding: 6px 3px;
}

.mini-metric-label {
    color: #8296b4;
    font-size: .56rem;
    font-weight: 800;
}

.mini-metric-value {
    color: white;
    font-size: .92rem;
    font-weight: 950;
    margin-top: 1px;
}


/* BUTTONS */

div.stButton > button {
    min-height: 37px !important;
    padding: 4px 7px !important;
    border-radius: 11px !important;
    font-size: .72rem !important;
    font-weight: 850 !important;
}


/* HISTORY */

[data-testid="stDataFrame"] {
    font-size: .7rem !important;
}


/* ANIMATIONS */

.motion-walking img {
    animation: walk .56s ease-in-out infinite alternate;
}

.motion-running img {
    animation: run .32s ease-in-out infinite alternate;
}

.motion-squat img {
    animation: squat .9s ease-in-out infinite;
}

.motion-bicep-curl img,
.motion-hammer-curl img {
    animation: curl .72s ease-in-out infinite alternate;
}

.motion-lunge img {
    animation: lunge .9s ease-in-out infinite alternate;
}

.motion-jumping-jack img {
    animation: jump .62s ease-in-out infinite;
}

.motion-shoulder-press img,
.motion-front-raise img,
.motion-lateral-raise img {
    animation: lift .76s ease-in-out infinite alternate;
}

.motion-workout img {
    animation: workout .65s ease-in-out infinite alternate;
}

.motion-rest img {
    animation: breathe 2s ease-in-out infinite;
}


@keyframes walk {
    0% {
        transform:
            translate(-7px, 0)
            rotate(-1.5deg);
    }

    50% {
        transform:
            translate(0, -7px)
            rotate(1.5deg);
    }

    100% {
        transform:
            translate(7px, 0)
            rotate(-1deg);
    }
}

@keyframes run {
    0% {
        transform:
            translate(-9px, 1px)
            rotate(-3deg);
    }

    100% {
        transform:
            translate(9px, -9px)
            rotate(3deg);
    }
}

@keyframes squat {
    0%, 100% {
        transform: translateY(0);
    }

    50% {
        transform:
            translateY(23px)
            scaleY(.96);
    }
}

@keyframes curl {
    0% {
        transform:
            translateY(0)
            rotate(-2deg);
    }

    100% {
        transform:
            translateY(-8px)
            rotate(2deg);
    }
}

@keyframes lunge {
    0% {
        transform:
            translate(-5px, 0);
    }

    100% {
        transform:
            translate(6px, 13px);
    }
}

@keyframes jump {
    0%, 100% {
        transform: translateY(0);
    }

    50% {
        transform:
            translateY(-18px)
            scale(1.02);
    }
}

@keyframes lift {
    0% {
        transform:
            translateY(0)
            rotate(-1deg);
    }

    100% {
        transform:
            translateY(-9px)
            rotate(1deg);
    }
}

@keyframes workout {
    0% {
        transform:
            translateY(0)
            rotate(-2deg);
    }

    100% {
        transform:
            translateY(-7px)
            rotate(2deg);
    }
}

@keyframes breathe {
    0%,100% {
        transform: scale(1);
    }

    50% {
        transform: scale(1.012);
    }
}


/* EXTRA MOBILE */

@media(max-width: 480px) {

    .block-container {
        padding-left: .45rem !important;
        padding-right: .45rem !important;
    }

    .avatar-preview {
        max-width: 145px;
        height: 137px;
    }

    .avatar-stage {
        max-width: 195px;
        height: 182px;
    }

    .activity-name {
        font-size: 1.15rem;
    }

    .mini-metric-value {
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
    "workout_running": False,
    "workout_start": None,
    "current_activity": "READY",
    "current_confidence": None,
    "confidence_label": "",
    "raw_prediction": None,
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
# LOAD ML MODEL
# ============================================================

@st.cache_resource
def load_ml():
    model = joblib.load(
        MODELS / "exercise_classifier.pkl"
    )

    scaler = joblib.load(
        MODELS / "scaler.pkl"
    )

    encoder = joblib.load(
        MODELS / "label_encoder.pkl"
    )

    metadata = {}

    metadata_path = (
        MODELS /
        "model_metadata.json"
    )

    if metadata_path.exists():
        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

    feature_names = (
        metadata.get("feature_names")
        or get_feature_names()
    )

    return (
        model,
        scaler,
        encoder,
        feature_names,
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_activity(
    records: list[dict[str, Any]]
):

    model, scaler, encoder, names = load_ml()

    features = extract_features_from_records(
        records
    )

    x = np.array(
        [[
            float(
                features.get(
                    name,
                    0.0
                )
            )
            for name in names
        ]],
        dtype=float,
    )

    x = scaler.transform(x)

    confidence = None
    probabilities = {}

    if hasattr(
        model,
        "predict_proba"
    ):

        probs = model.predict_proba(x)[0]

        class_ids = getattr(
            model,
            "classes_",
            np.arange(len(probs))
        )

        labels = []

        for class_id in class_ids:

            label = str(
                encoder.inverse_transform(
                    [int(class_id)]
                )[0]
            )

            label = (
                label
                .upper()
                .replace(
                    "_",
                    " "
                )
            )

            labels.append(label)

        best_index = int(
            np.argmax(probs)
        )

        raw_prediction = labels[
            best_index
        ]

        confidence = float(
            probs[
                best_index
            ]
        )

        probabilities = {
            label: float(probability)
            for label, probability
            in zip(
                labels,
                probs
            )
        }

    else:

        encoded = int(
            model.predict(x)[0]
        )

        raw_prediction = str(
            encoder.inverse_transform(
                [encoded]
            )[0]
        )

        raw_prediction = (
            raw_prediction
            .upper()
            .replace(
                "_",
                " "
            )
        )


    # Stationary check

    ax = np.array([
        float(
            record.get(
                "accel_x",
                0.0
            )
        )
        for record in records
    ])

    ay = np.array([
        float(
            record.get(
                "accel_y",
                0.0
            )
        )
        for record in records
    ])

    az = np.array([
        float(
            record.get(
                "accel_z",
                9.81
            )
        )
        for record in records
    ])

    movement_variance = float(
        np.var(ax)
        +
        np.var(ay)
        +
        np.var(az)
    )


    if movement_variance < REST_THRESHOLD:

        return (
            "REST",
            probabilities.get(
                "REST"
            ),
            "REST confidence"
            if "REST" in probabilities
            else "Stationary",
            raw_prediction,
        )


    if (
        confidence is not None
        and
        confidence < LOW_CONFIDENCE
    ):

        return (
            "WORKOUT",
            confidence,
            "Model confidence",
            raw_prediction,
        )


    return (
        raw_prediction,
        confidence,
        "Model confidence",
        raw_prediction,
    )


# ============================================================
# SMOOTH PREDICTION
# ============================================================

def smooth_prediction(label):

    recent = list(
        st.session_state.recent
    )

    recent.append(label)

    recent = recent[
        -SMOOTH_WINDOW:
    ]

    st.session_state.recent = recent

    if len(recent) < SMOOTH_WINDOW:
        return label

    winner, count = (
        Counter(recent)
        .most_common(1)[0]
    )

    if count >= 2:
        return winner

    current = (
        st.session_state
        .current_activity
    )

    if current != "READY":
        return current

    return label


# ============================================================
# REP COUNTING
# ============================================================

def find_rep_peaks(records):

    if len(records) < 12:
        return []

    ax = np.array([
        float(r.get("accel_x", 0))
        for r in records
    ])

    ay = np.array([
        float(r.get("accel_y", 0))
        for r in records
    ])

    az = np.array([
        float(r.get("accel_z", 9.81))
        for r in records
    ])

    magnitude = np.sqrt(
        ax * ax +
        ay * ay +
        az * az
    )

    dynamic = np.abs(
        magnitude -
        np.median(magnitude)
    )

    threshold = max(
        0.65,
        float(
            np.mean(dynamic)
            +
            0.75 *
            np.std(dynamic)
        )
    )

    peaks = []
    last_index = -999

    min_distance = max(
        6,
        len(records) // 10
    )

    for index in range(
        1,
        len(dynamic) - 1
    ):

        if (
            dynamic[index] > threshold
            and
            dynamic[index] >= dynamic[index - 1]
            and
            dynamic[index] > dynamic[index + 1]
            and
            index - last_index >= min_distance
        ):

            timestamp = int(
                records[index]
                .get(
                    "timestamp_ms",
                    0
                )
                or 0
            )

            peaks.append(timestamp)
            last_index = index

    return peaks


def update_reps(
    records,
    activity
):

    if activity not in REPETITIVE:
        return

    st.session_state.rep_totals.setdefault(
        activity,
        0
    )

    last_timestamp = int(
        st.session_state
        .last_rep_ts
        .get(
            activity,
            0
        )
    )

    for timestamp in find_rep_peaks(
        records
    ):

        if (
            timestamp > 0
            and
            timestamp - last_timestamp >= 550
        ):

            st.session_state.rep_totals[
                activity
            ] += 1

            last_timestamp = timestamp

    st.session_state.last_rep_ts[
        activity
    ] = last_timestamp


# ============================================================
# ACTIVITY HISTORY
# ============================================================

def start_activity(activity):

    st.session_state.current_activity = activity

    st.session_state.activity_started = (
        time.time()
    )

    if activity in REPETITIVE:

        if (
            st.session_state.last_non_rest
            != activity
        ):

            st.session_state.set_totals[
                activity
            ] = (
                st.session_state
                .set_totals
                .get(
                    activity,
                    0
                )
                + 1
            )

        st.session_state.segment_rep_start = (
            st.session_state
            .rep_totals
            .get(
                activity,
                0
            )
        )

    else:
        st.session_state.segment_rep_start = 0


def close_activity():

    activity = (
        st.session_state
        .current_activity
    )

    started = (
        st.session_state
        .activity_started
    )

    if (
        activity in (
            None,
            "READY",
        )
        or
        started is None
    ):
        return

    end_time = time.time()

    reps = 0

    if activity in REPETITIVE:

        reps = max(
            0,
            st.session_state
            .rep_totals
            .get(
                activity,
                0
            )
            -
            st.session_state
            .segment_rep_start
        )

    st.session_state.history.append(
        {
            "activity": activity,
            "start": started,
            "end": end_time,
            "duration": max(
                0,
                end_time - started
            ),
            "reps": reps,
            "confidence":
                st.session_state
                .current_confidence,
        }
    )

    if activity != "REST":
        st.session_state.last_non_rest = (
            activity
        )


def change_activity(
    new_activity
):

    old_activity = (
        st.session_state
        .current_activity
    )

    if old_activity == "READY":
        start_activity(
            new_activity
        )
        return

    if old_activity == new_activity:
        return

    close_activity()

    if new_activity == "REST":
        st.session_state.last_non_rest = None

    start_activity(
        new_activity
    )


# ============================================================
# WORKOUT
# ============================================================

def start_workout():

    st.session_state.workout_running = True

    st.session_state.workout_start = (
        time.time()
    )

    st.session_state.current_activity = (
        "READY"
    )

    st.session_state.activity_started = None
    st.session_state.history = []
    st.session_state.rep_totals = {}
    st.session_state.set_totals = {}
    st.session_state.last_rep_ts = {}
    st.session_state.recent = []
    st.session_state.last_non_rest = None
    st.session_state.summary = None


def end_workout():

    if not st.session_state.workout_running:
        return

    close_activity()

    end_time = time.time()

    start_time = (
        st.session_state.workout_start
        or end_time
    )

    active_time = sum(
        item["duration"]
        for item
        in st.session_state.history
        if item["activity"] != "REST"
    )

    activities = sorted({
        item["activity"]
        for item
        in st.session_state.history
        if item["activity"] != "REST"
    })

    st.session_state.summary = {
        "duration":
            max(
                0,
                end_time - start_time
            ),
        "active":
            active_time,
        "reps":
            sum(
                st.session_state
                .rep_totals
                .values()
            ),
        "sets":
            sum(
                st.session_state
                .set_totals
                .values()
            ),
        "activities":
            activities,
    }

    st.session_state.workout_running = False
    st.session_state.current_activity = "READY"
    st.session_state.activity_started = None
    st.session_state.recent = []


# ============================================================
# AVATAR
# ============================================================

def file_to_uri(path: Path):

    mime, _ = mimetypes.guess_type(
        path.name
    )

    encoded = (
        base64
        .b64encode(
            path.read_bytes()
        )
        .decode("ascii")
    )

    return (
        f"data:"
        f"{mime or 'image/png'}"
        f";base64,"
        f"{encoded}"
    )


def get_avatar_media(
    gender,
    activity
):

    folder = (
        AVATARS /
        gender
    )

    slug = ACTIVITY_SLUG.get(
        activity,
        "workout"
    )

    # Real animation assets have priority

    for extension in (
        ".gif",
        ".webp",
        ".mp4",
        ".png",
    ):

        candidate = (
            folder /
            f"{slug}{extension}"
        )

        if candidate.exists():
            return candidate

    default_image = (
        folder /
        "default.png"
    )

    if default_image.exists():
        return default_image

    return None


def show_avatar_preview(
    path,
    name
):

    if not path.exists():

        st.error(
            f"Missing {name} avatar."
        )

        return

    st.markdown(
        f"""
<div class="avatar-preview">
<img
src="{file_to_uri(path)}"
alt="{name}">
</div>
""",
        unsafe_allow_html=True,
    )


def current_elapsed():

    if (
        st.session_state.workout_running
        and
        st.session_state.activity_started
    ):

        return max(
            0,
            int(
                time.time()
                -
                st.session_state
                .activity_started
            )
        )

    return 0


def format_duration(seconds):

    seconds = int(
        max(
            0,
            seconds
        )
    )

    minutes, seconds = divmod(
        seconds,
        60
    )

    return f"{minutes:02d}:{seconds:02d}"


def show_current_avatar():

    gender = (
        st.session_state.avatar
        or "female"
    )

    activity = (
        st.session_state
        .current_activity
    )

    media = get_avatar_media(
        gender,
        activity
    )

    if media is None:

        st.error(
            "Avatar image missing."
        )

        return

    emoji = ACTIVITY_EMOJI.get(
        activity,
        "🔥"
    )

    if media.suffix.lower() == ".mp4":

        st.markdown(
            f"""
<div class="activity-overlay">
{emoji} {activity}
</div>
""",
            unsafe_allow_html=True,
        )

        st.video(
            str(media),
            autoplay=True,
            loop=True,
            muted=True,
        )

        return

    motion_class = (
        "motion-"
        +
        ACTIVITY_SLUG
        .get(
            activity,
            "workout"
        )
        .replace(
            "_",
            "-"
        )
    )

    st.markdown(
        f"""
<div class="avatar-stage {motion_class}">

<div class="activity-overlay">
{emoji} {activity}
</div>

<img
src="{file_to_uri(media)}"
alt="{activity} avatar">

</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# PHONE SENSOR COMPONENT
# ============================================================

SENSOR_HTML = r"""
<div class="sensor-card">

<div class="sensor-title-row">

<div>
<b>MOTION SENSOR</b>
<span id="message">
Tap Enable Sensor
</span>
</div>

<span id="badge">
● OFF
</span>

</div>


<div class="button-grid">

<button id="enable">
Enable Sensor
</button>

<button
id="start"
disabled>
Start Workout
</button>

<button
id="end"
disabled>
End Workout
</button>

</div>


<div class="sensor-grid">

<div class="sensor-box">

<div class="sensor-box-title">
ACCELEROMETER
<span id="acc-status">
OFF
</span>
</div>

<canvas id="acc-canvas"></canvas>

<div class="sensor-values">

<span>
X <b id="ax">0.00</b>
</span>

<span>
Y <b id="ay">0.00</b>
</span>

<span>
Z <b id="az">0.00</b>
</span>

</div>

</div>


<div class="sensor-box">

<div class="sensor-box-title">
GYROSCOPE
<span id="gyro-status">
OFF
</span>
</div>

<canvas id="gyro-canvas"></canvas>

<div class="sensor-values">

<span>
α <b id="ga">0.00</b>
</span>

<span>
β <b id="gb">0.00</b>
</span>

<span>
γ <b id="gg">0.00</b>
</span>

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
    color: #ffffff;

    border:
        1px solid
        rgba(255,255,255,.08);

    background:
        linear-gradient(
            145deg,
            #0b1930,
            #071426
        );

    border-radius: 16px;

    padding: 9px;

    font-family:
        Inter,
        system-ui;
}


.sensor-title-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
}


.sensor-title-row b {
    display: block;
    font-size: 10px;
    letter-spacing: .13em;
    color: #8ba0bd;
}


.sensor-title-row div span {
    display: block;
    font-size: 11px;
    color: #9db0ca;
    margin-top: 2px;
}


#badge {
    font-size: 10px;
    font-weight: 900;

    border:
        1px solid
        #35465e;

    border-radius: 999px;

    padding:
        4px 7px;
}


.button-grid {
    display: grid;

    grid-template-columns:
        1.2fr 1fr 1fr;

    gap: 5px;

    margin-top: 8px;
}


button {
    min-height: 35px;

    border-radius: 10px;

    border:
        1px solid
        rgba(255,255,255,.10);

    background:
        #142540;

    color: white;

    font-size: 10px;

    font-weight: 850;

    padding: 4px;
}


#enable {
    background:
        linear-gradient(
            90deg,
            #ff4e9f,
            #8e48ff
        );

    border: none;
}


#start {
    background:
        linear-gradient(
            90deg,
            #22b879,
            #24d6a0
        );

    border: none;
}


#end {
    background:
        linear-gradient(
            90deg,
            #d43d68,
            #ff536d
        );

    border: none;
}


button:disabled {
    opacity: .34;
}


.sensor-grid {
    display: grid;

    grid-template-columns:
        1fr 1fr;

    gap: 6px;

    margin-top: 7px;
}


.sensor-box {
    border:
        1px solid
        rgba(255,255,255,.06);

    background:
        #06111f;

    border-radius: 12px;

    padding: 6px;
}


.sensor-box-title {
    display: flex;
    justify-content: space-between;

    font-size: 8px;

    color:
        #9db0ca;

    font-weight: 850;
}


.sensor-box-title span {
    color: #50e2a4;
}


canvas {
    display: block;

    width: 100%;

    height: 58px;

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
    color: #ffffff;
}


.sensor-note {
    font-size: 8px;

    color: #657c9b;

    text-align: center;

    margin-top: 4px;
}


@media(max-width:420px) {

    .button-grid {
        grid-template-columns:
            1fr 1fr 1fr;
    }

    button {
        font-size: 9px;
        min-height: 33px;
    }

    .sensor-grid {
        grid-template-columns:
            1fr 1fr;
    }

    canvas {
        height: 50px;
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


if (!parentElement.__gymSenseState) {

const S = {

    enabled: false,

    calibrated: false,

    workout: false,

    buffer: [],

    lastEmit: 0,

    accHistory: [],

    gyroHistory: [],

    listener: null
};


parentElement.__gymSenseState = S;


const q =
selector =>
parentElement.querySelector(selector);


const enableButton =
q('#enable');

const startButton =
q('#start');

const endButton =
q('#end');

const message =
q('#message');

const badge =
q('#badge');

const accStatus =
q('#acc-status');

const gyroStatus =
q('#gyro-status');


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
        live
        ? '● LIVE'
        : '● OFF';

    badge.style.color =
        live
        ? '#56e5a8'
        : '#ffffff';

    accStatus.textContent =
        live
        ? 'LIVE'
        : 'OFF';

    gyroStatus.textContent =
        live
        ? 'LIVE'
        : 'OFF';
}


function addHistory(
    list,
    value
) {

    list.push(value);

    if (list.length > 55) {

        list.splice(
            0,
            list.length - 55
        );
    }
}


function drawGraph(
    canvas,
    rows,
    colors
) {

    const rect =
        canvas.getBoundingClientRect();

    const ratio =
        window.devicePixelRatio
        || 1;

    const width =
        Math.max(
            120,
            rect.width
        );

    const height =
        Math.max(
            45,
            rect.height
        );


    canvas.width =
        width * ratio;

    canvas.height =
        height * ratio;


    const ctx =
        canvas.getContext(
            '2d'
        );


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

        ctx.moveTo(
            0,
            y
        );

        ctx.lineTo(
            width,
            y
        );

        ctx.stroke();
    }


    if (rows.length < 2) {
        return;
    }


    const flattened =
        rows
        .flat()
        .filter(
            Number.isFinite
        );


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

        ctx.lineWidth =
            1.4;

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

                    ctx.moveTo(
                        x,
                        y
                    );

                } else {

                    ctx.lineTo(
                        x,
                        y
                    );
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
                    status:
                        'UNSUPPORTED',
                    ts:
                        Date.now()
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
                        status:
                            'DENIED',
                        ts:
                            Date.now()
                    }
                );

                return;
            }
        }


        if (!S.listener) {

            S.listener =
                event => {

                    const accel =
                        event
                        .accelerationIncludingGravity

                        ||

                        event
                        .acceleration

                        ||

                        {};


                    const rotation =
                        event
                        .rotationRate

                        ||

                        {};


                    const sample = {

                        accel_x:
                            Number(
                                accel.x
                                || 0
                            ),

                        accel_y:
                            Number(
                                accel.y
                                || 0
                            ),

                        accel_z:
                            Number(
                                accel.z
                                || 0
                            ),

                        gyro_alpha:
                            Number(
                                rotation.alpha
                                || 0
                            ),

                        gyro_beta:
                            Number(
                                rotation.beta
                                || 0
                            ),

                        gyro_gamma:
                            Number(
                                rotation.gamma
                                || 0
                            ),

                        timestamp_ms:
                            Date.now()
                    };


                    ax.textContent =
                        sample
                        .accel_x
                        .toFixed(2);

                    ay.textContent =
                        sample
                        .accel_y
                        .toFixed(2);

                    az.textContent =
                        sample
                        .accel_z
                        .toFixed(2);


                    ga.textContent =
                        sample
                        .gyro_alpha
                        .toFixed(2);

                    gb.textContent =
                        sample
                        .gyro_beta
                        .toFixed(2);

                    gg.textContent =
                        sample
                        .gyro_gamma
                        .toFixed(2);


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


                    if (S.workout) {

                        S.buffer.push(
                            sample
                        );


                        const now =
                            Date.now();


                        if (
                            S.buffer.length
                            >= 20

                            &&

                            now
                            -
                            S.lastEmit
                            >= 1600
                        ) {

                            setTriggerValue(
                                'window',
                                {
                                    samples:
                                        S.buffer
                                        .slice(
                                            -100
                                        ),
                                    ts:
                                        now
                                }
                            );


                            S.buffer =
                                S.buffer
                                .slice(
                                    -6
                                );


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

        setLive(true);

        enableButton.disabled = true;

        enableButton.textContent =
            '✓ Sensor Enabled';


        /*
        AUTO CALIBRATION
        No separate Calibrate button.
        */

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

                        clearInterval(
                            timer
                        );


                        S.calibrated =
                            true;


                        startButton.disabled =
                            false;


                        setMessage(
                            'Sensor ready'
                        );


                        setTriggerValue(
                            'sensor_status',
                            {
                                status:
                                    'READY',
                                ts:
                                    Date.now()
                            }
                        );
                    }

                },
                1000
            );


        setTriggerValue(
            'sensor_status',
            {
                status:
                    'LIVE',
                ts:
                    Date.now()
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
                status:
                    'ERROR',
                ts:
                    Date.now()
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
            event:
                'START',
            ts:
                Date.now()
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
            event:
                'END',
            ts:
                Date.now()
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
    .__gymSenseState;


if (
    data?.workoutRunning
    === false
    &&
    S.workout
) {

    S.workout = false;
}


return () => {};

}
"""


sensor_component = (
    st.components.v2.component(
        "gymsense_mobile_sensor",
        html=SENSOR_HTML,
        css=SENSOR_CSS,
        js=SENSOR_JS,
    )
)


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns(
    [2.5, 1],
    vertical_alignment="center",
)

with header_left:

    st.markdown(
        f"""
<div class="gs-header">

<div>

<div class="gs-brand">
GymSense <span>AI</span>
</div>

<div class="gs-sub">
Phone Motion Workout Recognition
</div>

</div>

<div class="status-pill">
● {st.session_state.sensor_status}
</div>

</div>
""",
        unsafe_allow_html=True,
    )


with header_right:

    if (
        st.session_state.avatar
        is not None
    ):

        switch = st.button(
            "↔ Avatar",
            use_container_width=True,
            disabled=
                st.session_state
                .workout_running,
        )

        if switch:

            st.session_state.avatar = None
            st.session_state.current_activity = "READY"
            st.session_state.current_confidence = None
            st.session_state.recent = []

            st.rerun()


# ============================================================
# AVATAR SELECTION
# ============================================================

if st.session_state.avatar is None:

    st.markdown(
        """
<div class="gs-card">

<div class="gs-section-title">
CHOOSE AVATAR
</div>

<div style="
font-size:1.25rem;
font-weight:950;
">
Select Your Workout Avatar
</div>

<div style="
font-size:.68rem;
color:#879bb9;
">
Avatar selection changes only the visual character.
</div>

</div>
""",
        unsafe_allow_html=True,
    )


    female_column, male_column = (
        st.columns(
            2,
            gap="small",
        )
    )


    with female_column:

        female = (
            AVATARS /
            "female" /
            "default.png"
        )

        show_avatar_preview(
            female,
            "Female"
        )

        if st.button(
            "Female",
            use_container_width=True,
            type="primary",
        ):

            st.session_state.avatar = (
                "female"
            )

            st.rerun()


    with male_column:

        male = (
            AVATARS /
            "male" /
            "default.png"
        )

        show_avatar_preview(
            male,
            "Male"
        )

        if st.button(
            "Male",
            use_container_width=True,
        ):

            st.session_state.avatar = (
                "male"
            )

            st.rerun()


    st.stop()


# ============================================================
# LIVE ACTIVITY PANEL
# ============================================================

st.markdown(
    '<div class="gs-section-title">'
    'LIVE WORKOUT'
    '</div>',
    unsafe_allow_html=True,
)


show_current_avatar()


activity = (
    st.session_state
    .current_activity
)

confidence = (
    st.session_state
    .current_confidence
)


st.markdown(
    f"""
<div class="activity-name">

{ACTIVITY_EMOJI.get(activity, "🔥")}
{activity}

</div>
""",
    unsafe_allow_html=True,
)


if confidence is not None:

    st.markdown(
        f"""
<div class="confidence">

Confidence:
{confidence * 100:.1f}%

</div>
""",
        unsafe_allow_html=True,
    )

else:

    st.markdown(
        """
<div class="confidence">

Waiting for sensor prediction

</div>
""",
        unsafe_allow_html=True,
    )


elapsed = current_elapsed()

current_reps = (
    st.session_state
    .rep_totals
    .get(
        activity,
        0
    )
    if activity in REPETITIVE
    else 0
)

current_sets = (
    st.session_state
    .set_totals
    .get(
        activity,
        0
    )
    if activity in REPETITIVE
    else 0
)


st.markdown(
    f"""
<div class="metric-row">

<div class="mini-metric">

<div class="mini-metric-label">
DURATION
</div>

<div class="mini-metric-value">
{format_duration(elapsed)}
</div>

</div>


<div class="mini-metric">

<div class="mini-metric-label">
REPS
</div>

<div class="mini-metric-value">
{current_reps if activity in REPETITIVE else "—"}
</div>

</div>


<div class="mini-metric">

<div class="mini-metric-label">
SETS
</div>

<div class="mini-metric-value">
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
                st.session_state
                .workout_running
            ),
        "prediction":
            st.session_state
            .current_activity,
    },
    key="gym-phone-sensor",
    on_sensor_status_change=
        lambda: None,
    on_session_event_change=
        lambda: None,
    on_window_change=
        lambda: None,
)


# ============================================================
# SENSOR STATUS EVENT
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
            ""
        )
    ).upper()

    if status == "LIVE":

        st.session_state.sensor_status = (
            "LIVE"
        )

    elif status == "READY":

        st.session_state.sensor_status = (
            "READY"
        )

        st.session_state.calibrated = (
            True
        )

    elif status:

        st.session_state.sensor_status = (
            status
        )


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
            ""
        )
    ).upper()

    if (
        event == "START"
        and
        not
        st.session_state
        .workout_running
    ):

        start_workout()
        st.rerun()

    elif (
        event == "END"
        and
        st.session_state
        .workout_running
    ):

        end_workout()
        st.rerun()


# ============================================================
# ML WINDOW
# ============================================================

sensor_window = getattr(
    sensor_result,
    "window",
    None,
)

if (
    sensor_window
    and
    st.session_state
    .workout_running
    and
    isinstance(
        sensor_window.get(
            "samples"
        ),
        list,
    )
):

    records = sensor_window[
        "samples"
    ]

    if len(records) >= 10:

        (
            predicted_activity,
            predicted_confidence,
            confidence_label,
            raw_prediction,
        ) = predict_activity(
            records
        )

        predicted_activity = (
            smooth_prediction(
                predicted_activity
            )
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

        change_activity(
            predicted_activity
        )

        update_reps(
            records,
            predicted_activity,
        )

        st.rerun()


# ============================================================
# COMPACT HISTORY
# ============================================================

st.markdown(
    '<div class="gs-section-title" '
    'style="margin-top:10px;">'
    'ACTIVITY HISTORY'
    '</div>',
    unsafe_allow_html=True,
)


history_rows = []


# LIVE CURRENT ACTIVITY FIRST

if (
    st.session_state.workout_running
    and
    st.session_state.activity_started
    and
    st.session_state.current_activity
    != "READY"
):

    start_time = (
        st.session_state
        .activity_started
    )

    activity = (
        st.session_state
        .current_activity
    )

    live_reps = (
        st.session_state
        .rep_totals
        .get(
            activity,
            0
        )
        if activity in REPETITIVE
        else 0
    )

    history_rows.append(
        {
            "Exercise":
                activity,

            "Start":
                time.strftime(
                    "%I:%M:%S %p",
                    time.localtime(
                        start_time
                    ),
                ),

            "End":
                "LIVE",

            "Duration":
                format_duration(
                    time.time()
                    -
                    start_time
                ),

            "Reps":
                live_reps
                if activity in REPETITIVE
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


# COMPLETED ACTIVITIES

for item in reversed(
    st.session_state.history[
        -15:
    ]
):

    history_rows.append(
        {
            "Exercise":
                item["activity"],

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
            38 +
            len(history_rows) * 34,
        ),
    )

else:

    st.caption(
        "Start workout. Detected exercises "
        "will appear here with time and duration."
    )


# ============================================================
# SUMMARY
# ============================================================

if st.session_state.summary:

    summary = (
        st.session_state.summary
    )

    st.markdown(
        """
<div class="gs-card">

<div class="gs-section-title">
WORKOUT COMPLETE
</div>

<div style="
font-size:1.05rem;
font-weight:950;
">
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
                summary[
                    "activities"
                ]
            )
        )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "GymSense AI • Ranjitha B K • "
    "1SB24AI041 • AIML"
)