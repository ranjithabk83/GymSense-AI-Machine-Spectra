from __future__ import annotations

import json
import math
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
# CONFIG
# ============================================================

st.set_page_config(
    page_title="GymSense AI",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets" / "avatars"

MODEL_PATH = MODELS_DIR / "exercise_classifier.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
META_PATH = MODELS_DIR / "model_metadata.json"

REST_MOTION_VARIANCE_THRESHOLD = 0.12
UNKNOWN_CONFIDENCE_THRESHOLD = 0.42
SMOOTHING_WINDOWS = 3

REPETITIVE_ACTIVITIES = {
    "BICEP CURL",
    "HAMMER CURL",
    "SQUAT",
    "LUNGE",
    "JUMPING JACK",
    "SHOULDER PRESS",
    "FRONT RAISE",
    "LATERAL RAISE",
}

ACTIVITY_SLUGS = {
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


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
      .stApp {
        background:
          radial-gradient(circle at 12% 0%, rgba(112,67,255,.16), transparent 28rem),
          radial-gradient(circle at 90% 4%, rgba(255,57,154,.12), transparent 28rem),
          #07101f;
      }
      [data-testid="stHeader"] { background: transparent; }
      .block-container {
        max-width: 980px;
        padding-top: 1rem;
        padding-bottom: 4rem;
      }
      .gs-top {
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:12px;
        padding:8px 0 14px;
      }
      .gs-brand {
        font-size:1.65rem;
        font-weight:900;
        letter-spacing:-.04em;
      }
      .gs-brand span { color:#ff5cab; }
      .gs-sub { color:#9fb0ca; font-size:.88rem; }
      .gs-live {
        display:inline-flex;
        align-items:center;
        gap:7px;
        border:1px solid rgba(255,92,171,.34);
        background:rgba(255,92,171,.08);
        color:#ff7fbd;
        border-radius:999px;
        padding:7px 11px;
        font-weight:800;
        font-size:.76rem;
      }
      .gs-card {
        border:1px solid rgba(255,255,255,.08);
        background:linear-gradient(145deg,rgba(17,38,70,.9),rgba(9,20,42,.93));
        border-radius:22px;
        padding:18px;
        margin-bottom:14px;
      }
      .gs-title {
        font-size:clamp(2rem,8vw,3.4rem);
        line-height:1;
        font-weight:950;
        letter-spacing:-.055em;
      }
      .gs-kicker {
        color:#93a7c4;
        font-size:.76rem;
        font-weight:800;
        letter-spacing:.18em;
      }
      .gs-status-ok { color:#69e6ad; font-weight:800; }
      .gs-status-warn { color:#ffd278; font-weight:800; }
      div[data-testid="stMetric"] {
        border:1px solid rgba(255,255,255,.08);
        background:rgba(255,255,255,.035);
        border-radius:16px;
        padding:12px 13px;
      }
      div.stButton > button {
        min-height:46px;
        border-radius:14px;
        font-weight:800;
      }
      .avatar-choice {
        border:1px solid rgba(255,255,255,.08);
        background:rgba(255,255,255,.035);
        border-radius:22px;
        padding:14px;
        text-align:center;
      }
      .small-note { color:#8fa2bd; font-size:.82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "avatar": None,
    "sensor_status": "OFF",
    "calibrated": False,
    "workout_running": False,
    "workout_start": None,
    "current_activity": "READY",
    "current_confidence": None,
    "current_confidence_label": "",
    "raw_prediction": None,
    "recent_predictions": [],
    "activity_started": None,
    "history": [],
    "rep_totals": {},
    "set_totals": {},
    "last_rep_ts": {},
    "segment_rep_start": 0,
    "last_activity_before_rest": None,
    "last_session_summary": None,
}

for key, value in defaults.items():
    st.session_state.setdefault(key, value)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_ml():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)

    metadata = {}
    if META_PATH.exists():
        metadata = json.loads(META_PATH.read_text(encoding="utf-8"))

    feature_names = metadata.get("feature_names") or get_feature_names()

    return {
        "model": model,
        "scaler": scaler,
        "encoder": encoder,
        "metadata": metadata,
        "feature_names": feature_names,
    }


def predict_window(records: list[dict[str, Any]]) -> dict[str, Any]:
    ml = load_ml()

    features = extract_features_from_records(records)
    vector = np.array(
        [[float(features.get(name, 0.0)) for name in ml["feature_names"]]],
        dtype=float,
    )
    scaled = ml["scaler"].transform(vector)

    model = ml["model"]
    encoder = ml["encoder"]

    if not hasattr(model, "predict_proba"):
        encoded = int(model.predict(scaled)[0])
        label = str(encoder.inverse_transform([encoded])[0]).upper().replace("_", " ")
        confidence = None
        prob_map = {}
    else:
        probs = model.predict_proba(scaled)[0]
        model_classes = getattr(model, "classes_", np.arange(len(probs)))

        decoded = []
        for class_id in model_classes:
            decoded.append(
                str(encoder.inverse_transform([int(class_id)])[0])
                .upper()
                .replace("_", " ")
            )

        max_index = int(np.argmax(probs))
        label = decoded[max_index]
        confidence = float(probs[max_index])
        prob_map = {name: float(prob) for name, prob in zip(decoded, probs)}

    ax = np.asarray([float(r.get("accel_x", 0.0)) for r in records])
    ay = np.asarray([float(r.get("accel_y", 0.0)) for r in records])
    az = np.asarray([float(r.get("accel_z", 9.81)) for r in records])

    motion_variance = float(np.var(ax) + np.var(ay) + np.var(az))
    rest_probability = prob_map.get("REST")

    if motion_variance < REST_MOTION_VARIANCE_THRESHOLD:
        final_label = "REST"
        final_confidence = rest_probability
        confidence_label = "ML REST probability" if rest_probability is not None else "Stationary"
    elif confidence is not None and confidence < UNKNOWN_CONFIDENCE_THRESHOLD:
        final_label = "WORKOUT"
        final_confidence = confidence
        confidence_label = "Best known-class confidence"
    else:
        final_label = label
        final_confidence = confidence
        confidence_label = "ML confidence"

    return {
        "activity": final_label,
        "confidence": final_confidence,
        "confidence_label": confidence_label,
        "raw_prediction": label,
        "motion_variance": motion_variance,
        "probabilities": prob_map,
    }


def smooth_prediction(new_label: str) -> str:
    recent = st.session_state.recent_predictions
    recent.append(new_label)
    recent = recent[-SMOOTHING_WINDOWS:]
    st.session_state.recent_predictions = recent

    if len(recent) < SMOOTHING_WINDOWS:
        return new_label

    winner, count = Counter(recent).most_common(1)[0]
    if count >= 2:
        return winner

    return st.session_state.current_activity if st.session_state.current_activity != "READY" else new_label


# ============================================================
# REAL SIGNAL REP COUNTER
# ============================================================

def detect_rep_timestamps(records: list[dict[str, Any]]) -> list[int]:
    """
    Conservative generic cycle detector.
    This uses real accelerometer movement, not fake counts.
    It should be tuned with your real exercise dataset before the competition.
    """
    if len(records) < 12:
        return []

    ax = np.asarray([float(r.get("accel_x", 0.0)) for r in records])
    ay = np.asarray([float(r.get("accel_y", 0.0)) for r in records])
    az = np.asarray([float(r.get("accel_z", 9.81)) for r in records])
    mag = np.sqrt(ax * ax + ay * ay + az * az)
    dynamic = np.abs(mag - np.median(mag))

    threshold = max(0.65, float(np.mean(dynamic) + 0.75 * np.std(dynamic)))
    peaks = []

    last_index = -999
    min_distance = max(6, len(records) // 10)

    for i in range(1, len(dynamic) - 1):
        if (
            dynamic[i] > threshold
            and dynamic[i] >= dynamic[i - 1]
            and dynamic[i] > dynamic[i + 1]
            and i - last_index >= min_distance
        ):
            ts = int(records[i].get("timestamp_ms", 0) or 0)
            peaks.append(ts)
            last_index = i

    return peaks


def update_reps(records: list[dict[str, Any]], activity: str) -> None:
    if activity not in REPETITIVE_ACTIVITIES:
        return

    st.session_state.rep_totals.setdefault(activity, 0)
    st.session_state.last_rep_ts.setdefault(activity, 0)

    last_ts = int(st.session_state.last_rep_ts[activity])

    for ts in detect_rep_timestamps(records):
        if ts <= 0:
            continue
        # Cooldown prevents duplicate counts across overlapping/adjacent windows.
        if ts - last_ts >= 550:
            st.session_state.rep_totals[activity] += 1
            last_ts = ts

    st.session_state.last_rep_ts[activity] = last_ts


# ============================================================
# ACTIVITY/HISTORY
# ============================================================

def start_activity(activity: str) -> None:
    now = time.time()
    st.session_state.current_activity = activity
    st.session_state.activity_started = now

    if activity in REPETITIVE_ACTIVITIES:
        if (
            st.session_state.last_activity_before_rest is None
            or st.session_state.last_activity_before_rest != activity
        ):
            st.session_state.set_totals[activity] = st.session_state.set_totals.get(activity, 0) + 1

        st.session_state.segment_rep_start = st.session_state.rep_totals.get(activity, 0)
    else:
        st.session_state.segment_rep_start = 0


def close_current_activity() -> None:
    activity = st.session_state.current_activity
    started = st.session_state.activity_started

    if activity in (None, "READY") or started is None:
        return

    end = time.time()
    reps = 0

    if activity in REPETITIVE_ACTIVITIES:
        reps = max(
            0,
            st.session_state.rep_totals.get(activity, 0)
            - st.session_state.segment_rep_start,
        )

    st.session_state.history.append(
        {
            "activity": activity,
            "start": started,
            "end": end,
            "duration": max(0.0, end - started),
            "reps": reps,
            "confidence": st.session_state.current_confidence,
        }
    )

    if activity == "REST":
        pass
    else:
        st.session_state.last_activity_before_rest = activity


def change_activity(new_activity: str) -> None:
    old = st.session_state.current_activity
    if old == "READY":
        start_activity(new_activity)
        return

    if new_activity == old:
        return

    close_current_activity()

    # A rest period allows the same exercise to become a new set later.
    if new_activity == "REST":
        st.session_state.last_activity_before_rest = old

    start_activity(new_activity)


def start_workout_python() -> None:
    st.session_state.workout_running = True
    st.session_state.workout_start = time.time()
    st.session_state.current_activity = "READY"
    st.session_state.activity_started = None
    st.session_state.history = []
    st.session_state.rep_totals = {}
    st.session_state.set_totals = {}
    st.session_state.last_rep_ts = {}
    st.session_state.recent_predictions = []
    st.session_state.last_activity_before_rest = None
    st.session_state.last_session_summary = None


def end_workout_python() -> None:
    if not st.session_state.workout_running:
        return

    close_current_activity()
    end = time.time()
    start = st.session_state.workout_start or end

    active_time = sum(
        item["duration"]
        for item in st.session_state.history
        if item["activity"] != "REST"
    )

    st.session_state.last_session_summary = {
        "duration": max(0.0, end - start),
        "active_time": active_time,
        "reps": sum(st.session_state.rep_totals.values()),
        "sets": sum(st.session_state.set_totals.values()),
        "activities": sorted(
            {item["activity"] for item in st.session_state.history if item["activity"] != "REST"}
        ),
    }

    st.session_state.workout_running = False
    st.session_state.current_activity = "READY"
    st.session_state.activity_started = None
    st.session_state.recent_predictions = []


# ============================================================
# AVATAR MEDIA
# ============================================================

def avatar_media_path(gender: str, activity: str) -> Path | None:
    slug = ACTIVITY_SLUGS.get(activity, "workout")
    folder = ASSETS_DIR / gender

    # Put your real animated files here.
    for ext in (".gif", ".webp", ".mp4", ".png"):
        path = folder / f"{slug}{ext}"
        if path.exists():
            return path

    default = folder / "default.png"
    return default if default.exists() else None


def show_current_avatar() -> None:
    gender = st.session_state.avatar or "female"
    activity = st.session_state.current_activity
    path = avatar_media_path(gender, activity)

    if path is not None:
        if path.suffix.lower() == ".mp4":
            st.video(str(path), autoplay=True, loop=True, muted=True)
        else:
            st.image(str(path), use_container_width=True)
    else:
        st.markdown("### 🏋️")


# ============================================================
# STREAMLIT V2 BROWSER SENSOR COMPONENT
# ============================================================

SENSOR_HTML = r"""
<div class="sensor-shell">
  <div class="sensor-head">
    <div>
      <div class="sensor-kicker">PHONE MOTION SENSORS</div>
      <div class="sensor-title">Live Sensor Monitor</div>
    </div>
    <div id="sensorBadge" class="status off">● OFF</div>
  </div>

  <div class="controls">
    <button id="enableBtn">Enable Phone Sensors</button>
    <button id="calibrateBtn" disabled>Calibrate</button>
    <button id="startBtn" disabled>Start Workout</button>
    <button id="endBtn" disabled>End Workout</button>
  </div>

  <div id="message" class="message">
    Enable phone sensors to begin.
  </div>

  <div class="grid">
    <div class="monitor">
      <div class="monitor-top">
        <strong>Accelerometer (m/s²)</strong>
        <span id="accelLive">OFF</span>
      </div>
      <canvas id="accelCanvas"></canvas>
      <div class="values">
        <span>X <b id="ax">0.00</b></span>
        <span>Y <b id="ay">0.00</b></span>
        <span>Z <b id="az">0.00</b></span>
      </div>
    </div>

    <div class="monitor">
      <div class="monitor-top">
        <strong>Rotation Rate (°/s)</strong>
        <span id="gyroLive">OFF</span>
      </div>
      <canvas id="gyroCanvas"></canvas>
      <div class="values">
        <span>α <b id="ga">0.00</b></span>
        <span>β <b id="gb">0.00</b></span>
        <span>γ <b id="gg">0.00</b></span>
      </div>
    </div>
  </div>

  <div class="not-ecg">Motion-sensor waveform • NOT ECG / heart-rate data</div>
</div>
"""

SENSOR_CSS = r"""
.sensor-shell {
  color:#f7f8fc;
  border:1px solid rgba(255,255,255,.08);
  background:linear-gradient(145deg,#0b1930,#071426);
  border-radius:22px;
  padding:16px;
  font-family:Inter,system-ui,sans-serif;
}
.sensor-head {
  display:flex; justify-content:space-between; gap:12px; align-items:center;
}
.sensor-kicker { color:#8ea3c1; font-size:11px; font-weight:800; letter-spacing:.14em; }
.sensor-title { font-size:20px; font-weight:900; margin-top:3px; }
.status { border-radius:999px; padding:6px 10px; font-size:11px; font-weight:900; }
.status.off { color:#9dafc8; border:1px solid #34445d; }
.status.live { color:#63e7aa; border:1px solid rgba(74,221,151,.35); background:rgba(74,221,151,.08); }
.controls { display:grid; grid-template-columns:repeat(2,1fr); gap:9px; margin-top:14px; }
button {
  border-radius:13px; min-height:44px; border:1px solid rgba(255,255,255,.10);
  background:#13233d; color:#f7f8fc; font-weight:800; cursor:pointer;
}
button:first-child, #startBtn { background:linear-gradient(90deg,#ff4d9e,#9a48ff); border:none; }
button:disabled { opacity:.38; cursor:not-allowed; }
.message { margin:12px 0; color:#9fb2cd; font-size:13px; }
.grid { display:grid; grid-template-columns:1fr; gap:12px; }
.monitor {
  border:1px solid rgba(255,255,255,.07);
  background:#061120; border-radius:17px; padding:12px;
}
.monitor-top { display:flex; justify-content:space-between; align-items:center; font-size:13px; }
.monitor-top span { color:#62e8aa; font-size:11px; font-weight:800; }
canvas { width:100%; height:120px; display:block; margin-top:8px; }
.values { display:flex; justify-content:space-between; gap:6px; margin-top:8px; font-size:12px; color:#9fb2cd; }
.values b { color:#fff; }
.not-ecg { color:#7288a7; font-size:11px; margin-top:10px; }
@media (min-width:700px) {
  .grid { grid-template-columns:1fr 1fr; }
  .controls { grid-template-columns:repeat(4,1fr); }
}
"""

SENSOR_JS = r"""
export default function(component) {
  const { parentElement, setTriggerValue, data } = component;

  if (!parentElement.__gymsense) {
    const S = {
      enabled:false,
      calibrated:false,
      workout:false,
      buffer:[],
      lastEmit:0,
      accelHistory:[],
      gyroHistory:[],
      listener:null,
      raf:null
    };
    parentElement.__gymsense = S;

    const q = (s) => parentElement.querySelector(s);
    const enableBtn = q("#enableBtn");
    const calibrateBtn = q("#calibrateBtn");
    const startBtn = q("#startBtn");
    const endBtn = q("#endBtn");
    const message = q("#message");
    const badge = q("#sensorBadge");

    const axEl=q("#ax"), ayEl=q("#ay"), azEl=q("#az");
    const gaEl=q("#ga"), gbEl=q("#gb"), ggEl=q("#gg");
    const accelLive=q("#accelLive"), gyroLive=q("#gyroLive");

    function setMsg(text) { message.textContent = text; }

    function setLive(live) {
      badge.textContent = live ? "● LIVE" : "● OFF";
      badge.className = live ? "status live" : "status off";
      accelLive.textContent = live ? "LIVE" : "OFF";
      gyroLive.textContent = live ? "LIVE" : "OFF";
    }

    function pushHistory(arr, triplet) {
      arr.push(triplet);
      if (arr.length > 90) arr.splice(0, arr.length - 90);
    }

    function draw(canvas, history, colors) {
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      const width = Math.max(280, rect.width);
      const height = 120;
      canvas.width = width * ratio;
      canvas.height = height * ratio;

      const ctx = canvas.getContext("2d");
      ctx.scale(ratio, ratio);
      ctx.clearRect(0,0,width,height);

      ctx.strokeStyle = "rgba(110,145,180,.14)";
      ctx.lineWidth = 1;
      for (let i=0;i<=6;i++) {
        const y = (height/6)*i;
        ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(width,y); ctx.stroke();
      }
      for (let i=0;i<=8;i++) {
        const x = (width/8)*i;
        ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,height); ctx.stroke();
      }

      if (history.length < 2) return;

      const flat = history.flat().filter(Number.isFinite);
      const maxAbs = Math.max(1, ...flat.map(v => Math.abs(v)));
      const center = height/2;

      for (let axis=0; axis<3; axis++) {
        ctx.strokeStyle = colors[axis];
        ctx.lineWidth = 2;
        ctx.beginPath();
        history.forEach((row, i) => {
          const x = history.length === 1 ? 0 : i * width/(history.length-1);
          const y = center - (row[axis]/maxAbs)*(height*0.42);
          if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
        });
        ctx.stroke();
      }
    }

    function renderLoop() {
      draw(q("#accelCanvas"), S.accelHistory, ["#ff5f79","#43b5ff","#39e69b"]);
      draw(q("#gyroCanvas"), S.gyroHistory, ["#ff5ac4","#45e7f2","#ffd45a"]);
      S.raf = requestAnimationFrame(renderLoop);
    }
    renderLoop();

    async function enableSensors() {
      try {
        if (!("DeviceMotionEvent" in window)) {
          setMsg("Motion sensors are not available in this browser/device.");
          setTriggerValue("sensor_status", {status:"UNSUPPORTED", ts:Date.now()});
          return;
        }

        if (typeof DeviceMotionEvent.requestPermission === "function") {
          const permission = await DeviceMotionEvent.requestPermission();
          if (permission !== "granted") {
            setMsg("Motion sensor permission was denied.");
            setTriggerValue("sensor_status", {status:"DENIED", ts:Date.now()});
            return;
          }
        }

        if (!S.listener) {
          S.listener = (event) => {
            const a = event.accelerationIncludingGravity || event.acceleration || {};
            const r = event.rotationRate || {};

            const sample = {
              accel_x: Number(a.x || 0),
              accel_y: Number(a.y || 0),
              accel_z: Number(a.z || 0),
              gyro_alpha: Number(r.alpha || 0),
              gyro_beta: Number(r.beta || 0),
              gyro_gamma: Number(r.gamma || 0),
              timestamp_ms: Date.now()
            };

            axEl.textContent = sample.accel_x.toFixed(2);
            ayEl.textContent = sample.accel_y.toFixed(2);
            azEl.textContent = sample.accel_z.toFixed(2);
            gaEl.textContent = sample.gyro_alpha.toFixed(2);
            gbEl.textContent = sample.gyro_beta.toFixed(2);
            ggEl.textContent = sample.gyro_gamma.toFixed(2);

            pushHistory(S.accelHistory, [sample.accel_x, sample.accel_y, sample.accel_z]);
            pushHistory(S.gyroHistory, [sample.gyro_alpha, sample.gyro_beta, sample.gyro_gamma]);

            if (S.workout) {
              S.buffer.push(sample);

              const now = Date.now();
              if (S.buffer.length >= 20 && now - S.lastEmit >= 1800) {
                setTriggerValue("window", {
                  samples: S.buffer.slice(-120),
                  ts: now
                });
                S.buffer = S.buffer.slice(-6);
                S.lastEmit = now;
              }
            }
          };

          window.addEventListener("devicemotion", S.listener, {passive:true});
        }

        S.enabled = true;
        setLive(true);
        enableBtn.disabled = true;
        enableBtn.textContent = "✓ Sensors Enabled";
        calibrateBtn.disabled = false;
        setMsg("Accelerometer and rotation-rate stream connected.");
        setTriggerValue("sensor_status", {status:"LIVE", ts:Date.now()});
      } catch (err) {
        setMsg("Sensor error: " + (err?.message || String(err)));
        setTriggerValue("sensor_status", {status:"ERROR", message:String(err), ts:Date.now()});
      }
    }

    function calibrate() {
      if (!S.enabled) return;
      calibrateBtn.disabled = true;
      let n = 3;
      setMsg("Keep the phone still… 3");

      const t = setInterval(() => {
        n -= 1;
        if (n > 0) {
          setMsg("Keep the phone still… " + n);
        } else {
          clearInterval(t);
          S.calibrated = true;
          calibrateBtn.textContent = "✓ Calibrated";
          startBtn.disabled = false;
          setMsg("Calibration complete. Ready to start.");
          setTriggerValue("sensor_status", {status:"CALIBRATED", ts:Date.now()});
        }
      }, 1000);
    }

    function startWorkout() {
      if (!S.enabled || !S.calibrated) return;
      S.workout = true;
      S.buffer = [];
      S.lastEmit = 0;
      startBtn.disabled = true;
      endBtn.disabled = false;
      setMsg("Workout in progress. Exercise recognition is active.");
      setTriggerValue("session_event", {event:"START", ts:Date.now()});
    }

    function endWorkout() {
      if (!S.workout) return;
      S.workout = false;
      endBtn.disabled = true;
      startBtn.disabled = false;
      setMsg("Workout ended. Session is being summarized.");
      setTriggerValue("session_event", {event:"END", ts:Date.now()});
    }

    enableBtn.addEventListener("click", enableSensors);
    calibrateBtn.addEventListener("click", calibrate);
    startBtn.addEventListener("click", startWorkout);
    endBtn.addEventListener("click", endWorkout);

    S.cleanup = () => {
      if (S.listener) window.removeEventListener("devicemotion", S.listener);
      if (S.raf) cancelAnimationFrame(S.raf);
    };
  }

  const S = parentElement.__gymsense;

  // Keep display aligned with Python state after Streamlit reruns.
  if (data?.workoutRunning === false && S.workout) {
    S.workout = false;
  }

  return () => {
    // Don't aggressively destroy sensor listeners on ordinary Streamlit rerenders.
    // Cleanup is only used when the component is truly unmounted.
  };
}
"""

sensor_component = st.components.v2.component(
    "gymsense_sensor_bridge",
    html=SENSOR_HTML,
    css=SENSOR_CSS,
    js=SENSOR_JS,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    f"""
    <div class="gs-top">
      <div>
        <div class="gs-brand">GymSense <span>AI</span></div>
        <div class="gs-sub">AI-Powered Workout Activity Recognition</div>
      </div>
      <div class="gs-live">● {st.session_state.sensor_status}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# AVATAR SELECTION — EVERY NEW STREAMLIT SESSION
# ============================================================

if st.session_state.avatar is None:
    st.markdown(
        """
        <div class="gs-card">
          <div class="gs-kicker">WELCOME</div>
          <div class="gs-title" style="font-size:2.5rem">Choose Your Avatar</div>
          <div class="gs-sub">Choose again whenever you open a new app session.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    female_col, male_col = st.columns(2)

    with female_col:
        st.image(str(ASSETS_DIR / "female" / "default.png"), use_container_width=True)
        if st.button("Female Avatar", use_container_width=True, type="primary"):
            st.session_state.avatar = "female"
            st.rerun()

    with male_col:
        st.image(str(ASSETS_DIR / "male" / "default.png"), use_container_width=True)
        if st.button("Male Avatar", use_container_width=True):
            st.session_state.avatar = "male"
            st.rerun()

    st.stop()


# ============================================================
# MAIN WORKOUT CARD
# ============================================================

left, right = st.columns([1.0, 1.05], gap="large")

with left:
    show_current_avatar()

with right:
    activity = st.session_state.current_activity
    confidence = st.session_state.current_confidence

    st.markdown('<div class="gs-kicker">CURRENT ACTIVITY</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="gs-title">{activity}</div>', unsafe_allow_html=True)

    if st.session_state.current_confidence_label:
        if confidence is None:
            st.caption(st.session_state.current_confidence_label)
        else:
            st.caption(
                f"{st.session_state.current_confidence_label}: "
                f"{confidence * 100:.1f}%"
            )

    elapsed = 0
    if st.session_state.workout_running and st.session_state.activity_started:
        elapsed = int(time.time() - st.session_state.activity_started)

    mins, secs = divmod(elapsed, 60)

    m1, m2, m3 = st.columns(3)
    m1.metric("Duration", f"{mins:02d}:{secs:02d}")

    if activity in REPETITIVE_ACTIVITIES:
        m2.metric("Reps", st.session_state.rep_totals.get(activity, 0))
        m3.metric("Set", st.session_state.set_totals.get(activity, 0))
    else:
        m2.metric("Reps", "—")
        m3.metric("Set", "—")

    if st.session_state.raw_prediction:
        st.caption(f"Raw ML class: {st.session_state.raw_prediction}")


# ============================================================
# SENSOR COMPONENT
# ============================================================

result = sensor_component(
    data={
        "workoutRunning": bool(st.session_state.workout_running),
        "avatar": st.session_state.avatar,
        "prediction": st.session_state.current_activity,
    },
    key="gymsense-phone-sensors",
    on_sensor_status_change=lambda: None,
    on_session_event_change=lambda: None,
    on_window_change=lambda: None,
)


# ============================================================
# HANDLE COMPONENT EVENTS
# ============================================================

sensor_status = getattr(result, "sensor_status", None)
if sensor_status:
    status = str(sensor_status.get("status", "")).upper()

    if status == "LIVE":
        st.session_state.sensor_status = "SENSOR LIVE"
    elif status == "CALIBRATED":
        st.session_state.sensor_status = "CALIBRATED"
        st.session_state.calibrated = True
    elif status:
        st.session_state.sensor_status = status

session_event = getattr(result, "session_event", None)
if session_event:
    event = str(session_event.get("event", "")).upper()

    if event == "START" and not st.session_state.workout_running:
        start_workout_python()
        st.rerun()

    if event == "END" and st.session_state.workout_running:
        end_workout_python()
        st.rerun()

window_event = getattr(result, "window", None)
if (
    window_event
    and st.session_state.workout_running
    and isinstance(window_event.get("samples"), list)
):
    records = window_event["samples"]

    if len(records) >= 10:
        prediction = predict_window(records)
        stable_activity = smooth_prediction(prediction["activity"])

        st.session_state.current_confidence = prediction["confidence"]
        st.session_state.current_confidence_label = prediction["confidence_label"]
        st.session_state.raw_prediction = prediction["raw_prediction"]

        change_activity(stable_activity)
        update_reps(records, stable_activity)

        st.rerun()


# ============================================================
# HISTORY
# ============================================================

st.markdown("### Activity History")

if not st.session_state.history:
    st.caption("Completed activity segments will appear here during the workout.")
else:
    rows = []
    for item in reversed(st.session_state.history[-20:]):
        start_txt = time.strftime("%I:%M:%S %p", time.localtime(item["start"]))
        end_txt = time.strftime("%I:%M:%S %p", time.localtime(item["end"]))
        duration = int(item["duration"])
        mins, secs = divmod(duration, 60)

        rows.append(
            {
                "Activity": item["activity"],
                "Start": start_txt,
                "End": end_txt,
                "Duration": f"{mins}m {secs}s",
                "Reps": item["reps"] if item["reps"] else "—",
                "Confidence": (
                    f"{item['confidence'] * 100:.1f}%"
                    if item["confidence"] is not None
                    else "—"
                ),
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ============================================================
# SESSION SUMMARY
# ============================================================

if st.session_state.last_session_summary:
    s = st.session_state.last_session_summary

    st.markdown("### 🏆 Workout Complete")
    a, b, c, d = st.columns(4)
    a.metric("Workout Time", f"{int(s['duration']//60)}m {int(s['duration']%60)}s")
    b.metric("Active Time", f"{int(s['active_time']//60)}m {int(s['active_time']%60)}s")
    c.metric("Total Reps", s["reps"])
    d.metric("Total Sets", s["sets"])

    st.write("Exercises detected:", ", ".join(s["activities"]) if s["activities"] else "None")


# ============================================================
# MODEL / JUDGE SECTION
# ============================================================

with st.expander("🧠 ML Model & Competition Details"):
    try:
        ml = load_ml()
        meta = ml["metadata"]

        st.write("**Selected model:**", meta.get("selected_model", ml["model"].__class__.__name__))
        st.write("**Classes:**", ", ".join(meta.get("classes", [])))
        st.write("**Features:**", len(meta.get("feature_names", [])))

        metrics = meta.get("evaluation_metrics", {})
        if metrics:
            st.write(
                "**Saved evaluation:**",
                {
                    "Accuracy": metrics.get("accuracy"),
                    "Precision": metrics.get("precision"),
                    "Recall": metrics.get("recall"),
                    "F1": metrics.get("f1_score"),
                },
            )

            values = [
                metrics.get("accuracy"),
                metrics.get("precision"),
                metrics.get("recall"),
                metrics.get("f1_score"),
            ]
            numeric = [float(v) for v in values if v is not None]
            if numeric and min(numeric) >= 99.9:
                st.warning(
                    "Your saved evaluation is extremely high. "
                    "If the current dataset is synthetic/highly controlled, "
                    "collect real phone sensor data before presenting this as real-world accuracy."
                )

        comparison = meta.get("model_comparison", {})
        if comparison:
            comp_rows = []
            for name, values in comparison.items():
                comp_rows.append(
                    {
                        "Model": name,
                        "Accuracy": values.get("accuracy"),
                        "Precision": values.get("precision"),
                        "Recall": values.get("recall"),
                        "F1": values.get("f1_score"),
                    }
                )
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    except Exception as exc:
        st.error(f"Could not load model information: {exc}")


with st.expander("🎬 Avatar animation files"):
    st.write(
        "The app automatically looks for exercise-specific animated files in:"
    )
    st.code(
        """assets/avatars/female/bicep_curl.gif
assets/avatars/female/squat.gif
assets/avatars/female/walking.gif
...
assets/avatars/male/bicep_curl.gif
assets/avatars/male/squat.gif
assets/avatars/male/walking.gif
..."""
    )
    st.info(
        "If an animation file is missing, the app uses the selected avatar's default image. "
        "Use GIF, animated WebP, or MP4 files for the high-quality moving character you want."
    )

st.caption(
    "GymSense AI • Ranjitha B K • 1SB24AI041 • AIML • MACHINE SPECTRA 1.0"
)
