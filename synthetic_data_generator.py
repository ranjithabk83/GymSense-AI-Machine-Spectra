"""
GYMSENSE AI - Biomechanical Sensor Data Synthesizer
Student: Ranjitha B K | USN: 1SB24AI041
Branch: AIML | Event: MACHINE SPECTRA 1.0

Generates realistic, physically-grounded continuous time-series sensor data
for 11 workout activity classes + rest based on human kinematics and sensor physics.
"""

import os
import numpy as np
import pandas as pd

# Exercise classes
EXERCISE_CLASSES = [
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
    "LATERAL RAISE"
]

SAMPLE_RATE_HZ = 20  # 20 samples per second (50ms interval)
DURATION_PER_SAMPLE_SEC = 25  # 25 seconds per recording session
REPETITIONS_PER_CLASS = 18  # 18 distinct subjects/sessions per exercise class


def generate_session_data(activity: str, duration_sec: float, sampling_rate: int, seed: int = 42) -> pd.DataFrame:
    """Generate time-series sensor stream for a single workout session."""
    np.random.seed(seed)
    n_points = int(duration_sec * sampling_rate)
    t = np.linspace(0, duration_sec, n_points)
    dt = 1.0 / sampling_rate

    # Base gravity vector (9.80665 m/s^2)
    g = 9.80665

    # Realistic sensor noise floor + drift + subtle motion jitter
    noise_ax = np.random.normal(0, 0.28, n_points)
    noise_ay = np.random.normal(0, 0.28, n_points)
    noise_az = np.random.normal(0, 0.28, n_points)
    noise_ga = np.random.normal(0, 2.2, n_points)
    noise_gb = np.random.normal(0, 2.2, n_points)
    noise_gg = np.random.normal(0, 2.2, n_points)

    # Subject variation parameters (tempo jitter, amplitude variability, fatigue decay)
    tempo_jitter = np.random.uniform(0.85, 1.18)
    amp_jitter = np.random.uniform(0.80, 1.25)
    fatigue = 1.0 - 0.05 * (t / duration_sec)  # slight fatigue over duration

    if activity == "REST":
        # Standing or sitting still with subtle breathing and natural micro-postural sway
        breath_freq = 0.25 * tempo_jitter
        ax = 0.2 * np.sin(2 * np.pi * breath_freq * t) + noise_ax
        ay = 0.4 * np.cos(2 * np.pi * breath_freq * t) + noise_ay
        az = g + 0.15 * np.sin(2 * np.pi * breath_freq * t + 0.5) + noise_az
        ga = 0.5 * np.sin(2 * np.pi * breath_freq * t) + noise_ga
        gb = 0.7 * np.cos(2 * np.pi * breath_freq * t) + noise_gb
        gg = 0.4 * np.sin(2 * np.pi * breath_freq * t) + noise_gg

    elif activity == "WALKING":
        freq = 1.8 * tempo_jitter  # Step cadence ~1.8 Hz
        # Rhythmic vertical bounce, forward surge, and lateral sway
        ax = amp_jitter * 1.2 * np.sin(np.pi * freq * t) + noise_ax  # Lateral sway at half frequency
        ay = g + amp_jitter * 2.8 * np.sin(2 * np.pi * freq * t) + noise_ay  # Vertical bounce
        az = amp_jitter * 2.0 * np.cos(2 * np.pi * freq * t + 0.3) + noise_az  # Forward/back surge
        ga = amp_jitter * 25.0 * np.sin(np.pi * freq * t) + noise_ga  # Yaw
        gb = amp_jitter * 45.0 * np.sin(2 * np.pi * freq * t) + noise_gb  # Pitch
        gg = amp_jitter * 35.0 * np.cos(np.pi * freq * t) + noise_gg  # Roll

    elif activity == "RUNNING":
        freq = 3.0 * tempo_jitter  # High step cadence ~3.0 Hz
        # High impact accelerations with harmonic spikes
        impact = np.exp(-((t % (1.0 / freq)) * freq * 4.0)) * 6.0
        ax = amp_jitter * (2.8 * np.sin(np.pi * freq * t)) + noise_ax
        ay = g + amp_jitter * (7.5 * np.sin(2 * np.pi * freq * t) + impact) + noise_ay
        az = amp_jitter * (5.5 * np.cos(2 * np.pi * freq * t + 0.2)) + noise_az
        ga = amp_jitter * 60.0 * np.sin(np.pi * freq * t) + noise_ga
        gb = amp_jitter * 110.0 * np.sin(2 * np.pi * freq * t) + noise_gb
        gg = amp_jitter * 85.0 * np.cos(np.pi * freq * t) + noise_gg

    elif activity == "BICEP CURL":
        freq = 0.42 * tempo_jitter  # ~2.4 seconds per repetition
        # Forearm rotation flips gravity projection from Y-dominant to Z-dominant
        curl_phase = 2 * np.pi * freq * t
        ax = amp_jitter * 0.7 * np.sin(curl_phase) + noise_ax
        ay = g * np.cos(0.9 * np.sin(curl_phase)) + amp_jitter * 2.2 * np.sin(curl_phase) + noise_ay
        az = g * np.sin(0.9 * np.sin(curl_phase)) + amp_jitter * 3.5 * np.cos(curl_phase) + noise_az
        ga = amp_jitter * 15.0 * np.sin(curl_phase) + noise_ga
        gb = amp_jitter * 95.0 * np.cos(curl_phase) + noise_gb  # Dominant pitch rotation
        gg = amp_jitter * 20.0 * np.sin(curl_phase + 0.4) + noise_gg

    elif activity == "HAMMER CURL":
        freq = 0.42 * tempo_jitter
        curl_phase = 2 * np.pi * freq * t
        # Neutral wrist orientation shifts the rotation vector to roll/yaw plane
        ax = amp_jitter * 1.8 * np.sin(curl_phase) + noise_ax
        ay = g * np.cos(0.85 * np.sin(curl_phase)) + amp_jitter * 2.5 * np.sin(curl_phase) + noise_ay
        az = g * np.sin(0.4 * np.sin(curl_phase)) + amp_jitter * 2.0 * np.cos(curl_phase) + noise_az
        ga = amp_jitter * 45.0 * np.sin(curl_phase) + noise_ga
        gb = amp_jitter * 30.0 * np.cos(curl_phase) + noise_gb
        gg = amp_jitter * 80.0 * np.cos(curl_phase) + noise_gg  # Dominant roll rotation

    elif activity == "SQUAT":
        freq = 0.36 * tempo_jitter  # ~2.8 seconds per repetition
        squat_phase = 2 * np.pi * freq * t
        # Descent (deceleration) -> bottom pause -> explosive drive upwards
        vertical_acc = -3.8 * np.sin(squat_phase) + 1.8 * np.sin(2 * squat_phase)
        torso_tilt = 0.45 * np.maximum(0.0, np.sin(squat_phase))
        ax = amp_jitter * 0.6 * np.sin(squat_phase) + noise_ax
        ay = (g * np.cos(torso_tilt)) + amp_jitter * vertical_acc + noise_ay
        az = (g * np.sin(torso_tilt)) + amp_jitter * 1.5 * np.cos(squat_phase) + noise_az
        ga = amp_jitter * 10.0 * np.sin(squat_phase) + noise_ga
        gb = amp_jitter * 55.0 * np.sin(squat_phase) + noise_gb  # Torso inclination angle
        gg = amp_jitter * 12.0 * np.cos(squat_phase) + noise_gg

    elif activity == "LUNGE":
        freq = 0.34 * tempo_jitter  # ~2.9 seconds per repetition
        lunge_phase = 2 * np.pi * freq * t
        # Asymmetrical step forward + drop + recover
        step_surge = 2.4 * np.cos(lunge_phase) * (np.sin(lunge_phase) > 0)
        ax = amp_jitter * 1.2 * np.sin(lunge_phase) + noise_ax
        ay = g + amp_jitter * (-3.2 * np.sin(lunge_phase) + 1.2 * np.cos(2 * lunge_phase)) + noise_ay
        az = amp_jitter * (step_surge + 1.8 * np.sin(lunge_phase)) + noise_az
        ga = amp_jitter * 25.0 * np.sin(lunge_phase) + noise_ga
        gb = amp_jitter * 48.0 * np.sin(lunge_phase + 0.3) + noise_gb
        gg = amp_jitter * 32.0 * np.cos(lunge_phase) + noise_gg

    elif activity == "JUMPING JACK":
        freq = 1.35 * tempo_jitter  # ~0.74 seconds per jump
        jack_phase = 2 * np.pi * freq * t
        # Massive synchronized lateral arm arcs and vertical jumps
        ax = amp_jitter * 6.5 * np.sin(jack_phase) + noise_ax  # Extreme lateral acceleration
        ay = g + amp_jitter * 7.8 * np.sin(jack_phase + 0.2) + noise_ay  # Vertical hop
        az = amp_jitter * 3.0 * np.cos(jack_phase) + noise_az
        ga = amp_jitter * 75.0 * np.cos(jack_phase) + noise_ga
        gb = amp_jitter * 40.0 * np.sin(jack_phase) + noise_gb
        gg = amp_jitter * 120.0 * np.sin(jack_phase) + noise_gg  # Huge lateral coronal rotation

    elif activity == "SHOULDER PRESS":
        freq = 0.40 * tempo_jitter  # ~2.5 seconds per repetition
        press_phase = 2 * np.pi * freq * t
        # Starting at shoulder height -> upward vertical drive against gravity -> lower
        thrust = 4.2 * np.sin(press_phase) - 1.5 * np.sin(2 * press_phase)
        ax = amp_jitter * 0.8 * np.sin(press_phase) + noise_ax
        ay = g + amp_jitter * thrust + noise_ay
        az = amp_jitter * 1.4 * np.cos(press_phase) + noise_az
        ga = amp_jitter * 18.0 * np.sin(press_phase) + noise_ga
        gb = amp_jitter * 32.0 * np.cos(press_phase) + noise_gb
        gg = amp_jitter * 22.0 * np.sin(press_phase) + noise_gg

    elif activity == "FRONT RAISE":
        freq = 0.38 * tempo_jitter  # ~2.6 seconds per repetition
        raise_phase = 2 * np.pi * freq * t
        # Sagittal forward arm raise creates dynamic tangential and centripetal acceleration in Y and Z
        ax = amp_jitter * 0.5 * np.sin(raise_phase) + noise_ax
        ay = g * np.cos(0.9 * np.maximum(0.0, np.sin(raise_phase))) + amp_jitter * 2.8 * np.sin(raise_phase) + noise_ay
        az = g * np.sin(0.9 * np.maximum(0.0, np.sin(raise_phase))) + amp_jitter * 4.2 * np.cos(raise_phase) + noise_az
        ga = amp_jitter * 14.0 * np.sin(raise_phase) + noise_ga
        gb = amp_jitter * 85.0 * np.cos(raise_phase) + noise_gb  # High pitch angular sweep
        gg = amp_jitter * 15.0 * np.sin(raise_phase) + noise_ga

    elif activity == "LATERAL RAISE":
        freq = 0.38 * tempo_jitter
        raise_phase = 2 * np.pi * freq * t
        # Coronal sideways arm raise creates strong lateral X acceleration and roll angular velocity
        ax = amp_jitter * 4.8 * np.sin(raise_phase) + noise_ax
        ay = g * np.cos(0.8 * np.maximum(0.0, np.sin(raise_phase))) + amp_jitter * 2.5 * np.sin(raise_phase) + noise_ay
        az = amp_jitter * 0.9 * np.cos(raise_phase) + noise_az
        ga = amp_jitter * 20.0 * np.sin(raise_phase) + noise_ga
        gb = amp_jitter * 18.0 * np.sin(raise_phase) + noise_gb
        gg = amp_jitter * 82.0 * np.cos(raise_phase) + noise_gg  # High roll angular sweep

    else:
        raise ValueError(f"Unknown activity: {activity}")

    df = pd.DataFrame({
        "timestamp_ms": (t * 1000).astype(int),
        "accel_x": np.round(ax, 4),
        "accel_y": np.round(ay, 4),
        "accel_z": np.round(az, 4),
        "gyro_alpha": np.round(ga, 4),
        "gyro_beta": np.round(gb, 4),
        "gyro_gamma": np.round(gg, 4),
        "activity": activity
    })
    return df


def generate_full_dataset(output_path: str = "data/sensor_data.csv"):
    """Generate and save multi-session dataset for all classes."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_sessions = []
    session_id = 0

    print(f"Starting sensor dataset synthesis...")
    print(f"Target classes ({len(EXERCISE_CLASSES)}): {EXERCISE_CLASSES}")

    for activity in EXERCISE_CLASSES:
        for rep in range(REPETITIONS_PER_CLASS):
            seed = session_id * 137 + 42
            df_sess = generate_session_data(
                activity=activity,
                duration_sec=DURATION_PER_SAMPLE_SEC,
                sampling_rate=SAMPLE_RATE_HZ,
                seed=seed
            )
            df_sess["session_id"] = session_id
            all_sessions.append(df_sess)
            session_id += 1

    dataset = pd.concat(all_sessions, ignore_index=True)
    dataset.to_csv(output_path, index=False)
    print(f"[OK] Generated {len(dataset)} sensor rows across {session_id} sessions.")
    print(f"[OK] Saved raw sensor dataset to: {output_path}")
    return dataset


if __name__ == "__main__":
    generate_full_dataset()
