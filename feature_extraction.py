"""
GYMSENSE AI - Feature Extraction Pipeline
Student: Ranjitha B K | USN: 1SB24AI041
Branch: AIML | Event: MACHINE SPECTRA 1.0

Extracts statistical, kinematic, and temporal features from sliding sensor windows
of Accelerometer (X, Y, Z) and Gyroscope (Alpha, Beta, Gamma) data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union


def compute_rms(signal: np.ndarray) -> float:
    """Compute Root Mean Square (RMS) of a signal."""
    return float(np.sqrt(np.mean(signal**2)))


def compute_energy(signal: np.ndarray) -> float:
    """Compute energy of a signal (mean squared value)."""
    return float(np.mean(signal**2))


def compute_zero_crossings(signal: np.ndarray) -> float:
    """Compute zero-crossing rate of a centered signal."""
    centered = signal - np.mean(signal)
    if len(centered) < 2:
        return 0.0
    zero_crosses = np.sum((centered[:-1] * centered[1:]) < 0)
    return float(zero_crosses / (len(signal) - 1))


def compute_iqr(signal: np.ndarray) -> float:
    """Compute Interquartile Range (IQR = Q3 - Q1)."""
    q75, q25 = np.percentile(signal, [75, 25])
    return float(q75 - q25)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation coefficient safely avoiding divide-by-zero."""
    std_x = np.std(x)
    std_y = np.std(y)
    if std_x < 1e-6 or std_y < 1e-6:
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    return 0.0 if np.isnan(corr) else float(corr)


def extract_features_from_window(
    accel_x: np.ndarray,
    accel_y: np.ndarray,
    accel_z: np.ndarray,
    gyro_alpha: np.ndarray,
    gyro_beta: np.ndarray,
    gyro_gamma: np.ndarray
) -> Dict[str, float]:
    """
    Extract 46 kinematic and statistical features from 3D Accelerometer and 3D Gyroscope windows.
    
    Parameters:
        accel_x, accel_y, accel_z: 1D numpy arrays (m/s^2)
        gyro_alpha, gyro_beta, gyro_gamma: 1D numpy arrays (deg/s)
        
    Returns:
        Dictionary mapping feature_name -> float value
    """
    # Ensure 1D float arrays
    ax = np.asarray(accel_x, dtype=np.float64)
    ay = np.asarray(accel_y, dtype=np.float64)
    az = np.asarray(accel_z, dtype=np.float64)
    ga = np.asarray(gyro_alpha, dtype=np.float64)
    gb = np.asarray(gyro_beta, dtype=np.float64)
    gg = np.asarray(gyro_gamma, dtype=np.float64)

    # Accelerometer Magnitude |A|
    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
    # Gyroscope Magnitude |G|
    gyro_mag = np.sqrt(ga**2 + gb**2 + gg**2)

    features = {}

    # --- Accelerometer Axis Features (X, Y, Z, Mag) ---
    accel_signals = {
        'accel_x': ax,
        'accel_y': ay,
        'accel_z': az,
        'accel_mag': accel_mag
    }

    for name, sig in accel_signals.items():
        features[f"{name}_mean"] = float(np.mean(sig))
        features[f"{name}_std"] = float(np.std(sig))
        features[f"{name}_var"] = float(np.var(sig))
        features[f"{name}_min"] = float(np.min(sig))
        features[f"{name}_max"] = float(np.max(sig))
        features[f"{name}_range"] = float(np.ptp(sig))
        features[f"{name}_rms"] = compute_rms(sig)
        features[f"{name}_energy"] = compute_energy(sig)
        features[f"{name}_iqr"] = compute_iqr(sig)
        features[f"{name}_zc"] = compute_zero_crossings(sig)

    # Cross-axis correlations for Accelerometer
    features['accel_corr_xy'] = safe_corr(ax, ay)
    features['accel_corr_xz'] = safe_corr(ax, az)
    features['accel_corr_yz'] = safe_corr(ay, az)

    # --- Gyroscope Axis Features (Alpha, Beta, Gamma, Mag) ---
    gyro_signals = {
        'gyro_alpha': ga,
        'gyro_beta': gb,
        'gyro_gamma': gg,
        'gyro_mag': gyro_mag
    }

    for name, sig in gyro_signals.items():
        features[f"{name}_mean"] = float(np.mean(sig))
        features[f"{name}_std"] = float(np.std(sig))
        features[f"{name}_var"] = float(np.var(sig))
        features[f"{name}_min"] = float(np.min(sig))
        features[f"{name}_max"] = float(np.max(sig))
        features[f"{name}_range"] = float(np.ptp(sig))
        features[f"{name}_rms"] = compute_rms(sig)
        features[f"{name}_energy"] = compute_energy(sig)

    # Cross-axis correlations for Gyroscope
    features['gyro_corr_ab'] = safe_corr(ga, gb)
    features['gyro_corr_ag'] = safe_corr(ga, gg)
    features['gyro_corr_bg'] = safe_corr(gb, gg)

    return features


def extract_features_from_records(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Extract features from a list of raw sensor dictionary records.
    Expected keys: accel_x, accel_y, accel_z, gyro_alpha, gyro_beta, gyro_gamma
    """
    if not records or len(records) == 0:
        raise ValueError("Sensor records list is empty")

    ax = np.array([r.get('accel_x', r.get('x', 0.0)) for r in records])
    ay = np.array([r.get('accel_y', r.get('y', 0.0)) for r in records])
    az = np.array([r.get('accel_z', r.get('z', 9.81)) for r in records])
    ga = np.array([r.get('gyro_alpha', r.get('alpha', 0.0)) for r in records])
    gb = np.array([r.get('gyro_beta', r.get('beta', 0.0)) for r in records])
    gg = np.array([r.get('gyro_gamma', r.get('gamma', 0.0)) for r in records])

    return extract_features_from_window(ax, ay, az, ga, gb, gg)


def get_feature_names() -> List[str]:
    """Return ordered list of all feature names."""
    dummy = np.zeros(20)
    dummy_feat = extract_features_from_window(dummy, dummy, dummy, dummy, dummy, dummy)
    return list(dummy_feat.keys())


if __name__ == "__main__":
    feat_names = get_feature_names()
    print(f"Total features extracted per window: {len(feat_names)}")
    print("Features sample:", feat_names[:10])
