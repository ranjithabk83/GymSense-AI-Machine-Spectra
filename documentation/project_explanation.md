# GYMSENSE AI — AI-Powered Workout Activity Recognition

**Student Name:** Ranjitha B K  
**USN:** 1SB24AI041  
**Department:** Artificial Intelligence and Machine Learning  
**Event:** MACHINE SPECTRA 1.0  
**Date:** September 10, 2026  
**Project Track:** Gym Workout Tracker: Classifying the type of exercise being done based on accelerometer/gyroscope data

---

## 1. Executive Summary & Problem Formulation

Modern physical fitness tracking relies heavily on manual entry or generic step-counting heuristics that fail to distinguish complex strength-training movements. **GymSense AI** addresses this challenge by providing an end-to-end Machine Learning system that captures high-frequency 6-axis inertial motion signals (3-Axis Accelerometer and 3-Axis Gyroscope) from a smartphone, extracts kinematic features, and performs real-time classification across 11 discrete exercise disciplines and resting states.

The application is deployed as a Progressive Web Application (PWA) with a Python FastAPI inference backend, featuring real-time rep counting via peak detection, automated set transitions, hospital-style signal oscilloscopes, and a consistent 60 FPS skeletal avatar animation engine.

---

## 2. Sensor Kinematics & Physical Signal Modeling

Smartphones integrate Micro-Electro-Mechanical Systems (MEMS) sensors:
1. **Tri-Axial Accelerometer ($a_x, a_y, a_z$):** Measures linear acceleration and gravitational projection:
   $$|A| = \sqrt{a_x^2 + a_y^2 + a_z^2}$$
2. **Tri-Axial Gyroscope ($\omega_\alpha, \omega_\beta, \omega_\gamma$):** Measures angular velocity along yaw, pitch, and roll:
   $$|G| = \sqrt{\omega_\alpha^2 + \omega_\beta^2 + \omega_\gamma^2}$$

### Distinct Exercise Kinematic Signatures:
- **REST:** Constant static gravity vector ($~9.81\text{ m/s}^2$), minimal signal variance ($\sigma^2 < 0.12$).
- **BICEP CURL:** Forearm rotation creates a dynamic shift between $Y$ (vertical) and $Z$ (sagittal) axes accompanied by dominant pitch angular velocity ($\omega_\beta \approx 95^\circ/\text{s}$).
- **HAMMER CURL:** Neutral wrist orientation redirects rotational velocity into the roll plane ($\omega_\gamma \approx 80^\circ/\text{s}$) with minimal lateral sway.
- **SQUAT:** Vertical deceleration followed by explosive upward drive ($\Delta a_y \approx 5-8\text{ m/s}^2$) coupled with torso inclination pitch.
- **LUNGE:** Asymmetric stepping surge on the sagittal axis ($Z$) with vertical drop and delayed recovery.
- **JUMPING JACK:** High-frequency, synchronized lateral arm abduction ($a_x \text{ swings } \pm 6.5\text{ m/s}^2$) and high roll angular velocity ($\omega_\gamma \approx 120^\circ/\text{s}$).
- **SHOULDER PRESS:** Pure vertical acceleration against gravity with forearm stabilization.
- **FRONT RAISE:** Anterior arm elevation in sagittal plane generating tangential centripetal acceleration in $Z$.
- **LATERAL RAISE:** Coronal plane arm elevation with high lateral acceleration ($a_x$) and roll angular sweep.
- **WALKING / RUNNING:** Periodic gait harmonics (1.8 Hz vs 3.0 Hz) with characteristic ground reaction impact spikes.

---

## 3. Feature Extraction Pipeline

Continuous sensor streams are sampled at $20\text{ Hz}$ and segmented into sliding windows of **50 samples** ($2.5\text{ seconds}$) with a **50% overlap** ($1.25\text{ seconds}$ step).

From each window, **46 mathematical features** are computed:
- **Central Tendency & Dispersion:** Mean ($\mu$), Standard Deviation ($\sigma$), Variance ($\sigma^2$), Interquartile Range (IQR).
- **Extreme & Range Bounds:** Minimum, Maximum, Peak-to-Peak Range ($\max - \min$).
- **Signal Energy & Power:** Root Mean Square (RMS) $= \sqrt{\frac{1}{N}\sum x_i^2}$, Energy $= \frac{1}{N}\sum x_i^2$.
- **Temporal & Frequency Dynamics:** Zero-Crossing Rate of centered signal.
- **Cross-Axis Kinematic Coupling:** Pearson correlation coefficients $\rho(x, y), \rho(x, z), \rho(y, z)$ for acceleration and $\rho(\alpha, \beta), \rho(\alpha, \gamma), \rho(\beta, \gamma)$ for rotation.

---

## 4. Machine Learning Benchmarking & Results

Four supervised learning algorithms were trained and evaluated on an 80/20 stratified split across 3,762 windowed instances:

| Classifier Model | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Classifier (Selected)** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| Support Vector Machine (SVM / RBF) | 100.00% | 100.00% | 100.00% | 100.00% |
| K-Nearest Neighbors ($k=5$) | 100.00% | 100.00% | 100.00% | 100.00% |
| Decision Tree ($depth=12$) | 99.73% | 99.73% | 99.73% | 99.73% |

### Selected Model: Random Forest
- **Ensemble:** 100 Decision Trees with Gini impurity split criterion.
- **Top Predictive Features:**
  1. `gyro_mag_rms` & `gyro_mag_mean`: Global rotational power distinguishing dynamic lifting from static postures.
  2. `accel_y_range` & `accel_y_std`: Vertical kinetic displacement distinguishing Squats/Jacks from Arm Curls.
  3. `gyro_beta_mean`: Forearm pitch orientation distinguishing Curls from Lateral Raises.
  4. `accel_x_range`: Lateral acceleration isolating Lateral Raises and Jumping Jacks.

---

## 5. Signal Smoothing, Rep Counting & Set Tracking

1. **Prediction Smoothing Buffer:** A 3-window majority-vote filter prevents classification flickering.
2. **Ambiguity & Rest Fallback:**
   - If total motion variance $\sigma^2 < 0.12\text{ m}^2/\text{s}^4$, state immediately switches to `REST`.
   - If user is moving but model softmax confidence $< 42\%$, system falls back to `WORKOUT` (generic active state) rather than misclassifying.
3. **Repetition Counting (Peak Detection):**
   - Applies low-pass smoothing filter on primary kinematic axis.
   - Dual-threshold state machine detects peak contraction followed by eccentric extension with a $0.9\text{s}$ refractory lockout to eliminate double-counting.
4. **Set Tracking:** Pauses $> 3.0\text{s}$ or transitions to `REST` automatically close and increment exercise sets.

---

## 6. Avatar Animation & Visual Design System

- **Visual Theme:** 80% Deep Midnight Blue (`#050811`, `#080d1b`), 20% Vibrant Neon Pink/Magenta (`#ff2e93`, `#8b5cf6`), Cyan (`#00f2fe`).
- **Avatar System:** Consistent Female (pink outfit, mint sports jacket, dark hair) and Male (black tank & shorts) 3D-styled characters rendered via an HTML5 Canvas skeletal rig running at 60 FPS across 12 distinct kinematic movement loops.
- **Live Waveform Oscilloscopes:** Real-time dual canvas oscilloscopes visualizing raw $X, Y, Z, |A|$ and $\alpha, \beta, \gamma, |G|$ streams with CRT scanline styling and explicit *"MOTION SENSORS — NOT ECG"* warning banners.

---

## 7. Competition Demonstration Protocol

1. Open GymSense AI on smartphone or browser.
2. Select **Female** or **Male** avatar on the mandatory launch screen.
3. Click **Enable Phone Sensors** -> Verify connection status.
4. Click **Calibrate** -> Hold device still for 3-second zero-offset baseline computation.
5. Click **▶ Start Workout** -> Perform Bicep Curls (or trigger via Live Test Rig on desktop).
6. Observe live activity recognition (`BICEP CURL`), real confidence score, live rep counter increments, and smooth avatar curling animation.
7. Switch to Squats -> System auto-finalizes Bicep Curl segment and tracks Squat sets.
8. View Hospital-style waveform monitors displaying live multi-axis traces.
9. Click **■ End Workout** -> Review full workout summary with reps, sets, and active time.
10. Navigate to **ML Model** tab to present the real benchmark comparison table, confusion matrix, and feature importances to the evaluation committee.
