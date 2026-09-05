# GymSense AI — AI-Powered Workout Activity Recognition

**Student Name:** Ranjitha B K  
**USN:** 1SB24AI041  
**Department:** Artificial Intelligence and Machine Learning  
**Event:** MACHINE SPECTRA 1.0  
**Date:** September 10, 2026  

---

## 🌟 Overview

**GymSense AI** is a competition-grade mobile Progressive Web Application (PWA) with a Python FastAPI backend that performs real-time classification of physical workout exercises from smartphone 3-axis Accelerometer and 3-axis Gyroscope sensor streams using supervised Machine Learning.

---

## 🚀 Key Features

- **Real Machine Learning Engine**: 40+ kinematic/statistical features extracted per 2.5s window; trained and benchmarked across **Random Forest**, **KNN**, **Decision Tree**, and **SVM** classifiers.
- **60 FPS Consistent Skeletal Avatars**: High-fidelity Female (pink outfit + mint sports jacket) and Male (black tank + shorts) avatars animated across 12 discrete exercise states (`REST`, `WALKING`, `RUNNING`, `BICEP CURL`, `HAMMER CURL`, `SQUAT`, `LUNGE`, `JUMPING JACK`, `SHOULDER PRESS`, `FRONT RAISE`, `LATERAL RAISE`, `WORKOUT`).
- **Live Phone Sensors & 3s Calibration**: Web `DeviceMotionEvent` and `DeviceOrientationEvent` with zero-bias baseline calibration.
- **Peak-Detection Rep Counter & Auto Set Tracking**: Smooth dual-threshold peak detection with refractory filter and automatic set accumulation.
- **Hospital Vital-Style Waveform Monitors**: High-tech real-time multi-trace oscilloscopes for Accelerometer ($X, Y, Z, |A|$) and Gyroscope ($\alpha, \beta, \gamma, |G|$).
- **Interactive Competition Test Rig**: Integrated motion simulator for judge demonstrations on laptops/desktops without physical sensors.
- **PWA & Offline Capable**: Standalone installable app shell (`manifest.json`, `service-worker.js`).
- **Full History & Analytics**: LocalStorage persistence, expandable activity segment logs, and exercise volume distributions.
- **Developer Studio**: Real-time sensor data recorder with CSV export.

---

## 🛠️ Project Structure

```
GymSense-AI/
│
├── main.py                     # FastAPI backend server & inference endpoints
├── train_model.py              # ML model training, benchmarking & serialization
├── feature_extraction.py       # 46-feature kinematic extraction pipeline
├── synthetic_data_generator.py # Biomechanical time-series dataset synthesizer
├── generate_icons.py           # PWA icon generator
├── requirements.txt            # Python dependencies
├── README.md                   # Project manual & setup instructions
├── .gitignore                  # Git ignore rules
│
├── data/
│   └── sensor_data.csv         # Raw multi-session sensor dataset
│
├── models/
│   ├── exercise_classifier.pkl # Trained Random Forest model
│   ├── scaler.pkl              # Fitted StandardScaler
│   ├── label_encoder.pkl       # Target LabelEncoder
│   └── model_metadata.json     # Benchmark scores, confusion matrix, feature importances
│
├── static/
│   ├── css/
│   │   └── style.css           # Premium fitness dark UI stylesheet
│   ├── js/
│   │   ├── app.js              # Master client controller
│   │   ├── sensors.js          # DeviceMotion & Gyroscope manager + Simulator
│   │   ├── workout.js          # Session state machine, rep counter & set tracker
│   │   ├── charts.js           # Waveform oscilloscopes & ML inspection charts
│   │   ├── avatarEngine.js     # 60 FPS skeletal avatar animation engine
│   │   ├── history.js          # LocalStorage history & analytics manager
│   │   └── pwa.js              # Service worker registration & install prompt
│   ├── icons/                  # 192x192, 512x512, apple-touch-icon
│   ├── manifest.json           # PWA web manifest
│   └── service-worker.js       # PWA offline cache worker
│
├── templates/
│   └── index.html              # Single-page mobile PWA markup
│
└── documentation/
    └── project_explanation.md  # Detailed technical documentation for judges
```

---

## 💻 Windows PowerShell Execution Commands

### 1. Create Virtual Environment
```powershell
python -m venv venv
```

### 2. Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install Requirements
```powershell
pip install -r requirements.txt
```

### 4. Train and Benchmark the ML Models
```powershell
python train_model.py
```

### 5. Run the Application Locally
```powershell
python main.py
```
*Open your browser and navigate to:* `http://localhost:8000` (or your local network IP on phone via HTTPS/tunnel).

---

## 🌐 Deploy to GitHub & Render

### 6. Push to GitHub
```powershell
git init
git add .
git commit -m "Initial commit: GymSense AI competition release"
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/gymsense-ai.git
git push -u origin main
```

### 7. Deploy to Render
1. Go to [Render Dashboard](https://dashboard.render.com/) and click **New +** -> **Web Service**.
2. Connect your GitHub repository `gymsense-ai`.
3. Configure the service:
   - **Name:** `gymsense-ai`
   - **Environment:** `Python`
   - **Build Command:** `pip install -r requirements.txt && python train_model.py`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Click **Create Web Service**.
5. Once deployed, open the Render URL (`https://gymsense-ai.onrender.com`) on your smartphone to install as a standalone PWA with full secure HTTPS sensor support!
