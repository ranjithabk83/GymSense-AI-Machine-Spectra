"""
GYMSENSE AI - Machine Learning Model Training & Evaluation Pipeline
Student Name: Ranjitha B K | USN: 1SB24AI041
Branch: Artificial Intelligence and Machine Learning
Event: MACHINE SPECTRA 1.0 | Date: September 10, 2026

Trains and compares 4 real Machine Learning classifiers:
1. Random Forest Classifier
2. K-Nearest Neighbors (KNN)
3. Decision Tree Classifier
4. Support Vector Classifier (SVM / RBF Kernel)

Extracts 46 kinematic/statistical features, performs stratified train/test split,
computes precision, recall, F1, confusion matrix, and feature importances,
and serializes artifacts to the models/ directory.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from feature_extraction import extract_features_from_window, get_feature_names
from synthetic_data_generator import generate_full_dataset, EXERCISE_CLASSES


WINDOW_SIZE = 50       # 50 samples = 2.5 seconds at 20Hz
STEP_SIZE = 25         # 50% overlap (1.25s step)
DATA_PATH = "data/sensor_data.csv"
MODELS_DIR = "models"


def prepare_dataset(data_path: str = DATA_PATH):
    """Load continuous time-series sensor data, segment into windows, and extract features."""
    if not os.path.exists(data_path):
        print(f"Dataset '{data_path}' not found. Generating fresh dataset...")
        df_raw = generate_full_dataset(data_path)
    else:
        print(f"Loading sensor dataset from '{data_path}'...")
        df_raw = pd.read_csv(data_path)

    print(f"Total raw sensor records: {len(df_raw)}")
    
    feature_rows = []
    labels = []
    
    # Process session by session to prevent cross-session window leakage
    grouped = df_raw.groupby(["session_id", "activity"])
    
    for (sess_id, act), session_df in grouped:
        ax = session_df["accel_x"].values
        ay = session_df["accel_y"].values
        az = session_df["accel_z"].values
        ga = session_df["gyro_alpha"].values
        gb = session_df["gyro_beta"].values
        gg = session_df["gyro_gamma"].values
        
        n_samples = len(ax)
        for start_idx in range(0, n_samples - WINDOW_SIZE + 1, STEP_SIZE):
            end_idx = start_idx + WINDOW_SIZE
            feats = extract_features_from_window(
                accel_x=ax[start_idx:end_idx],
                accel_y=ay[start_idx:end_idx],
                accel_z=az[start_idx:end_idx],
                gyro_alpha=ga[start_idx:end_idx],
                gyro_beta=gb[start_idx:end_idx],
                gyro_gamma=gg[start_idx:end_idx]
            )
            feature_rows.append(feats)
            labels.append(act)

    X_df = pd.DataFrame(feature_rows)
    y_series = pd.Series(labels)

    print(f"[OK] Extracted {len(X_df)} sliding windows with {X_df.shape[1]} features each.")
    print(f"Class distribution:\n{y_series.value_counts().to_string()}")
    return X_df, y_series


def train_and_evaluate():
    """Execute complete training, model benchmarking, and artifact serialization."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    X_df, y_series = prepare_dataset()
    feature_names = list(X_df.columns)

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_series)
    class_names = list(label_encoder.classes_)

    # 80/20 Stratified Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_df.values,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print(f"\nTraining set: {X_train.shape[0]} windows | Test set: {X_test.shape[0]} windows")

    # Define Candidate Classifiers
    candidate_models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        ),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            random_state=42
        ),
        "Support Vector Machine (SVM)": SVC(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
    }

    comparison_results = {}
    best_model_name = None
    best_f1 = -1.0
    trained_models = {}

    print("\n" + "="*70)
    print("            MACHINE LEARNING MODEL BENCHMARKING")
    print("="*70)

    for name, model in candidate_models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        rec = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))

        trained_models[name] = model
        comparison_results[name] = {
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "raw_accuracy": acc,
            "raw_f1": f1
        }

        print(f"-> Accuracy: {acc*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1*100:.2f}%")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    print("\n" + "="*70)
    print(f"[BEST MODEL SELECTED] {best_model_name} (F1 Score: {best_f1*100:.2f}%)")
    print("="*70)

    # Best Model Evaluation details
    best_model = trained_models[best_model_name]
    y_test_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_test_pred)

    # Feature Importances (from Random Forest)
    rf_model = trained_models["Random Forest"]
    importances = rf_model.feature_importances_
    feat_importance_pairs = sorted(
        [{"feature": feat, "importance": round(float(imp) * 100, 3)} 
         for feat, imp in zip(feature_names, importances)],
        key=lambda x: x["importance"],
        reverse=True
    )

    # Save artifacts
    model_file = os.path.join(MODELS_DIR, "exercise_classifier.pkl")
    scaler_file = os.path.join(MODELS_DIR, "scaler.pkl")
    encoder_file = os.path.join(MODELS_DIR, "label_encoder.pkl")
    metadata_file = os.path.join(MODELS_DIR, "model_metadata.json")

    joblib.dump(best_model, model_file)
    joblib.dump(scaler, scaler_file)
    joblib.dump(label_encoder, encoder_file)

    metadata = {
        "project": "GymSense AI - AI-Powered Workout Activity Recognition",
        "student": {
            "name": "Ranjitha B K",
            "usn": "1SB24AI041",
            "branch": "Artificial Intelligence and Machine Learning",
            "event": "MACHINE SPECTRA 1.0",
            "date": "September 10, 2026"
        },
        "selected_model": best_model_name,
        "algorithm": str(best_model.__class__.__name__),
        "trained_timestamp": datetime.now().isoformat(),
        "sampling_rate_hz": 20,
        "window_size_samples": WINDOW_SIZE,
        "window_duration_seconds": WINDOW_SIZE / 20.0,
        "total_training_windows": int(X_train.shape[0]),
        "total_testing_windows": int(X_test.shape[0]),
        "total_features": len(feature_names),
        "classes": class_names,
        "feature_names": feature_names,
        "evaluation_metrics": comparison_results[best_model_name],
        "model_comparison": comparison_results,
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "labels": class_names
        },
        "top_feature_importances": feat_importance_pairs[:15],
        "all_feature_importances": feat_importance_pairs
    }

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[OK] Model successfully saved to: {model_file}")
    print(f"[OK] Scaler saved to: {scaler_file}")
    print(f"[OK] Label Encoder saved to: {encoder_file}")
    print(f"[OK] Model Metadata saved to: {metadata_file}")
    print("\nClassification Report for Best Model:")
    print(classification_report(y_test, y_test_pred, target_names=class_names, digits=4))

    return metadata


if __name__ == "__main__":
    train_and_evaluate()
