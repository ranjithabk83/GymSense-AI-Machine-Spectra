"""
GYMSENSE AI - FastAPI Backend Server
Student: Ranjitha B K | USN: 1SB24AI041
Branch: AIML | Event: MACHINE SPECTRA 1.0 | Date: Sept 10, 2026

Real-time Exercise Recognition API powered by scikit-learn Machine Learning.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from feature_extraction import (
    extract_features_from_records,
    get_feature_names,
)


# ============================================================
# APP CONFIGURATION
# ============================================================

app = FastAPI(
    title="GymSense AI",
    description="AI-Powered Workout Activity Recognition Backend",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
DATA_DIR = os.path.join(BASE_DIR, "data")


if os.path.exists(STATIC_DIR):
    app.mount(
        "/static",
        StaticFiles(directory=STATIC_DIR),
        name="static",
    )


# ============================================================
# THRESHOLDS
# ============================================================

# This is NOT confidence.
# It is only used to determine whether the phone is nearly still.
REST_MOTION_VARIANCE_THRESHOLD = 0.12

# If the model cannot confidently identify an exercise,
# display WORKOUT instead of guessing.
UNKNOWN_CONFIDENCE_THRESHOLD = 0.42


# ============================================================
# ML STATE
# ============================================================

ml_state = {
    "model": None,
    "scaler": None,
    "label_encoder": None,
    "metadata": None,
    "feature_names": [],
    "classes": [],
    "is_loaded": False,
}


# ============================================================
# LOAD MODEL
# ============================================================

def load_ml_models():
    """
    Load trained model, scaler, label encoder and metadata.
    """

    try:

        model_path = os.path.join(
            MODELS_DIR,
            "exercise_classifier.pkl",
        )

        scaler_path = os.path.join(
            MODELS_DIR,
            "scaler.pkl",
        )

        encoder_path = os.path.join(
            MODELS_DIR,
            "label_encoder.pkl",
        )

        meta_path = os.path.join(
            MODELS_DIR,
            "model_metadata.json",
        )

        required_files = [
            model_path,
            scaler_path,
            encoder_path,
        ]

        if not all(
            os.path.exists(path)
            for path in required_files
        ):

            ml_state["is_loaded"] = False

            print(
                "[WARN] ML model files not found. "
                "Train the model first."
            )

            return


        ml_state["model"] = joblib.load(
            model_path
        )

        ml_state["scaler"] = joblib.load(
            scaler_path
        )

        ml_state["label_encoder"] = joblib.load(
            encoder_path
        )


        if os.path.exists(meta_path):

            with open(
                meta_path,
                "r",
                encoding="utf-8",
            ) as file:

                ml_state["metadata"] = json.load(
                    file
                )


            ml_state["feature_names"] = (
                ml_state["metadata"].get(
                    "feature_names",
                    get_feature_names(),
                )
            )


            ml_state["classes"] = (
                ml_state["metadata"].get(
                    "classes",
                    list(
                        ml_state[
                            "label_encoder"
                        ].classes_
                    ),
                )
            )


        else:

            ml_state["metadata"] = None

            ml_state["feature_names"] = (
                get_feature_names()
            )

            ml_state["classes"] = list(
                ml_state[
                    "label_encoder"
                ].classes_
            )


        ml_state["is_loaded"] = True


        print(
            f"[OK] Loaded "
            f"{ml_state['model'].__class__.__name__} "
            f"with "
            f"{len(ml_state['classes'])} classes."
        )


    except Exception as error:

        ml_state["is_loaded"] = False

        print(
            f"[ERROR] Failed to load ML model: "
            f"{error}"
        )


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    load_ml_models()


# ============================================================
# REQUEST MODELS
# ============================================================

class SensorReading(BaseModel):

    accel_x: float = Field(
        ...,
        description="Acceleration X",
    )

    accel_y: float = Field(
        ...,
        description="Acceleration Y",
    )

    accel_z: float = Field(
        ...,
        description="Acceleration Z",
    )

    gyro_alpha: float = Field(
        0.0,
        description="Gyroscope Alpha",
    )

    gyro_beta: float = Field(
        0.0,
        description="Gyroscope Beta",
    )

    gyro_gamma: float = Field(
        0.0,
        description="Gyroscope Gamma",
    )

    timestamp_ms: Optional[int] = None


class PredictRequest(BaseModel):

    sensor_window: Optional[
        List[Dict[str, Any]]
    ] = None

    features: Optional[
        Dict[str, float]
    ] = None


class TrainingDataPayload(BaseModel):

    activity: str

    samples: List[
        Dict[str, Any]
    ]

    notes: Optional[str] = None


# ============================================================
# MOTION VARIANCE
# ============================================================

def calculate_motion_variance(
    records: List[Dict[str, Any]]
) -> float:

    """
    Measures how much the phone is moving.

    Very low variance means the phone is
    probably still/resting.
    """

    ax = [
        float(
            row.get(
                "accel_x",
                row.get(
                    "x",
                    0.0,
                ),
            )
        )
        for row in records
    ]

    ay = [
        float(
            row.get(
                "accel_y",
                row.get(
                    "y",
                    0.0,
                ),
            )
        )
        for row in records
    ]

    az = [
        float(
            row.get(
                "accel_z",
                row.get(
                    "z",
                    9.81,
                ),
            )
        )
        for row in records
    ]


    if not ax or not ay or not az:

        return 0.0


    motion_variance = (
        np.var(ax)
        +
        np.var(ay)
        +
        np.var(az)
    )


    return float(
        motion_variance
    )


# ============================================================
# MODEL PROBABILITY
# ============================================================

def get_probability_output(
    model,
    encoder,
    scaled_vector,
):

    """
    Run actual ML prediction.

    No fake probability is created.

    Returns:

    predicted_label
    confidence
    probability_distribution
    """

    # --------------------------------------------------------
    # MODEL WITHOUT predict_proba
    # --------------------------------------------------------

    if not hasattr(
        model,
        "predict_proba",
    ):

        encoded_prediction = (
            model.predict(
                scaled_vector
            )[0]
        )


        predicted_label = str(

            encoder.inverse_transform(
                [
                    int(
                        encoded_prediction
                    )
                ]
            )[0]

        )


        return (
            predicted_label,
            None,
            {},
        )


    # --------------------------------------------------------
    # REAL MODEL PROBABILITIES
    # --------------------------------------------------------

    probabilities = (
        model.predict_proba(
            scaled_vector
        )[0]
    )


    max_position = int(
        np.argmax(
            probabilities
        )
    )


    model_classes = getattr(
        model,
        "classes_",
        np.arange(
            len(
                probabilities
            )
        ),
    )


    encoded_prediction = int(
        model_classes[
            max_position
        ]
    )


    predicted_label = str(

        encoder.inverse_transform(
            [
                encoded_prediction
            ]
        )[0]

    )


    confidence = float(
        probabilities[
            max_position
        ]
    )


    decoded_labels = []


    for encoded_class in model_classes:

        label = str(

            encoder.inverse_transform(
                [
                    int(
                        encoded_class
                    )
                ]
            )[0]

        )

        decoded_labels.append(
            label
        )


    probability_distribution = {

        label: round(
            float(
                probability
            ),
            4,
        )

        for label, probability

        in zip(
            decoded_labels,
            probabilities,
        )
    }


    return (
        predicted_label,
        confidence,
        probability_distribution,
    )


# ============================================================
# STABILITY LABEL
# ============================================================

def confidence_to_stability(
    confidence: Optional[float]
) -> str:

    if confidence is None:

        return "Unavailable"


    if confidence >= 0.85:

        return "Excellent"


    if confidence >= 0.65:

        return "Good"


    if confidence >= (
        UNKNOWN_CONFIDENCE_THRESHOLD
    ):

        return "Fair"


    return "Low"


# ============================================================
# HOME
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
async def serve_index():

    index_path = os.path.join(
        TEMPLATES_DIR,
        "index.html",
    )


    if os.path.exists(
        index_path
    ):

        with open(
            index_path,
            "r",
            encoding="utf-8",
        ) as file:

            return HTMLResponse(
                content=file.read(),
                status_code=200,
            )


    return HTMLResponse(
        content=(
            "<h1>"
            "GymSense AI is starting..."
            "</h1>"
        ),
        status_code=200,
    )


# ============================================================
# PWA MANIFEST
# ============================================================

@app.get(
    "/manifest.json"
)
async def serve_manifest():

    manifest_path = os.path.join(
        STATIC_DIR,
        "manifest.json",
    )


    if os.path.exists(
        manifest_path
    ):

        return FileResponse(
            manifest_path,
            media_type=(
                "application/"
                "manifest+json"
            ),
        )


    raise HTTPException(
        status_code=404,
        detail=(
            "manifest.json "
            "not found"
        ),
    )


# ============================================================
# SERVICE WORKER
# ============================================================

@app.get(
    "/service-worker.js"
)
async def serve_service_worker():

    service_worker_path = (
        os.path.join(
            STATIC_DIR,
            "service-worker.js",
        )
    )


    if os.path.exists(
        service_worker_path
    ):

        return FileResponse(

            service_worker_path,

            media_type=(
                "application/"
                "javascript"
            ),

            headers={
                "Service-Worker-Allowed": "/"
            },

        )


    raise HTTPException(
        status_code=404,
        detail=(
            "service-worker.js "
            "not found"
        ),
    )


# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/health"
)
async def health_check():

    if not ml_state[
        "is_loaded"
    ]:

        load_ml_models()


    return {

        "status": "healthy",

        "app_name": (
            "GymSense AI"
        ),

        "author": (
            "Ranjitha B K "
            "(1SB24AI041)"
        ),

        "event": (
            "MACHINE SPECTRA 1.0"
        ),

        "model_loaded": (
            ml_state[
                "is_loaded"
            ]
        ),

        "timestamp": (
            datetime
            .now()
            .isoformat()
        ),

    }


# ============================================================
# MODEL INFO
# ============================================================

@app.get(
    "/model-info"
)
async def get_model_info():

    if not ml_state[
        "is_loaded"
    ]:

        load_ml_models()


    if ml_state[
        "metadata"
    ]:

        return ml_state[
            "metadata"
        ]


    return {

        "status": (
            "training_or_loading"
        ),

        "message": (
            "Model metadata "
            "is not available."
        ),

    }


# ============================================================
# PREDICTION
# ============================================================

@app.post(
    "/predict"
)
async def predict_activity(
    payload: PredictRequest
):

    """
    MAIN ML PREDICTION ROUTE.

    Logic:

    Very little movement
    -> REST

    Known exercise with good
    model confidence
    -> Exercise name

    Movement but low confidence
    -> WORKOUT

    REST confidence is taken
    from model.predict_proba().

    NO fake 0.985 value.
    """

    # --------------------------------------------------------
    # MODEL CHECK
    # --------------------------------------------------------

    if not ml_state[
        "is_loaded"
    ]:

        load_ml_models()


        if not ml_state[
            "is_loaded"
        ]:

            raise HTTPException(

                status_code=503,

                detail=(
                    "ML model is "
                    "not loaded. "
                    "Train the model first."
                ),

            )


    try:

        # ====================================================
        # RAW SENSOR WINDOW
        # ====================================================

        if (
            payload.sensor_window
            is not None
        ):

            if len(
                payload.sensor_window
            ) < 10:

                raise HTTPException(

                    status_code=400,

                    detail=(
                        "sensor_window "
                        "must contain at "
                        "least 10 readings"
                    ),

                )


            window_records = (
                payload.sensor_window
            )


            features_dict = (
                extract_features_from_records(
                    window_records
                )
            )


            total_motion_var = (
                calculate_motion_variance(
                    window_records
                )
            )


        # ====================================================
        # PRE-COMPUTED FEATURES
        # ====================================================

        elif (
            payload.features
            is not None
        ):

            features_dict = (
                payload.features
            )


            total_motion_var = float(

                features_dict.get(

                    "accel_mag_var",

                    features_dict.get(
                        "accel_variance",
                        1.0,
                    ),

                )

            )


        else:

            raise HTTPException(

                status_code=400,

                detail=(
                    "Provide either "
                    "'sensor_window' "
                    "or 'features'."
                ),

            )


        # ====================================================
        # FEATURE ORDER
        # ====================================================

        feature_names = (

            ml_state[
                "feature_names"
            ]

            or

            get_feature_names()

        )


        feature_vector = np.array(

            [

                [

                    float(
                        features_dict.get(
                            feature_name,
                            0.0,
                        )
                    )

                    for feature_name
                    in feature_names

                ]

            ],

            dtype=float,

        )


        # ====================================================
        # SCALE FEATURES
        # ====================================================

        scaled_vector = (

            ml_state[
                "scaler"
            ].transform(
                feature_vector
            )

        )


        # ====================================================
        # ACTUAL ML MODEL
        # ====================================================

        model = ml_state[
            "model"
        ]


        encoder = ml_state[
            "label_encoder"
        ]


        (
            predicted_class,
            raw_confidence,
            probability_distribution,
        ) = get_probability_output(

            model,

            encoder,

            scaled_vector,

        )


        predicted_class = (

            predicted_class
            .upper()
            .replace(
                "_",
                " ",
            )

        )


        # ====================================================
        # NORMALIZE CLASS NAMES
        # ====================================================

        normalized_probabilities = {

            str(label)
            .upper()
            .replace(
                "_",
                " "
            ):
            probability

            for label, probability

            in probability_distribution.items()

        }


        # ====================================================
        # ACTUAL REST PROBABILITY
        # ====================================================

        rest_probability = (

            normalized_probabilities
            .get(
                "REST"
            )

        )


        # ====================================================
        # MOTION DETECTION
        # ====================================================

        is_stationary = (

            total_motion_var

            <

            REST_MOTION_VARIANCE_THRESHOLD

        )


        # ====================================================
        # FINAL LOGIC
        # ====================================================

        # ----------------------------------------------------
        # 1. REST
        # ----------------------------------------------------

        if is_stationary:

            final_activity = (
                "REST"
            )

            final_display = (
                "REST"
            )


            # IMPORTANT:
            # This comes from the real
            # ML probability.
            final_confidence = (
                rest_probability
            )


            confidence_source = (
                "model_rest_probability"
            )


            if (
                final_confidence
                is None
            ):

                stability = (
                    "Stable"
                )

            else:

                stability = (
                    confidence_to_stability(
                        final_confidence
                    )
                )


        # ----------------------------------------------------
        # 2. UNKNOWN MOVEMENT
        # ----------------------------------------------------

        elif (

            raw_confidence
            is not None

            and

            raw_confidence
            <
            UNKNOWN_CONFIDENCE_THRESHOLD

        ):

            final_activity = (
                "WORKOUT"
            )

            final_display = (
                "WORKOUT"
            )


            # This is NOT pretending
            # WORKOUT had this probability.
            #
            # It represents the highest
            # known-class model confidence.
            final_confidence = (
                raw_confidence
            )


            confidence_source = (
                "highest_known_class_probability"
            )


            stability = (
                "Low"
            )


        # ----------------------------------------------------
        # 3. KNOWN EXERCISE
        # ----------------------------------------------------

        else:

            final_activity = (
                predicted_class
            )

            final_display = (
                predicted_class
            )


            final_confidence = (
                raw_confidence
            )


            if (
                raw_confidence
                is not None
            ):

                confidence_source = (
                    "model_predict_proba"
                )

            else:

                confidence_source = (
                    "unavailable"
                )


            stability = (
                confidence_to_stability(
                    final_confidence
                )
            )


        # ====================================================
        # RESPONSE
        # ====================================================

        return {

            "activity": (
                final_activity
            ),

            "display_name": (
                final_display
            ),

            "confidence": (

                round(
                    float(
                        final_confidence
                    ),
                    4,
                )

                if (
                    final_confidence
                    is not None
                )

                else None

            ),

            "confidence_available": (

                final_confidence
                is not None

            ),

            "confidence_source": (
                confidence_source
            ),

            "is_resting": (

                final_activity
                ==
                "REST"

            ),

            "probabilities": (
                normalized_probabilities
            ),

            "raw_model_prediction": (
                predicted_class
            ),

            "raw_model_confidence": (

                round(
                    float(
                        raw_confidence
                    ),
                    4,
                )

                if (
                    raw_confidence
                    is not None
                )

                else None

            ),

            "stability": (
                stability
            ),

            "motion_energy": (

                round(
                    float(
                        total_motion_var
                    ),
                    6,
                )

            ),

            "rest_motion_threshold": (
                REST_MOTION_VARIANCE_THRESHOLD
            ),

            "unknown_confidence_threshold": (
                UNKNOWN_CONFIDENCE_THRESHOLD
            ),

        }


    except HTTPException:

        raise


    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=(
                f"Inference error: "
                f"{str(error)}"
            ),

        )


# ============================================================
# TRAINING DATA
# ============================================================

@app.post(
    "/training-data"
)
async def save_training_data(
    payload: TrainingDataPayload
):

    """
    Save real phone sensor readings
    for future model training.
    """

    os.makedirs(
        DATA_DIR,
        exist_ok=True,
    )


    csv_file = os.path.join(
        DATA_DIR,
        "user_recorded_data.csv",
    )


    try:

        rows = []


        recorded_at = (
            datetime
            .now()
            .isoformat()
        )


        for sample in (
            payload.samples
        ):

            rows.append(

                {

                    "recorded_at": (
                        recorded_at
                    ),

                    "activity": (
                        payload.activity
                    ),

                    "accel_x": (
                        sample.get(
                            "accel_x",
                            sample.get(
                                "x",
                                0.0,
                            ),
                        )
                    ),

                    "accel_y": (
                        sample.get(
                            "accel_y",
                            sample.get(
                                "y",
                                0.0,
                            ),
                        )
                    ),

                    "accel_z": (
                        sample.get(
                            "accel_z",
                            sample.get(
                                "z",
                                0.0,
                            ),
                        )
                    ),

                    "gyro_alpha": (
                        sample.get(
                            "gyro_alpha",
                            sample.get(
                                "alpha",
                                0.0,
                            ),
                        )
                    ),

                    "gyro_beta": (
                        sample.get(
                            "gyro_beta",
                            sample.get(
                                "beta",
                                0.0,
                            ),
                        )
                    ),

                    "gyro_gamma": (
                        sample.get(
                            "gyro_gamma",
                            sample.get(
                                "gamma",
                                0.0,
                            ),
                        )
                    ),

                    "notes": (
                        payload.notes
                        or
                        ""
                    ),

                }

            )


        if not rows:

            raise HTTPException(

                status_code=400,

                detail=(
                    "No training "
                    "samples provided."
                ),

            )


        new_data = (
            pd.DataFrame(
                rows
            )
        )


        if os.path.exists(
            csv_file
        ):

            new_data.to_csv(

                csv_file,

                mode="a",

                header=False,

                index=False,

            )


        else:

            new_data.to_csv(

                csv_file,

                index=False,

            )


        return {

            "status": (
                "success"
            ),

            "recorded_samples": (
                len(
                    rows
                )
            ),

            "file": (
                "data/"
                "user_recorded_data.csv"
            ),

        }


    except HTTPException:

        raise


    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=(

                "Failed to save "
                "training data: "
                f"{str(error)}"

            ),

        )


# ============================================================
# LOCAL SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn


    uvicorn.run(

        "main:app",

        host="0.0.0.0",

        port=8000,

        reload=True,

    )