
import os
import json
import joblib
import numpy as np
import pandas as pd
import cv2

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

MODELS_DIR = os.path.join(PROJECT_DIR, "models")
SEGMENTATION_DIR = os.path.join(PROJECT_DIR, "segmentation")
IMAGE_DIR = os.path.join(SEGMENTATION_DIR, "images")

CLINICAL_MODEL_PATH = os.path.join(
    MODELS_DIR, "best_ckd_model.pkl"
)

SCALER_PATH = os.path.join(
    MODELS_DIR, "scaler.pkl"
)

STAGE_MODEL_PATH = os.path.join(
    MODELS_DIR, "stage_model.pkl"
)

IMAGE_RF_PATH = os.path.join(
    MODELS_DIR, "image_random_forest.pkl"
)

UNET_PATH = os.path.join(
    SEGMENTATION_DIR, "models", "best_unet.pth"
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="NephroPredictor API",
    description="CKD prediction using clinical and ultrasound data",
    version="1.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# OPTIONAL PYTORCH / U-NET
# ============================================================

TORCH_AVAILABLE = False
TORCH_ERROR = None
torch = None
UNet = None

try:
    import torch

    TORCH_AVAILABLE = True

    try:
        from unet_model import UNet
    except Exception:
        # Try importing from segmentation folder
        import sys

        if SEGMENTATION_DIR not in sys.path:
            sys.path.append(SEGMENTATION_DIR)

        from unet_model import UNet

except Exception as e:
    TORCH_AVAILABLE = False
    TORCH_ERROR = str(e)


# ============================================================
# LOAD MODELS
# ============================================================

clinical_model = None
scaler = None
stage_model = None
image_model = None
unet_model = None


# ---------- Clinical model ----------

try:
    clinical_model = joblib.load(CLINICAL_MODEL_PATH)
    print("Clinical model loaded successfully.")
except Exception as e:
    print("Clinical model loading failed:", e)


# ---------- Scaler ----------

try:
    scaler = joblib.load(SCALER_PATH)
    print("Scaler loaded successfully.")
except Exception as e:
    print("Scaler loading failed:", e)


# ---------- Stage model ----------

try:
    stage_model = joblib.load(STAGE_MODEL_PATH)
    print("Stage model loaded successfully.")
except Exception as e:
    print("Stage model loading failed:", e)


# ---------- Image model ----------

try:
    image_model = joblib.load(IMAGE_RF_PATH)
    print("Image model loaded successfully.")
except Exception as e:
    print("Image model loading failed:", e)


# ---------- U-Net ----------

if TORCH_AVAILABLE:

    try:

        unet_model = UNet()

        checkpoint = torch.load(
            UNET_PATH,
            map_location="cpu"
        )

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            unet_model.load_state_dict(
                checkpoint["model_state_dict"]
            )
        else:
            unet_model.load_state_dict(checkpoint)

        unet_model.eval()

        print("U-Net loaded successfully.")

    except Exception as e:

        unet_model = None
        TORCH_ERROR = str(e)

        print(
            "U-Net loading failed:",
            TORCH_ERROR
        )

else:

    print(
        "PyTorch unavailable. "
        "U-Net endpoints will remain unavailable."
    )


# ============================================================
# CLINICAL FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "serum_creatinine",
    "gfr",
    "bun",
    "serum_calcium",
    "oxalate_levels",
    "urine_ph",
    "blood_pressure",
    "ana",
    "c3_c4",
    "hematuria",
    "smoking",
    "alcohol",
    "painkiller_usage",
    "family_history",
    "physical_activity",
    "diet",
    "water_intake",
    "weight_changes",
    "stress_level",
    "months"
]


# ============================================================
# PYDANTIC MODEL
# ============================================================

class PatientData(BaseModel):

    serum_creatinine: float = Field(ge=0)
    gfr: float = Field(ge=0)

    bun: float = Field(ge=0)

    serum_calcium: float = Field(ge=0)

    oxalate_levels: float = Field(ge=0)

    urine_ph: float = Field(
        ge=0,
        le=14
    )

    blood_pressure: float = Field(ge=0)

    ana: float = Field(ge=0)
    c3_c4: float = Field(ge=0)

    hematuria: float = Field(ge=0)

    smoking: float = Field(ge=0)
    alcohol: float = Field(ge=0)

    painkiller_usage: float = Field(ge=0)

    family_history: float = Field(ge=0)

    physical_activity: float = Field(ge=0)

    diet: float = Field(ge=0)

    water_intake: float = Field(ge=0)

    weight_changes: float = Field(ge=0)

    stress_level: float = Field(ge=0)

    months: float = Field(ge=0)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "NephroPredictor API is running",
        "status": "Healthy"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "Healthy",

        "model_loaded":
            clinical_model is not None,

        "ckd_model":
            "Loaded"
            if clinical_model is not None
            else "Unavailable",

        "scaler":
            "Loaded"
            if scaler is not None
            else "Unavailable",

        "stage_model":
            "Loaded"
            if stage_model is not None
            else "Unavailable",

        "unet_model":
            "Loaded"
            if unet_model is not None
            else "Unavailable",

        "image_model":
            "Loaded"
            if image_model is not None
            else "Unavailable",

        "torch_available":
            TORCH_AVAILABLE
    }


# ============================================================
# CLINICAL PREDICTION
# ============================================================

@app.post("/predict")
def predict(patient: PatientData):

    # --------------------------------------------------------
    # Check models
    # --------------------------------------------------------

    if clinical_model is None:

        raise HTTPException(
            status_code=500,
            detail="Clinical model is not loaded."
        )

    if scaler is None:

        raise HTTPException(
            status_code=500,
            detail="Clinical scaler is not loaded."
        )


    # --------------------------------------------------------
    # Convert patient data to dictionary
    # --------------------------------------------------------

    patient_data = patient.model_dump()


    # --------------------------------------------------------
    # IMPORTANT:
    # Create DataFrame in EXACT training feature order
    # --------------------------------------------------------

    input_df = pd.DataFrame(
        [[patient_data[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS
    )


    # --------------------------------------------------------
    # SCALE RAW CLINICAL VALUES
    # --------------------------------------------------------

    input_scaled = scaler.transform(input_df)


    # --------------------------------------------------------
    # CKD PREDICTION
    # --------------------------------------------------------

    prediction = int(
        clinical_model.predict(input_scaled)[0]
    )


    # --------------------------------------------------------
    # PROBABILITY
    # --------------------------------------------------------

    probabilities = clinical_model.predict_proba(
        input_scaled
    )[0]

    confidence = float(
        np.max(probabilities) * 100
    )


    # --------------------------------------------------------
    # LABEL
    # --------------------------------------------------------

    if prediction == 1:

        prediction_label = "CKD Detected"

    else:

        prediction_label = "Healthy"


    # --------------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------------

    if prediction == 1:

        if confidence >= 80:
            risk_level = "High"

        elif confidence >= 60:
            risk_level = "Moderate"

        else:
            risk_level = "Low"

    else:

        if confidence >= 80:
            risk_level = "Low"

        elif confidence >= 60:
            risk_level = "Moderate"

        else:
            risk_level = "High"


    # --------------------------------------------------------
    # CKD STAGE
    # --------------------------------------------------------

    stage = None
    stage_number = None

    if prediction == 1 and stage_model is not None:

        try:

            stage_prediction = stage_model.predict(
                input_scaled
            )[0]

            stage_number = int(
                round(float(stage_prediction))
            )

            stage = f"Stage {stage_number}"

        except Exception as e:

            print(
                "Stage prediction failed:",
                e
            )


    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {

        "status": "Success",

        "prediction":
            prediction_label,

        "confidence":
            round(confidence, 2),

        "risk_level":
            risk_level,

        "stage":
            stage,

        "stage_number":
            stage_number
    }


# ============================================================
# IMAGE PREDICTION
# ============================================================

@app.post("/predict-image")
async def predict_image(
    file: UploadFile = File(...)
):

    if not TORCH_AVAILABLE or unet_model is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Ultrasound prediction is currently unavailable "
                "because U-Net/PyTorch could not be loaded."
            )
        )

    if image_model is None:

        raise HTTPException(
            status_code=500,
            detail="Image classification model is not loaded."
        )


    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    contents = await file.read()

    image_array = np.frombuffer(
        contents,
        np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )


    if image is None:

        raise HTTPException(
            status_code=400,
            detail="Invalid ultrasound image."
        )


    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    image = cv2.resize(
        image,
        (256, 256)
    )


    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    image_float = (
        image_rgb.astype(np.float32) / 255.0
    )


    # --------------------------------------------------------
    # Tensor
    # --------------------------------------------------------

    tensor = torch.tensor(
        image_float
    ).permute(
        2, 0, 1
    ).unsqueeze(0)


    # --------------------------------------------------------
    # U-Net segmentation
    # --------------------------------------------------------

    with torch.no_grad():

        output = unet_model(
            tensor
        )

        probability = torch.sigmoid(
            output
        )

        mask = (
            probability > 0.5
        ).float()


    # --------------------------------------------------------
    # Mask
    # --------------------------------------------------------

    mask_np = (
        mask.squeeze()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )


    # --------------------------------------------------------
    # Extract image features
    # --------------------------------------------------------

    kidney_pixels = np.sum(mask_np)

    kidney_area = float(kidney_pixels)

    contours, _ = cv2.findContours(
        mask_np,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    kidney_perimeter = 0.0
    kidney_width = 0.0
    kidney_height = 0.0

    if contours:

        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        kidney_perimeter = float(
            cv2.arcLength(
                largest_contour,
                True
            )
        )

        x, y, w, h = cv2.boundingRect(
            largest_contour
        )

        kidney_width = float(w)
        kidney_height = float(h)


    if kidney_perimeter > 0:

        circularity = (
            4 * np.pi * kidney_area
        ) / (
            kidney_perimeter ** 2
        )

    else:

        circularity = 0.0


    if kidney_height > 0:

        aspect_ratio = (
            kidney_width /
            kidney_height
        )

    else:

        aspect_ratio = 0.0


    # --------------------------------------------------------
    # Intensity features
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY
    )


    mean_intensity = float(
        np.mean(gray)
    )

    std_intensity = float(
        np.std(gray)
    )

    min_intensity = float(
        np.min(gray)
    )

    max_intensity = float(
        np.max(gray)
    )

    median_intensity = float(
        np.median(gray)
    )


    # --------------------------------------------------------
    # ROI intensity
    # --------------------------------------------------------

    roi_pixels = gray[
        mask_np == 1
    ]


    if len(roi_pixels) > 0:

        roi_mean = float(
            np.mean(roi_pixels)
        )

        roi_std = float(
            np.std(roi_pixels)
        )

    else:

        roi_mean = 0.0
        roi_std = 0.0


    # --------------------------------------------------------
    # Feature vector
    # --------------------------------------------------------

    image_features = np.array([

        kidney_area,
        kidney_perimeter,
        kidney_width,
        kidney_height,
        aspect_ratio,
        circularity,
        mean_intensity,
        std_intensity,
        min_intensity,
        max_intensity,
        median_intensity,
        roi_mean,
        roi_std

    ]).reshape(1, -1)


    # --------------------------------------------------------
    # Image prediction
    # --------------------------------------------------------

    image_prediction = int(
        image_model.predict(
            image_features
        )[0]
    )


    image_probabilities = (
        image_model.predict_proba(
            image_features
        )[0]
    )


    image_probability = float(
        image_probabilities[1] * 100
    )


    if image_prediction == 1:

        image_label = "Pathological"

    else:

        image_label = "Healthy"


    return {

        "status": "Success",

        "prediction":
            image_label,

        "image_probability":
            round(
                image_probability,
                2
            ),

        "kidney_area":
            round(
                kidney_area,
                2
            ),

        "kidney_perimeter":
            round(
                kidney_perimeter,
                2
            ),

        "segmentation_available":
            True
    }


# ============================================================
# FUSION PREDICTION
# ============================================================

@app.post("/fusion-predict")
async def fusion_predict(
    patient: str = Form(...),
    file: UploadFile = File(...)
):

    if not TORCH_AVAILABLE or unet_model is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Fusion prediction is currently unavailable "
                "because U-Net/PyTorch could not be loaded."
            )
        )


    # --------------------------------------------------------
    # Parse patient JSON
    # --------------------------------------------------------

    try:

        patient_data = json.loads(patient)

        patient_model = PatientData(
            **patient_data
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Invalid patient data: {str(e)}"
        )


    # --------------------------------------------------------
    # Clinical prediction
    # --------------------------------------------------------

    clinical_result = predict(
        patient_model
    )


    # --------------------------------------------------------
    # Image prediction
    # --------------------------------------------------------

    image_result = await predict_image(
        file
    )


    clinical_probability = (
        clinical_result["confidence"] / 100
        if clinical_result["prediction"]
        == "CKD Detected"
        else
        1 -
        clinical_result["confidence"] / 100
    )


    image_probability = (
        image_result["image_probability"] / 100
    )


    # --------------------------------------------------------
    # Prototype late fusion
    # --------------------------------------------------------

    clinical_weight = 0.60
    image_weight = 0.40


    final_probability = (
        clinical_weight *
        clinical_probability

        +

        image_weight *
        image_probability
    )


    if final_probability >= 0.50:

        final_prediction = "CKD Detected"

    else:

        final_prediction = "Healthy"


    # --------------------------------------------------------
    # Stage
    # --------------------------------------------------------

    stage = clinical_result["stage"]
    stage_number = clinical_result["stage_number"]


    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    return {

        "status": "Success",

        "prediction":
            final_prediction,

        "clinical_prediction":
            clinical_result["prediction"],

        "clinical_probability":
            round(
                clinical_probability * 100,
                2
            ),

        "image_prediction":
            image_result["prediction"],

        "image_probability":
            round(
                image_probability * 100,
                2
            ),

        "final_probability":
            round(
                final_probability * 100,
                2
            ),

        "confidence":
            round(
                final_probability * 100,
                2
            ),

        "risk_level":
            clinical_result["risk_level"],

        "stage":
            stage,

        "stage_number":
            stage_number
    }


# ============================================================
# HISTORY
# ============================================================

prediction_history = []


@app.get("/history")
def get_history():

    return {

        "status": "Success",

        "count":
            len(prediction_history),

        "history":
            prediction_history
    }


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    print()
    print("=" * 60)
    print("NephroPredictor API")
    print("=" * 60)

    print(
        "Clinical Model:",
        "Loaded"
        if clinical_model is not None
        else "Unavailable"
    )

    print(
        "Scaler:",
        "Loaded"
        if scaler is not None
        else "Unavailable"
    )

    print(
        "Stage Model:",
        "Loaded"
        if stage_model is not None
        else "Unavailable"
    )

    print(
        "Image Model:",
        "Loaded"
        if image_model is not None
        else "Unavailable"
    )

    print(
        "PyTorch:",
        "Available"
        if TORCH_AVAILABLE
        else "Unavailable"
    )

    print(
        "U-Net:",
        "Loaded"
        if unet_model is not None
        else "Unavailable"
    )

    print("=" * 60)
    print()

