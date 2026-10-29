
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import numpy as np
import joblib
import os
import json
import io
import sys

from PIL import Image

# --------------------------------------------------
# PYTORCH
# --------------------------------------------------

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
    print(f"PyTorch loaded successfully: {torch.__version__}")

except Exception as e:
    TORCH_AVAILABLE = False
    print(f"PyTorch import failed: {type(e).__name__}: {e}")


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="NephroPredictor API",
    description="CKD Early Detection API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# --------------------------------------------------
# DIRECTORIES
# --------------------------------------------------

BASE_DIR = r"C:\Users\HP\OneDrive\Desktop\NephroPredictorr"
MODEL_DIR = os.path.join(BASE_DIR, "models")


# --------------------------------------------------
# CLINICAL FEATURES
# --------------------------------------------------

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


# --------------------------------------------------
# LOAD CLINICAL MODEL
# --------------------------------------------------

clinical_model = None

try:
    clinical_model = joblib.load(
        os.path.join(MODEL_DIR, "best_ckd_model.pkl")
    )
    print("Clinical model loaded successfully.")

except Exception as e:
    print("Clinical model loading failed:", e)


# --------------------------------------------------
# LOAD SCALER
# --------------------------------------------------

scaler = None

try:
    scaler = joblib.load(
        os.path.join(MODEL_DIR, "scaler.pkl")
    )
    print("Scaler loaded successfully.")

except Exception as e:
    print("Scaler loading failed:", e)


# --------------------------------------------------
# LOAD STAGE MODEL
# --------------------------------------------------

stage_model = None

try:
    stage_model = joblib.load(
        os.path.join(MODEL_DIR, "stage_model.pkl")
    )
    print("Stage model loaded successfully.")

except Exception as e:
    print("Stage model loading failed:", e)


# --------------------------------------------------
# LOAD U-NET SEGMENTATION MODEL
# --------------------------------------------------

unet_model = None

if TORCH_AVAILABLE:
    try:
        segmentation_dir = os.path.join(
            BASE_DIR,
            "segmentation"
        )

        if segmentation_dir not in sys.path:
            sys.path.insert(0, segmentation_dir)

        from unet_model import UNet

        unet_model = UNet()

        unet_path = os.path.join(
            BASE_DIR,
            "segmentation",
            "models",
            "best_unet.pth"
        )

        state_dict = torch.load(
            unet_path,
            map_location="cpu"
        )

        unet_model.load_state_dict(state_dict)
        unet_model.eval()

        print("U-Net loaded successfully.")

    except Exception as e:
        print("U-Net loading failed:", e)
        unet_model = None


# --------------------------------------------------
# LOAD IMAGE CLASSIFIER
# --------------------------------------------------

image_model = None

try:
    image_model_path = os.path.join(
        MODEL_DIR,
        "image_random_forest.pkl"
    )

    image_model = joblib.load(image_model_path)

    print("Image model loaded successfully.")
    print("Image model classes:", image_model.classes_)

except Exception as e:
    print("Image model loading failed:", e)
    image_model = None


# --------------------------------------------------
# PATIENT DATA MODEL
# --------------------------------------------------

class PatientData(BaseModel):
    serum_creatinine: float
    gfr: float
    bun: float
    serum_calcium: float
    oxalate_levels: float
    urine_ph: float
    blood_pressure: float
    ana: float
    c3_c4: float
    hematuria: float
    smoking: float
    alcohol: float
    painkiller_usage: float
    family_history: float
    physical_activity: float
    diet: float
    water_intake: float
    weight_changes: float
    stress_level: float
    months: float


# --------------------------------------------------
# ROOT ENDPOINT
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "NephroPredictor API is running"
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "Healthy",
        "model_loaded": clinical_model is not None,
        "ckd_model": (
            "Loaded" if clinical_model is not None
            else "Not Loaded"
        ),
        "scaler": (
            "Loaded" if scaler is not None
            else "Not Loaded"
        ),
        "stage_model": (
            "Loaded" if stage_model is not None
            else "Not Loaded"
        ),
        "unet_model": (
            "Loaded" if unet_model is not None
            else "Not Loaded"
        ),
        "image_model": (
            "Loaded" if image_model is not None
            else "Not Loaded"
        ),
        "torch_available": TORCH_AVAILABLE
    }


# --------------------------------------------------
# CLINICAL PREDICTION
# --------------------------------------------------

@app.post("/predict")
def predict(patient: PatientData):

    if clinical_model is None:
        return {
            "status": "Error",
            "message": "Clinical model not loaded"
        }

    if scaler is None:
        return {
            "status": "Error",
            "message": "Scaler not loaded"
        }

    try:
        patient_data = patient.model_dump()

        input_df = pd.DataFrame(
            [patient_data],
            columns=FEATURE_COLUMNS
        )

        input_scaled = pd.DataFrame(
            scaler.transform(input_df),
            columns=FEATURE_COLUMNS
        )

        prediction = int(
            clinical_model.predict(input_scaled)[0]
        )

        probabilities = clinical_model.predict_proba(
            input_scaled
        )[0]

        model_classes = list(clinical_model.classes_)

        predicted_class_index = model_classes.index(prediction)

        confidence = float(
            probabilities[predicted_class_index] * 100
        )

        if 0 not in model_classes:
            return {
                "status": "Error",
                "message": "CKD class 0 was not found in clinical model."
            }

        ckd_class_index = model_classes.index(0)

        ckd_probability = float(
            probabilities[ckd_class_index]
        )

        if prediction == 0:
            prediction_label = "CKD Detected"

            if confidence >= 80:
                risk_level = "High"
            elif confidence >= 60:
                risk_level = "Moderate"
            else:
                risk_level = "Low"

        else:
            prediction_label = "Healthy"

            if confidence >= 80:
                risk_level = "Low"
            elif confidence >= 60:
                risk_level = "Moderate"
            else:
                risk_level = "High"

        stage = None
        stage_number = None

        if prediction == 0 and stage_model is not None:
            try:
                stage_prediction = stage_model.predict(
                    input_scaled
                )[0]

                stage_number = int(
                    round(float(stage_prediction))
                )

                stage = f"Stage {stage_number}"

            except Exception as e:
                print("Stage prediction failed:", e)

        return {
            "status": "Success",
            "prediction": prediction_label,
            "confidence": round(confidence, 2),
            "ckd_probability": round(
                ckd_probability * 100, 2
            ),
            "risk_level": risk_level,
            "stage": stage,
            "stage_number": stage_number
        }

    except Exception as e:
        print("Clinical prediction failed:", repr(e))

        return {
            "status": "Error",
            "message": str(e)
        }


# --------------------------------------------------
# ULTRASOUND IMAGE PREDICTION
# --------------------------------------------------

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):

    if image_model is None:
        return {
            "status": "Error",
            "message": "Image model not loaded"
        }

    if unet_model is None or not TORCH_AVAILABLE:
        return {
            "status": "Error",
            "message": "U-Net model or PyTorch not available"
        }

    try:
        # Read image
        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        image = image.resize((256, 256))

        image_array = (
            np.array(image).astype(np.float32) / 255.0
        )

        # Convert image to tensor
        image_tensor = torch.tensor(
            image_array
        ).permute(2, 0, 1).unsqueeze(0)

        # U-Net segmentation
        with torch.no_grad():
            output = unet_model(image_tensor)
            mask_probability = torch.sigmoid(output)
            mask = (mask_probability > 0.5).float()

        mask_np = mask.squeeze().cpu().numpy()
        kidney_pixels = mask_np > 0

        # Shape features
        kidney_area = float(np.sum(kidney_pixels))

        rows, cols = np.where(kidney_pixels)

        if len(rows) > 0:
            kidney_height = float(
                rows.max() - rows.min() + 1
            )

            kidney_width = float(
                cols.max() - cols.min() + 1
            )

        else:
            kidney_height = 0.0
            kidney_width = 0.0

        if kidney_height > 0:
            aspect_ratio = kidney_width / kidney_height
        else:
            aspect_ratio = 0.0

        # Intensity features
        gray_image = np.array(
            image.convert("L")
        ).astype(np.float32)

        if np.any(kidney_pixels):
            kidney_intensity = gray_image[kidney_pixels]

            mean_intensity = float(
                np.mean(kidney_intensity)
            )

            std_intensity = float(
                np.std(kidney_intensity)
            )

            min_intensity = float(
                np.min(kidney_intensity)
            )

            max_intensity = float(
                np.max(kidney_intensity)
            )

            median_intensity = float(
                np.median(kidney_intensity)
            )

        else:
            mean_intensity = 0.0
            std_intensity = 0.0
            min_intensity = 0.0
            max_intensity = 0.0
            median_intensity = 0.0

        roi_mean = mean_intensity
        roi_std = std_intensity

        # Perimeter and circularity
        mask_uint8 = kidney_pixels.astype(np.uint8)

        horizontal_edges = np.abs(
            np.diff(
                mask_uint8.astype(np.int8),
                axis=1
            )
        ).sum()

        vertical_edges = np.abs(
            np.diff(
                mask_uint8.astype(np.int8),
                axis=0
            )
        ).sum()

        kidney_perimeter = float(
            horizontal_edges + vertical_edges
        )

        if kidney_perimeter > 0:
            circularity = float(
                (4 * np.pi * kidney_area)
                / (kidney_perimeter ** 2)
            )
        else:
            circularity = 0.0

        # Construct the feature DataFrame
        image_features = pd.DataFrame(
            [[
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
            ]],
            columns=[
                "kidney_area",
                "kidney_perimeter",
                "kidney_width",
                "kidney_height",
                "aspect_ratio",
                "circularity",
                "mean_intensity",
                "std_intensity",
                "min_intensity",
                "max_intensity",
                "median_intensity",
                "roi_mean",
                "roi_std"
            ]
        )

        # Predict class
        image_prediction = int(
            image_model.predict(image_features)[0]
        )
        print("DEBUG - Model classes:", image_model.classes_)
        print("DEBUG - Predicted class:", image_prediction)
        print(
    "DEBUG - Probabilities:",
    image_model.predict_proba(image_features)[0]
)
        image_probabilities = image_model.predict_proba(
            image_features
        )[0]

        model_classes = list(image_model.classes_)

        print("Image model classes:", model_classes)
        print(
            "Extracted image features:",
            image_features.to_dict(orient="records")
        )
        print("Raw image prediction:", image_prediction)
        print("Class probabilities:", image_probabilities)

        # This assumes training label 1 = Pathological
        # and training label 0 = Healthy.
        if 1 not in model_classes or 0 not in model_classes:
            return {
                "status": "Error",
                "message": (
                    "Expected classes 0 and 1 were not both "
                    "found in the image model."
                )
            }

        pathological_class_index = model_classes.index(1)

        image_probability = float(
            image_probabilities[pathological_class_index] * 100
        )

        if image_prediction == 1:
            image_label = "Pathological"
        elif image_prediction == 0:
            image_label = "Healthy"
        else:
            return {
                "status": "Error",
                "message": (
                    f"Unexpected image prediction class: "
                    f"{image_prediction}"
                )
            }

        return {
            "status": "Success",
            "prediction": image_label,
            "image_probability": round(
                image_probability, 2
            )
        }

    except Exception as e:
        print("Image prediction failed:", repr(e))

        return {
            "status": "Error",
            "message": str(e)
        }


# --------------------------------------------------
# CLINICAL + IMAGE FUSION
# --------------------------------------------------

@app.post("/fusion-predict")
async def fusion_predict(
    patient: str = Form(...),
    file: UploadFile = File(...)
):

    try:
        patient_dict = json.loads(patient)

        patient_model = PatientData(**patient_dict)

        clinical_result = predict(patient_model)

        if clinical_result.get("status") != "Success":
            return clinical_result

        image_result = await predict_image(file)

        if image_result.get("status") != "Success":
            return image_result

        # Probability that clinical model predicts CKD
        clinical_probability = (
            clinical_result["ckd_probability"] / 100
        )

        # Probability that image model predicts pathological
        image_probability = (
            image_result["image_probability"] / 100
        )

        # Existing project weights
        clinical_weight = 0.60
        image_weight = 0.40

        final_probability = (
            clinical_weight * clinical_probability
            + image_weight * image_probability
        )

        if final_probability >= 0.50:
            final_prediction = "CKD Detected"
        else:
            final_prediction = "Healthy"

        confidence = final_probability * 100

        if final_probability >= 0.80:
            risk_level = "High"
        elif final_probability >= 0.50:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        stage = None
        stage_number = None

        if final_prediction == "CKD Detected":
            stage = clinical_result.get("stage")
            stage_number = clinical_result.get("stage_number")

        return {
            "status": "Success",
            "clinical_prediction": clinical_result["prediction"],
            "clinical_probability": round(
                clinical_probability * 100, 2
            ),
            "image_prediction": image_result["prediction"],
            "image_probability": round(
                image_probability * 100, 2
            ),
            "final_prediction": final_prediction,
            "final_probability": round(
                final_probability * 100, 2
            ),
            "confidence": round(confidence, 2),
            "risk_level": risk_level,
            "stage": stage,
            "stage_number": stage_number
        }

    except Exception as e:
        print("Fusion prediction failed:", repr(e))

        return {
            "status": "Error",
            "message": str(e)
        }

