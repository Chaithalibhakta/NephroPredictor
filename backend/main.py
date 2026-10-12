
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


# ============================================================
# TORCH
# ============================================================


try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    print(f"PyTorch loaded successfully: {torch.__version__}")

except Exception as e:
    TORCH_AVAILABLE = False
    print(f"PyTorch import failed: {type(e).__name__}: {e}")




# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="NephroPredictor API",
    description="CKD Early Detection API",
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
# BASE DIRECTORIES
# ============================================================

BASE_DIR = r"C:\Users\HP\OneDrive\Desktop\NephroPredictorr"

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# CLINICAL FEATURE COLUMNS
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
# CLINICAL MODEL
# ============================================================

clinical_model = None

try:
    clinical_model = joblib.load(
        os.path.join(
            MODEL_DIR,
            "best_ckd_model.pkl"
        )
    )

    print("Clinical model loaded successfully.")

except Exception as e:
    print(
        "Clinical model loading failed:",
        e
    )


# ============================================================
# SCALER
# ============================================================

scaler = None

try:
    scaler = joblib.load(
        os.path.join(
            MODEL_DIR,
            "scaler.pkl"
        )
    )

    print("Scaler loaded successfully.")

except Exception as e:
    print(
        "Scaler loading failed:",
        e
    )


# ============================================================
# STAGE MODEL
# ============================================================

stage_model = None

try:
    stage_model = joblib.load(
        os.path.join(
            MODEL_DIR,
            "stage_model.pkl"
        )
    )

    print("Stage model loaded successfully.")

except Exception as e:
    print(
        "Stage model loading failed:",
        e
    )


# ============================================================
# U-NET
# ============================================================

unet_model = None

if TORCH_AVAILABLE:

    try:

        # Add segmentation directory so the exact
        # trained U-Net architecture can be imported.

        segmentation_dir = os.path.join(
            BASE_DIR,
            "segmentation"
        )

        if segmentation_dir not in sys.path:
            sys.path.insert(
                0,
                segmentation_dir
            )

        from unet_model import UNet

        # Create U-Net

        unet_model = UNet()

        # Correct trained U-Net path

        unet_path = os.path.join(
            BASE_DIR,
            "segmentation",
            "models",
            "best_unet.pth"
        )

        # Load trained weights

        state_dict = torch.load(
            unet_path,
            map_location="cpu"
        )

        unet_model.load_state_dict(
            state_dict
        )

        unet_model.eval()

        print("U-Net loaded successfully.")

    except Exception as e:

        print(
            "U-Net loading failed:",
            e
        )

        unet_model = None


# ============================================================
# IMAGE MODEL
# ============================================================

image_model = None

try:

    # Correct trained image Random Forest path

    image_model_path = os.path.join(
        BASE_DIR,
        "models",
        "image_random_forest.pkl"
    )

    image_model = joblib.load(
        image_model_path
    )

    print("Image model loaded successfully.")

except Exception as e:

    print(
        "Image model loading failed:",
        e
    )

    image_model = None


# ============================================================
# PATIENT MODEL
# ============================================================

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


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "NephroPredictor API is running"
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
            else "Not Loaded",

        "scaler":
            "Loaded"
            if scaler is not None
            else "Not Loaded",

        "stage_model":
            "Loaded"
            if stage_model is not None
            else "Not Loaded",

        "unet_model":
            "Loaded"
            if unet_model is not None
            else "Not Loaded",

        "image_model":
            "Loaded"
            if image_model is not None
            else "Not Loaded",

        "torch_available":
            TORCH_AVAILABLE
    }


# ============================================================
# CLINICAL PREDICTION
# ============================================================

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

        # ----------------------------------------------------
        # Raw input in exact feature order
        # ----------------------------------------------------

        input_df = pd.DataFrame(
            [patient_data],
            columns=FEATURE_COLUMNS
        )

        # ----------------------------------------------------
        # Scale clinical input
        # ----------------------------------------------------

        input_scaled = pd.DataFrame(
            scaler.transform(input_df),
            columns=FEATURE_COLUMNS
        )

        # ----------------------------------------------------
        # Clinical prediction
        # ----------------------------------------------------

        prediction = int(
            clinical_model.predict(
                input_scaled
            )[0]
        )

        probabilities = (
            clinical_model.predict_proba(
                input_scaled
            )[0]
        )

        predicted_class_index = list(
            clinical_model.classes_
        ).index(
            prediction
        )

        confidence = float(
            probabilities[
                predicted_class_index
            ] * 100
        )

        # ----------------------------------------------------
        # CKD probability
        #
        # Class 0 = CKD
        # ----------------------------------------------------

        ckd_class_index = list(
            clinical_model.classes_
        ).index(0)

        ckd_probability = float(
            probabilities[
                ckd_class_index
            ]
        )

        # ----------------------------------------------------
        # Prediction label
        # ----------------------------------------------------

        if prediction == 0:

            prediction_label = "CKD Detected"

        else:

            prediction_label = "Healthy"

        # ----------------------------------------------------
        # Risk level
        #
        # Current project definition:
        # risk is based on model confidence,
        # not CKD stage severity.
        # ----------------------------------------------------

        if prediction == 0:

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

        # ----------------------------------------------------
        # CKD STAGE
        #
        # Stage model was trained using the
        # processed/scaled clinical dataset.
        #
        # Dataset labels are already:
        # 1, 2, 3, 4, 5
        #
        # Therefore NO +1 is required.
        # ----------------------------------------------------

        stage = None
        stage_number = None

        if (
            prediction == 0
            and
            stage_model is not None
        ):

            try:

                stage_prediction = (
                    stage_model.predict(
                        input_scaled
                    )[0]
                )

                stage_number = int(
                    round(
                        float(
                            stage_prediction
                        )
                    )
                )

                stage = (
                    f"Stage {stage_number}"
                )

            except Exception as e:

                print(
                    "Stage prediction failed:",
                    e
                )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {

            "status": "Success",

            "prediction":
                prediction_label,

            "confidence":
                round(
                    confidence,
                    2
                ),

            "risk_level":
                risk_level,

            "stage":
                stage,

            "stage_number":
                stage_number
        }

    except Exception as e:

        return {

            "status": "Error",

            "message":
                str(e)
        }


# ============================================================
# IMAGE PREDICTION
# ============================================================

@app.post("/predict-image")
async def predict_image(
    file: UploadFile = File(...)
):

    if image_model is None:

        return {
            "status": "Error",
            "message": "Image model not loaded"
        }

    if unet_model is None:

        return {
            "status": "Error",
            "message": "U-Net model not loaded"
        }

    try:

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        image = image.resize(
            (256, 256)
        )

        image_array = (
            np.array(image)
            .astype(np.float32)
            / 255.0
        )

        # ----------------------------------------------------
        # U-Net segmentation
        # ----------------------------------------------------

        image_tensor = torch.tensor(
            image_array
        ).permute(
            2,
            0,
            1
        ).unsqueeze(0)

        with torch.no_grad():

            output = unet_model(
                image_tensor
            )

            mask_probability = (
                torch.sigmoid(output)
            )

            mask = (
                mask_probability > 0.5
            ).float()

        mask_np = (
            mask.squeeze().numpy()
        )

        # ----------------------------------------------------
        # Extract kidney features
        # ----------------------------------------------------

        kidney_pixels = (
            mask_np > 0
        )

        kidney_area = float(
            np.sum(kidney_pixels)
        )

        rows, cols = np.where(
            kidney_pixels
        )

        if len(rows) > 0:

            kidney_height = float(
                rows.max()
                - rows.min()
                + 1
            )

            kidney_width = float(
                cols.max()
                - cols.min()
                + 1
            )

        else:

            kidney_height = 0.0
            kidney_width = 0.0

        # ----------------------------------------------------
        # Aspect ratio
        # ----------------------------------------------------

        if kidney_height > 0:

            aspect_ratio = (
                kidney_width
                /
                kidney_height
            )

        else:

            aspect_ratio = 0.0

        # ----------------------------------------------------
        # Intensity features
        # ----------------------------------------------------

        gray_image = np.array(
            image.convert("L")
        ).astype(
            np.float32
        )

        if np.any(kidney_pixels):

            kidney_intensity = (
                gray_image[
                    kidney_pixels
                ]
            )

            mean_intensity = float(
                np.mean(
                    kidney_intensity
                )
            )

            std_intensity = float(
                np.std(
                    kidney_intensity
                )
            )

            min_intensity = float(
                np.min(
                    kidney_intensity
                )
            )

            max_intensity = float(
                np.max(
                    kidney_intensity
                )
            )

            median_intensity = float(
                np.median(
                    kidney_intensity
                )
            )

        else:

            mean_intensity = 0.0
            std_intensity = 0.0
            min_intensity = 0.0
            max_intensity = 0.0
            median_intensity = 0.0

        # ----------------------------------------------------
        # ROI features
        # ----------------------------------------------------

        roi_mean = mean_intensity
        roi_std = std_intensity

        # ----------------------------------------------------
        # Perimeter approximation
        # ----------------------------------------------------

        mask_uint8 = (
            kidney_pixels
            .astype(
                np.uint8
            )
        )

        horizontal_edges = np.abs(
            np.diff(
                mask_uint8.astype(
                    np.int8
                ),
                axis=1
            )
        ).sum()

        vertical_edges = np.abs(
            np.diff(
                mask_uint8.astype(
                    np.int8
                ),
                axis=0
            )
        ).sum()

        kidney_perimeter = float(
            horizontal_edges
            +
            vertical_edges
        )

        # ----------------------------------------------------
        # Circularity
        # ----------------------------------------------------

        if kidney_perimeter > 0:

            circularity = float(
                (
                    4
                    *
                    np.pi
                    *
                    kidney_area
                )
                /
                (
                    kidney_perimeter
                    **
                    2
                )
            )

        else:

            circularity = 0.0

        # ----------------------------------------------------
        # Image feature dataframe
        #
        # Must match the image model's
        # training feature order.
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Image model prediction
        # ----------------------------------------------------

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

        # Class 1 = Pathological

        pathological_class_index = list(
            image_model.classes_
        ).index(1)

        image_probability = float(
            image_probabilities[
                pathological_class_index
            ]
            *
            100
        )

        # ----------------------------------------------------
        # Image label
        # ----------------------------------------------------

        if image_prediction == 1:

            image_label = "Pathological"

        else:

            image_label = "Healthy"

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {

            "status": "Success",

            "prediction":
                image_label,

            "image_probability":
                round(
                    image_probability,
                    2
                )
        }

    except Exception as e:

        return {

            "status": "Error",

            "message":
                str(e)
        }


# ============================================================
# LATE FUSION
# ============================================================

@app.post("/fusion-predict")
async def fusion_predict(

    patient: str = Form(...),

    file: UploadFile = File(...)
):

    try:

        # ----------------------------------------------------
        # Parse patient JSON
        # ----------------------------------------------------

        patient_dict = json.loads(
            patient
        )

        patient_model = PatientData(
            **patient_dict
        )

        # ----------------------------------------------------
        # Clinical prediction
        # ----------------------------------------------------

        clinical_result = predict(
            patient_model
        )

        if (
            clinical_result.get(
                "status"
            )
            !=
            "Success"
        ):

            return clinical_result

        # ----------------------------------------------------
        # Image prediction
        # ----------------------------------------------------

        image_result = await predict_image(
            file
        )

        if (
            image_result.get(
                "status"
            )
            !=
            "Success"
        ):

            return image_result

        # ----------------------------------------------------
        # Clinical probability
        #
        # Convert clinical result into
        # probability of CKD.
        # ----------------------------------------------------

        if (
            clinical_result[
                "prediction"
            ]
            ==
            "CKD Detected"
        ):

            clinical_probability = (
                clinical_result[
                    "confidence"
                ]
                /
                100
            )

        else:

            clinical_probability = (
                1
                -
                (
                    clinical_result[
                        "confidence"
                    ]
                    /
                    100
                )
            )

        # ----------------------------------------------------
        # Image probability
        # ----------------------------------------------------

        image_probability = (
            image_result[
                "image_probability"
            ]
            /
            100
        )

        # ----------------------------------------------------
        # Prototype late fusion
        #
        # Clinical = 60%
        # Image = 40%
        # ----------------------------------------------------

        clinical_weight = 0.60
        image_weight = 0.40

        final_probability = (

            clinical_weight
            *
            clinical_probability

            +

            image_weight
            *
            image_probability
        )

        # ----------------------------------------------------
        # Final prediction
        # ----------------------------------------------------

        if final_probability >= 0.50:

            final_prediction = (
                "CKD Detected"
            )

        else:

            final_prediction = (
                "Healthy"
            )

        # ----------------------------------------------------
        # Final confidence
        # ----------------------------------------------------

        confidence = (
            final_probability
            *
            100
        )

        # ----------------------------------------------------
        # Final risk
        # ----------------------------------------------------

        if final_probability >= 0.80:

            risk_level = "High"

        elif final_probability >= 0.50:

            risk_level = "Moderate"

        else:

            risk_level = "Low"

        # ----------------------------------------------------
        # Stage
        #
        # If final prediction is CKD,
        # use the clinical stage already calculated
        # by the clinical prediction.
        #
        # No +1 is required because the stage model
        # already predicts labels 1-5.
        # ----------------------------------------------------

        stage = None
        stage_number = None

        if (
            final_prediction
            ==
            "CKD Detected"
        ):

            stage = clinical_result.get(
                "stage"
            )

            stage_number = clinical_result.get(
                "stage_number"
            )

        # ----------------------------------------------------
        # Final response
        # ----------------------------------------------------

        return {

            "status":
                "Success",

            "clinical_prediction":
                clinical_result[
                    "prediction"
                ],

            "clinical_probability":
                round(
                    clinical_probability
                    *
                    100,
                    2
                ),

            "image_prediction":
                image_result[
                    "prediction"
                ],

            "image_probability":
                round(
                    image_probability
                    *
                    100,
                    2
                ),

            "final_probability":
                round(
                    final_probability
                    *
                    100,
                    2
                ),

            "confidence":
                round(
                    confidence,
                    2
                ),

            "risk_level":
                risk_level,

            "stage":
                stage,

            "stage_number":
                stage_number
        }

    except Exception as e:

        return {

            "status":
                "Error",

            "message":
                str(e)
        }

