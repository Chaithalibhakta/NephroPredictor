from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="NephroPredictor API")

# -----------------------------
# Load Model & Scaler
# -----------------------------
model = joblib.load("../models/best_ckd_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

# -----------------------------
# Temporary Patient History
# -----------------------------
patient_history = []

# -----------------------------
# Patient Input Model
# -----------------------------
class PatientData(BaseModel):
    serum_creatinine: float
    gfr: float
    bun: float
    serum_calcium: float
    ana: int
    c3_c4: int
    hematuria: int
    oxalate_levels: float
    urine_ph: float
    blood_pressure: float
    physical_activity: int
    diet: int
    water_intake: float
    smoking: int
    alcohol: int
    painkiller_usage: int
    family_history: int
    weight_changes: int
    stress_level: int
    months: int


# -----------------------------
# Home API
# -----------------------------
@app.get("/")
def home():
    return {
        "message": "Welcome to NephroPredictor API",
        "status": "Running Successfully"
    }


# -----------------------------
# Prediction API
# -----------------------------
@app.post("/predict")
def predict(data: PatientData):
    try:

        # Create DataFrame
        df = pd.DataFrame([{
            "serum_creatinine": data.serum_creatinine,
            "gfr": data.gfr,
            "bun": data.bun,
            "serum_calcium": data.serum_calcium,
            "ana": data.ana,
            "c3_c4": data.c3_c4,
            "hematuria": data.hematuria,
            "oxalate_levels": data.oxalate_levels,
            "urine_ph": data.urine_ph,
            "blood_pressure": data.blood_pressure,
            "physical_activity": data.physical_activity,
            "diet": data.diet,
            "water_intake": data.water_intake,
            "smoking": data.smoking,
            "alcohol": data.alcohol,
            "painkiller_usage": data.painkiller_usage,
            "family_history": data.family_history,
            "weight_changes": data.weight_changes,
            "stress_level": data.stress_level,
            "months": data.months
        }])

        # Scale Data
        scaled = scaler.transform(df)

        # Prediction
        prediction = model.predict(scaled)[0]

        # Confidence
        probability = model.predict_proba(scaled)[0]
        confidence = round(float(max(probability)) * 100, 2)

        # Result
        result = "CKD Detected" if prediction == 1 else "No CKD"

        # Risk Level
        if confidence >= 90:
            risk = "High"
        elif confidence >= 70:
            risk = "Moderate"
        else:
            risk = "Low"

        # Save History
        patient_history.append({
            "serum_creatinine": data.serum_creatinine,
            "gfr": data.gfr,
            "prediction": result,
            "confidence": confidence,
            "risk_level": risk
        })

        return {
            "prediction": result,
            "confidence": confidence,
            "risk_level": risk
        }

    except Exception as e:
        return {
            "status": "Error",
            "message": str(e)
        }


# -----------------------------
# Patient History API
# -----------------------------
@app.get("/history")
def history():
    return {
        "total_predictions": len(patient_history),
        "history": patient_history
    }


# -----------------------------
# Health Check API
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "Healthy",
        "model_loaded": True
    }