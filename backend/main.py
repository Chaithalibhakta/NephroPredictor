from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="NephroPredictor API")

# ======================================================
# Load Models
# ======================================================

model = joblib.load("../models/best_ckd_model.pkl")
scaler = joblib.load("../models/scaler.pkl")
stage_model = joblib.load("../models/stage_model.pkl")

patient_history = []

print("=" * 50)
print("CKD Model Loaded")
print(type(model))
print(model.classes_)
print("=" * 50)

# ======================================================
# Input Model
# ======================================================

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


# ======================================================
# Home
# ======================================================

@app.get("/")
def home():
    return {
        "message": "Welcome to NephroPredictor API",
        "status": "Running Successfully"
    }


# ======================================================
# Prediction
# ======================================================

@app.post("/predict")
def predict(data: PatientData):

    try:

        df = pd.DataFrame([[
            data.serum_creatinine,
            data.gfr,
            data.bun,
            data.serum_calcium,
            data.oxalate_levels,
            data.urine_ph,
            data.blood_pressure,
            data.ana,
            data.c3_c4,
            data.hematuria,
            data.smoking,
            data.alcohol,
            data.painkiller_usage,
            data.family_history,
            data.physical_activity,
            data.diet,
            data.water_intake,
            data.weight_changes,
            data.stress_level,
            data.months
        ]],
        columns=[
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
        ])

        print("\n================ INPUT ================")
        print(df)

        scaled = scaler.transform(df)

        prediction = int(model.predict(scaled)[0])

        probabilities = model.predict_proba(scaled)[0]

        print("Prediction :", prediction)
        print("Probability :", probabilities)

        # 0 = CKD
        # 1 = Healthy

        if prediction == 0:

            result = "CKD Detected"

            stage = int(stage_model.predict(df)[0])
            stage_names = {
                            1: "Stage 1 (Mild Kidney Damage)",
                            2: "Stage 2 (Mild Loss of Kidney Function)",
                            3: "Stage 3 (Moderate CKD)",
                            4: "Stage 4 (Severe CKD)",
                            5: "Stage 5 (Kidney Failure)"
                        }

            confidence = round(float(probabilities[0]) * 100, 2)

            if stage == 1:
                risk = "Low"

            elif stage == 2:
                risk = "Moderate"

            elif stage == 3:
                risk = "Moderate"

            elif stage == 4:
                risk = "High"

            elif stage == 5:
                risk = "Critical"

            else:
                risk = "Unknown"

        else:

            result = "Healthy"

            stage = None

            confidence = round(float(probabilities[1]) * 100, 2)

            risk = "Low"

        patient_history.append({
            "prediction": result,
            "stage": stage,
            "confidence": confidence,
            "risk_level": risk
        })

        print("\n================ OUTPUT ================")
        print(result)
        print(stage)
        print(confidence)
        print(risk)

        return {
                "prediction": result,
                "stage": stage_names.get(stage) if stage is not None else None,
                "confidence": confidence,
                "risk_level": risk
            }

    except Exception as e:

        print(e)

        return {
            "status": "Error",
            "message": str(e)
        }
        # ======================================================
# Prediction History
# ======================================================

@app.get("/history")
def history():

    return {
        "total_predictions": len(patient_history),
        "history": patient_history
    }


# ======================================================
# Health Check
# ======================================================

@app.get("/health")
def health():

    return {
        "status": "Healthy",
        "model_loaded": True,
        "ckd_model": "Loaded",
        "stage_model": "Loaded"
    }