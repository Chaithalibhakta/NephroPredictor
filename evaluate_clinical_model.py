import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

BASE_DIR = r"C:\Users\HP\OneDrive\Desktop\NephroPredictorr"

DATA_PATH = BASE_DIR + r"\dataset\clinical\processed_ckd_dataset.csv"
MODEL_PATH = BASE_DIR + r"\models\best_ckd_model.pkl"

# Load data
df = pd.read_csv(DATA_PATH)

# Exact features used by the clinical model
features = [
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

X = df[features]
y = df["ckd_pred"]

# Load trained model
model = joblib.load(MODEL_PATH)

# Prediction
y_pred = model.predict(X)

# Metrics
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, zero_division=0)
recall = recall_score(y, y_pred, zero_division=0)
f1 = f1_score(y, y_pred, zero_division=0)

print("\n========== CLINICAL MODEL RESULTS ==========")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\n========== CONFUSION MATRIX ==========")
print(confusion_matrix(y, y_pred))

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y, y_pred, zero_division=0))