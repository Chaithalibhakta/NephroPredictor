import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
MODEL_PATH = BASE_DIR + r"\models\clinical_holdout_model.pkl"

# Load dataset
df = pd.read_csv(DATA_PATH)

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

# 80/20 stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Train new model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Test on untouched test set
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n========== CLINICAL HOLDOUT RESULTS ==========")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\n========== CONFUSION MATRIX ==========")
print(confusion_matrix(y_test, y_pred))

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred, zero_division=0))

# Save new model
joblib.dump(model, MODEL_PATH)

print("\nModel saved to:")
print(MODEL_PATH)