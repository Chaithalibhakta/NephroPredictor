import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

BASE_DIR = r"C:\Users\HP\OneDrive\Desktop\NephroPredictorr"

MODEL_PATH = os.path.join(BASE_DIR, "models", "stage_model.pkl")
TEST_PATH = os.path.join(BASE_DIR, "dataset", "clinical", "clinical_test.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "segmentation", "features")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("LOADING STAGE MODEL")
print("=" * 70)

model = joblib.load(MODEL_PATH)
df = pd.read_csv(TEST_PATH)

print("Test samples:", len(df))
print("Columns:", df.columns.tolist())

# Exact 20 features expected by the model
features = [
    'serum_creatinine',
    'gfr',
    'bun',
    'serum_calcium',
    'oxalate_levels',
    'urine_ph',
    'blood_pressure',
    'ana',
    'c3_c4',
    'hematuria',
    'smoking',
    'alcohol',
    'painkiller_usage',
    'family_history',
    'physical_activity',
    'diet',
    'water_intake',
    'weight_changes',
    'stress_level',
    'months'
]

X_test = df[features]
y_test = df["ckd_stage"]

print("\nStage distribution:")
print(y_test.value_counts().sort_index())

print("\n" + "=" * 70)
print("STAGE MODEL EVALUATION")
print("=" * 70)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4, 5])
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[1, 2, 3, 4, 5],
    target_names=[
        "Stage 1",
        "Stage 2",
        "Stage 3",
        "Stage 4",
        "Stage 5"
    ],
    zero_division=0
))

# Save results
summary = pd.DataFrame([{
    "Model": "Random Forest Stage Model",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1
}])

summary_path = os.path.join(
    OUTPUT_DIR,
    "stage_model_test_summary.csv"
)

cm_path = os.path.join(
    OUTPUT_DIR,
    "stage_model_test_confusion_matrix.csv"
)

summary.to_csv(summary_path, index=False)

pd.DataFrame(
    cm,
    index=["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"],
    columns=["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"]
).to_csv(cm_path)

print("\nSaved:")
print(summary_path)
print(cm_path)

print("\n" + "=" * 70)
print("STAGE MODEL EVALUATION COMPLETED")
print("=" * 70)