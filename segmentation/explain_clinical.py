import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_ckd_model.pkl"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "features",
    "clinical_feature_importance.csv"
)

FEATURE_NAMES = [
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

model = joblib.load(MODEL_PATH)

importance = model.feature_importances_

result = pd.DataFrame({
    "feature": FEATURE_NAMES,
    "importance": importance
})

result = result.sort_values(
    by="importance",
    ascending=False
)

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

result.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 50)
print("CLINICAL FEATURE IMPORTANCE")
print("=" * 50)

print(result.to_string(index=False))

print("=" * 50)

print(
    f"\nSaved to:\n{OUTPUT_PATH}"
)