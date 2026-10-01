import os
import pandas as pd

BASE_DIR = r"C:\Users\HP\OneDrive\Desktop\NephroPredictorr"
FEATURE_DIR = os.path.join(BASE_DIR, "segmentation", "features")

rows = [
    {
        "Model": "Clinical Random Forest",
        "Task": "CKD Classification",
        "Accuracy": 0.99998,
        "Precision": 1.00000,
        "Recall": 0.99996,
        "F1": 0.99998
    },
    {
        "Model": "Clinical Stage Random Forest",
        "Task": "CKD Stage Classification",
        "Accuracy": 1.00000,
        "Precision": 1.00000,
        "Recall": 1.00000,
        "F1": 1.00000
    },
    {
        "Model": "Image Random Forest",
        "Task": "Ultrasound Classification",
        "Accuracy": 0.7517,
        "Precision": 0.7766,
        "Recall": 0.9522,
        "F1": 0.8555
    },
    {
        "Model": "Image XGBoost",
        "Task": "Ultrasound Classification",
        "Accuracy": 0.7416,
        "Precision": 0.7887,
        "Recall": 0.9087,
        "F1": 0.8444
    }
]

df = pd.DataFrame(rows)

output_path = os.path.join(
    FEATURE_DIR,
    "final_model_metrics.csv"
)

df.to_csv(output_path, index=False)

print("=" * 60)
print("FINAL MODEL METRICS")
print("=" * 60)
print(df.to_string(index=False))
print()
print("Saved to:")
print(output_path)