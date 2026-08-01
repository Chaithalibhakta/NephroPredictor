import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load processed dataset
df = pd.read_csv("../dataset/clinical/processed_ckd_dataset.csv")

# Features
X = df.drop(["ckd_pred", "ckd_stage"], axis=1)

# Load trained model
model = joblib.load("../models/best_ckd_model.pkl")

print("Loaded Model:", type(model))

# Use a sample of data (faster and avoids memory issues)
X_sample = X.sample(min(100, len(X)), random_state=42)

# Create SHAP Explainer
explainer = shap.Explainer(model, X_sample)

# Calculate SHAP values
shap_values = explainer(X_sample)

# Summary Plot
plt.figure(figsize=(10, 6))
shap.plots.beeswarm(shap_values, show=False)

plt.tight_layout()
plt.savefig("../reports/shap_summary.png", dpi=300)

print("SHAP Summary Plot generated successfully!")