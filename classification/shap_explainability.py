import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("../dataset/clinical/processed_ckd_dataset.csv")

X = df.drop(["ckd_pred", "ckd_stage"], axis=1)

# Load Model
model = joblib.load("../models/best_ckd_model.pkl")

print("Loaded Model:", type(model))

# Get Booster
booster = model.get_booster()

# Sample Data
X_sample = X.sample(200, random_state=42)

# SHAP Explainer
explainer = shap.TreeExplainer(booster)

shap_values = explainer.shap_values(X_sample)

# Summary Plot
shap.summary_plot(shap_values, X_sample, show=False)

plt.tight_layout()
plt.savefig("../results/shap_summary.png", dpi=300)

print("SHAP Summary Plot Generated Successfully!")