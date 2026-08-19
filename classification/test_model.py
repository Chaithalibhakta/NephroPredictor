import joblib
import pandas as pd

model = joblib.load("../models/best_ckd_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

df = pd.read_csv("../dataset/clinical/processed_ckd_dataset.csv")

healthy = df[df["ckd_pred"] == 1].head(1)
ckd = df[df["ckd_pred"] == 0].head(1)

print("Healthy sample")
print(healthy)

print("\nCKD sample")
print(ckd)

Xh = healthy.drop(["ckd_pred", "ckd_stage"], axis=1)
Xc = ckd.drop(["ckd_pred", "ckd_stage"], axis=1)

print("\nHealthy prediction")
print(model.predict(Xh))
print(model.predict_proba(Xh))

print("\nCKD prediction")
print(model.predict(Xc))
print(model.predict_proba(Xc))
import joblib

model = joblib.load("../models/best_ckd_model.pkl")

print(model)