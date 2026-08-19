import pandas as pd

df = pd.read_csv("../dataset/clinical/processed_ckd_dataset.csv")

print(df.groupby("ckd_pred")[["gfr", "serum_creatinine"]].mean())
import joblib

model = joblib.load("../models/best_ckd_model.pkl")

print(model)

print(type(model))
print(model.classes_)

