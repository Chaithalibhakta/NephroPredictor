import joblib

model = joblib.load("../models/best_ckd_model.pkl")

print(model.feature_names_in_)