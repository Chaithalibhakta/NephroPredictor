import joblib

scaler = joblib.load("../models/scaler.pkl")

print(type(scaler))
print("Mean:")
print(scaler.mean_)

print("\nScale:")
print(scaler.scale_)