import joblib

BASE_DIR = r"C:\Users\HP\OneDrive\Desktop\NephroPredictorr"

model = joblib.load(
    BASE_DIR + r"\models\best_ckd_model.pkl"
)

scaler = joblib.load(
    BASE_DIR + r"\models\scaler.pkl"
)

encoders = joblib.load(
    BASE_DIR + r"\models\label_encoders.pkl"
)

print("========== MODEL ==========")
print(model)

print("\n========== SCALER ==========")
print(type(scaler))

if hasattr(scaler, "feature_names_in_"):
    print("Scaler features:")
    print(scaler.feature_names_in_)

if hasattr(scaler, "n_features_in_"):
    print("Scaler number of features:")
    print(scaler.n_features_in_)

print("\n========== LABEL ENCODERS ==========")
print(type(encoders))

if isinstance(encoders, dict):
    print("Encoder columns:")
    for column, encoder in encoders.items():
        print("\n", column)
        print("Classes:", encoder.classes_)
else:
    print(encoders)