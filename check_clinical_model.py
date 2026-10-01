import joblib

MODEL_PATH = r"C:\Users\HP\OneDrive\Desktop\NephroPredictorr\models\best_ckd_model.pkl"

model = joblib.load(MODEL_PATH)

print("MODEL TYPE:")
print(type(model))

print("\nMODEL:")
print(model)

print("\nFEATURE NAMES:")
if hasattr(model, "feature_names_in_"):
    print(model.feature_names_in_)
else:
    print("Model does not contain feature_names_in_")

print("\nNUMBER OF FEATURES:")
if hasattr(model, "n_features_in_"):
    print(model.n_features_in_)
else:
    print("Not available")