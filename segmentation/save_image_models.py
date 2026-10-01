import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

BASE_DIR = r"C:\Users\HP\OneDrive\Desktop\NephroPredictorr"

FEATURE_DIR = os.path.join(BASE_DIR, "segmentation", "features")
MODEL_DIR = os.path.join(BASE_DIR, "models")

TRAIN_PATH = os.path.join(
    FEATURE_DIR,
    "train_image_features.csv"
)

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 70)
print("LOADING IMAGE FEATURES")
print("=" * 70)

df = pd.read_csv(TRAIN_PATH)

feature_columns = [
    "kidney_area",
    "kidney_perimeter",
    "kidney_width",
    "kidney_height",
    "aspect_ratio",
    "circularity",
    "mean_intensity",
    "std_intensity",
    "min_intensity",
    "max_intensity",
    "median_intensity",
    "roi_mean",
    "roi_std"
]

X = df[feature_columns]
y = df["diagnosis"]

print("Training samples:", len(df))
print("Features:", len(feature_columns))

# --------------------------------------------------
# RANDOM FOREST
# --------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf.fit(X, y)

rf_path = os.path.join(
    MODEL_DIR,
    "image_random_forest.pkl"
)

joblib.dump(rf, rf_path)

print("Random Forest saved:")
print(rf_path)

# --------------------------------------------------
# XGBOOST
# --------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING XGBOOST")
print("=" * 70)

xgb = XGBClassifier(
    n_estimators=200,
    random_state=42,
    eval_metric="logloss"
)

xgb.fit(X, y)

xgb_path = os.path.join(
    MODEL_DIR,
    "image_xgboost.pkl"
)

joblib.dump(xgb, xgb_path)

print("XGBoost saved:")
print(xgb_path)

print("\n" + "=" * 70)
print("IMAGE MODELS SAVED SUCCESSFULLY")
print("=" * 70)