import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "image_random_forest.pkl"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "segmentation",
    "features",
    "image_feature_importance.csv"
)

FEATURE_NAMES = [
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

model = joblib.load(MODEL_PATH)

importance = model.feature_importances_

result = pd.DataFrame({
    "feature": FEATURE_NAMES,
    "importance": importance
})

result = result.sort_values(
    by="importance",
    ascending=False
)

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

result.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 50)
print("IMAGE FEATURE IMPORTANCE")
print("=" * 50)

print(result.to_string(index=False))

print("=" * 50)

print(
    f"\nSaved to:\n{OUTPUT_PATH}"
)