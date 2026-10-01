import pandas as pd
import os

# ============================================================
# PATH
# ============================================================

FEATURE_FILE = r"C:\Users\HP\OneDrive\Desktop\NephroPredictorr\segmentation\features\train_image_features.csv"

# ============================================================
# LOAD FEATURE DATA
# ============================================================

df = pd.read_csv(FEATURE_FILE)

# Find images where U-Net predicted no kidney pixels
zero_df = df[df["kidney_area"] == 0].copy()

print("=" * 60)
print("ZERO KIDNEY-AREA IMAGE INSPECTION")
print("=" * 60)

print("\nTotal images:", len(df))
print("Zero kidney-area images:", len(zero_df))

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n1. CLASS DISTRIBUTION OF ZERO-AREA IMAGES")
print("-" * 60)

print(zero_df["diagnosis"].value_counts())

print("\nClass names:")
print(zero_df["class_name"].value_counts())

# ============================================================
# IMAGE IDs
# ============================================================

print("\n2. ZERO-AREA IMAGE IDs")
print("-" * 60)

for image_id in zero_df["image_id"]:
    print(image_id)

# ============================================================
# SAVE LIST
# ============================================================

OUTPUT_FILE = r"C:\Users\HP\OneDrive\Desktop\NephroPredictorr\segmentation\features\zero_kidney_area_images.csv"

zero_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 60)
print("RESULT SAVED")
print("=" * 60)

print(OUTPUT_FILE)