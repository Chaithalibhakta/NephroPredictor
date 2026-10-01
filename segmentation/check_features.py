import pandas as pd
import os

# ============================================================
# PATH
# ============================================================

FEATURE_FILE = r"C:\Users\HP\OneDrive\Desktop\NephroPredictorr\segmentation\features\train_image_features.csv"


# ============================================================
# CHECK FILE
# ============================================================

if not os.path.exists(FEATURE_FILE):
    print("ERROR: Feature file not found!")
    print(FEATURE_FILE)
    exit()

print("=" * 60)
print("IMAGE FEATURE DATASET VALIDATION")
print("=" * 60)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(FEATURE_FILE)

print("\n1. DATASET SHAPE")
print("-" * 60)
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ============================================================
# COLUMN NAMES
# ============================================================

print("\n2. COLUMNS")
print("-" * 60)

for column in df.columns:
    print("-", column)


# ============================================================
# MISSING VALUES
# ============================================================

print("\n3. MISSING VALUES")
print("-" * 60)

missing = df.isnull().sum()

if missing.sum() == 0:
    print("✓ No missing values found.")
else:
    print("Missing values found:")
    print(missing[missing > 0])


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n4. CLASS DISTRIBUTION")
print("-" * 60)

print(df["diagnosis"].value_counts())

print("\nClass names:")
print(df["class_name"].value_counts())


# ============================================================
# ZERO KIDNEY AREA
# ============================================================

print("\n5. ZERO KIDNEY AREA")
print("-" * 60)

zero_area = (df["kidney_area"] == 0).sum()

print("Images with kidney_area = 0:", zero_area)
print(
    "Percentage:",
    round((zero_area / len(df)) * 100, 2),
    "%"
)


# ============================================================
# FEATURE STATISTICS
# ============================================================

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

print("\n6. FEATURE STATISTICS")
print("-" * 60)

print(df[feature_columns].describe().round(4))


# ============================================================
# DUPLICATE IMAGE IDs
# ============================================================

print("\n7. DUPLICATE IMAGE IDs")
print("-" * 60)

duplicates = df["image_id"].duplicated().sum()

print("Duplicate image IDs:", duplicates)

if duplicates == 0:
    print("✓ No duplicate image IDs found.")
else:
    print("⚠ Duplicate image IDs found.")


# ============================================================
# INVALID VALUES
# ============================================================

print("\n8. INVALID VALUES")
print("-" * 60)

numeric_df = df[feature_columns]

infinite_values = numeric_df.isin([float("inf"), float("-inf")]).sum().sum()

print("Infinite values:", infinite_values)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)

print("\nDataset ready for next analysis if:")
print("✓ Rows = 1389")
print("✓ Columns = 17")
print("✓ No missing values")
print("✓ No duplicate image IDs")
print("✓ No infinite values")

print("\nDo NOT train the classifier yet.")
print("First review the results above.")