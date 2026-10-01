import pandas as pd
import numpy as np
import cv2
import scipy.io as sio
import os

# ==========================================
# PATHS
# ==========================================

CSV_PATH = "kidney_images/kidney_dataset.csv"

IMAGE_DIR = "kidney_images/images"
MASK_DIR = "kidney_images/masks"

# ==========================================
# LOAD CSV
# ==========================================

df = pd.read_csv(CSV_PATH)

print("==========================================")
print("NEPHROPREDICTOR DATASET ANALYSIS")
print("==========================================")

print(f"\nTotal samples: {len(df)}")

# ==========================================
# CLASS DISTRIBUTION
# ==========================================

print("\nCLASS DISTRIBUTION")
print("------------------------------------------")

class_counts = df["class_name"].value_counts()

print(class_counts)

print("\nClass percentages:")

class_percentages = (
    df["class_name"]
    .value_counts(normalize=True)
    * 100
)

print(class_percentages.round(2))

# ==========================================
# IMAGE DIMENSIONS
# ==========================================

widths = []
heights = []

print("\nReading image dimensions...")

for image_file in os.listdir(IMAGE_DIR):

    if not image_file.lower().endswith(".jpg"):
        continue

    image_path = os.path.join(
        IMAGE_DIR,
        image_file
    )

    image = cv2.imread(image_path)

    if image is not None:

        height, width = image.shape[:2]

        heights.append(height)
        widths.append(width)

print("\nIMAGE DIMENSIONS")
print("------------------------------------------")

print("Unique heights:", sorted(set(heights)))
print("Unique widths :", sorted(set(widths)))

print("Minimum height:", min(heights))
print("Maximum height:", max(heights))

print("Minimum width :", min(widths))
print("Maximum width :", max(widths))

# ==========================================
# MASK COVERAGE
# ==========================================

mask_coverages = []

print("\nCalculating mask coverage...")

for mask_file in os.listdir(MASK_DIR):

    if not mask_file.lower().endswith(".mat"):
        continue

    mask_path = os.path.join(
        MASK_DIR,
        mask_file
    )

    try:

        mat = sio.loadmat(mask_path)

        mask = mat["mask"]

        kidney_pixels = np.sum(mask == 1)

        total_pixels = mask.size

        coverage = (
            kidney_pixels / total_pixels
        ) * 100

        mask_coverages.append(coverage)

    except Exception:
        pass

print("\nMASK COVERAGE")
print("------------------------------------------")

print(
    f"Minimum: {min(mask_coverages):.2f}%"
)

print(
    f"Maximum: {max(mask_coverages):.2f}%"
)

print(
    f"Mean: {np.mean(mask_coverages):.2f}%"
)

print(
    f"Median: {np.median(mask_coverages):.2f}%"
)

# ==========================================
# BOUNDING BOX DISTRIBUTION
# ==========================================

print("\nBOUNDING BOX DISTRIBUTION")
print("------------------------------------------")

print(
    df["num_bounding_boxes"]
    .value_counts()
    .sort_index()
)

print(
    "\nImages containing bounding boxes:",
    (df["num_bounding_boxes"] > 0).sum()
)

print(
    "Images without bounding boxes:",
    (df["num_bounding_boxes"] == 0).sum()
)

# ==========================================
# PATHOLOGY DISTRIBUTION
# ==========================================

print("\nGLOBAL PATHOLOGY DISTRIBUTION")
print("------------------------------------------")

print(
    "Pathology 1:"
)

print(
    df["pathology_1"]
    .value_counts(dropna=False)
    .sort_index()
)

print(
    "\nPathology 2:"
)

print(
    df["pathology_2"]
    .value_counts(dropna=False)
    .sort_index()
)

# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n==========================================")
print("ANALYSIS COMPLETE")
print("==========================================")