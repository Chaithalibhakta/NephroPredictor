import os
import cv2
import pandas as pd
import numpy as np

# ============================================================
# PATHS
# ============================================================

DATASET_DIR = r"D:\NephroPredictor_Synthetic\synthetic_500k"

IMAGE_DIR = os.path.join(
    DATASET_DIR,
    "images"
)

MASK_DIR = os.path.join(
    DATASET_DIR,
    "masks"
)

CSV_PATH = os.path.join(
    DATASET_DIR,
    "labels.csv"
)

# ============================================================
# EXPECTED VALUES
# ============================================================

EXPECTED_TOTAL = 500_000
EXPECTED_HEALTHY = 250_000
EXPECTED_PATHOLOGICAL = 250_000

EXPECTED_SIZE = (224, 224)

# ============================================================
# START
# ============================================================

print("=" * 60)
print("NEPHROPREDICTOR 500K DATASET VALIDATION")
print("=" * 60)

# ============================================================
# CHECK DIRECTORIES
# ============================================================

if not os.path.exists(IMAGE_DIR):
    print("ERROR: Images directory not found!")
    print(IMAGE_DIR)
    exit()

if not os.path.exists(MASK_DIR):
    print("ERROR: Masks directory not found!")
    print(MASK_DIR)
    exit()

if not os.path.exists(CSV_PATH):
    print("ERROR: labels.csv not found!")
    print(CSV_PATH)
    exit()

print("\nDirectories found: OK")

# ============================================================
# COUNT FILES
# ============================================================

print("\nCounting files...")

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(".jpg")
]

mask_files = [
    f for f in os.listdir(MASK_DIR)
    if f.lower().endswith(".png")
]

print("\nFILE COUNTS")
print("-" * 40)

print("Images :", len(image_files))
print("Masks  :", len(mask_files))

# ============================================================
# CSV
# ============================================================

print("\nReading labels.csv...")

df = pd.read_csv(CSV_PATH)

print("CSV records:", len(df))

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nCLASS DISTRIBUTION")
print("-" * 40)

print(
    df["class_name"].value_counts()
)

healthy_count = (
    df["diagnosis"] == 0
).sum()

pathological_count = (
    df["diagnosis"] == 1
).sum()

print("\nHealthy:", healthy_count)
print("Pathological:", pathological_count)

# ============================================================
# CHECK DUPLICATE IDs
# ============================================================

print("\nCHECKING DUPLICATE IDs...")

duplicate_ids = df[
    df["synthetic_id"].duplicated()
]

print(
    "Duplicate IDs:",
    len(duplicate_ids)
)

# ============================================================
# CHECK MISSING IMAGE/MASK FILES
# ============================================================

print("\nCHECKING IMAGE/MASK PAIRS...")

image_set = {
    os.path.splitext(f)[0]
    for f in image_files
}

mask_set = {
    os.path.splitext(f)[0]
    for f in mask_files
}

csv_set = set(
    df["synthetic_id"].astype(str)
)

missing_images = csv_set - image_set
missing_masks = csv_set - mask_set

extra_images = image_set - csv_set
extra_masks = mask_set - csv_set

print(
    "Missing images:",
    len(missing_images)
)

print(
    "Missing masks:",
    len(missing_masks)
)

print(
    "Extra images:",
    len(extra_images)
)

print(
    "Extra masks:",
    len(extra_masks)
)

# ============================================================
# SAMPLE IMAGE VALIDATION
# ============================================================

print("\nVALIDATING SAMPLE IMAGES/MASKS...")

# We don't need to open all 500K files.
# Check a representative sample.

sample_count = min(
    1000,
    len(image_files)
)

sample_files = image_files[:sample_count]

unreadable_images = 0
unreadable_masks = 0
wrong_dimensions = 0
invalid_masks = 0
empty_masks = 0

for i, image_file in enumerate(sample_files):

    image_path = os.path.join(
        IMAGE_DIR,
        image_file
    )

    mask_name = (
        os.path.splitext(image_file)[0]
        + ".png"
    )

    mask_path = os.path.join(
        MASK_DIR,
        mask_name
    )

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    image = cv2.imread(
        image_path
    )

    if image is None:

        unreadable_images += 1
        continue

    height, width = image.shape[:2]

    if (
        width,
        height
    ) != EXPECTED_SIZE:

        wrong_dimensions += 1

    # --------------------------------------------------------
    # MASK
    # --------------------------------------------------------

    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:

        unreadable_masks += 1
        continue

    unique_values = np.unique(mask)

    # Valid mask should contain
    # 0 and/or 255

    if not np.all(
        np.isin(
            unique_values,
            [0, 255]
        )
    ):

        invalid_masks += 1

    if np.count_nonzero(mask) == 0:

        empty_masks += 1

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (i + 1) % 100 == 0:

        print(
            f"Checked {i + 1}/{sample_count}"
        )

# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 60)
print("500K DATASET VALIDATION REPORT")
print("=" * 60)

print("\nEXPECTED")
print("-" * 40)

print(
    "Expected images:",
    EXPECTED_TOTAL
)

print(
    "Expected masks:",
    EXPECTED_TOTAL
)

print(
    "Expected CSV records:",
    EXPECTED_TOTAL
)

print(
    "Expected healthy:",
    EXPECTED_HEALTHY
)

print(
    "Expected pathological:",
    EXPECTED_PATHOLOGICAL
)

print("\nACTUAL")
print("-" * 40)

print(
    "Actual images:",
    len(image_files)
)

print(
    "Actual masks:",
    len(mask_files)
)

print(
    "Actual CSV records:",
    len(df)
)

print(
    "Actual healthy:",
    healthy_count
)

print(
    "Actual pathological:",
    pathological_count
)

print("\nERRORS")
print("-" * 40)

print(
    "Duplicate IDs:",
    len(duplicate_ids)
)

print(
    "Missing images:",
    len(missing_images)
)

print(
    "Missing masks:",
    len(missing_masks)
)

print(
    "Extra images:",
    len(extra_images)
)

print(
    "Extra masks:",
    len(extra_masks)
)

print(
    "Unreadable sample images:",
    unreadable_images
)

print(
    "Unreadable sample masks:",
    unreadable_masks
)

print(
    "Wrong dimensions in sample:",
    wrong_dimensions
)

print(
    "Invalid mask values in sample:",
    invalid_masks
)

print(
    "Empty masks in sample:",
    empty_masks
)

# ============================================================
# OVERALL RESULT
# ============================================================

success = True

if len(image_files) != EXPECTED_TOTAL:
    success = False

if len(mask_files) != EXPECTED_TOTAL:
    success = False

if len(df) != EXPECTED_TOTAL:
    success = False

if healthy_count != EXPECTED_HEALTHY:
    success = False

if pathological_count != EXPECTED_PATHOLOGICAL:
    success = False

if len(duplicate_ids) != 0:
    success = False

if len(missing_images) != 0:
    success = False

if len(missing_masks) != 0:
    success = False

if len(extra_images) != 0:
    success = False

if len(extra_masks) != 0:
    success = False

if unreadable_images != 0:
    success = False

if unreadable_masks != 0:
    success = False

if wrong_dimensions != 0:
    success = False

if invalid_masks != 0:
    success = False

if empty_masks != 0:
    success = False

print("\n")

if success:

    print("=" * 60)
    print("✅ DATASET VALIDATION PASSED")
    print("=" * 60)

else:

    print("=" * 60)
    print("⚠️ DATASET VALIDATION FOUND PROBLEMS")
    print("=" * 60)

print("\nValidation complete.")