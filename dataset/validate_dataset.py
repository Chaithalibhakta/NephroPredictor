import os
import cv2
import numpy as np
import scipy.io as sio

# ==========================================
# PATHS
# ==========================================

IMAGE_DIR = "kidney_images/images"
MASK_DIR = "kidney_images/masks"

# ==========================================
# COUNTERS
# ==========================================

total = 0
valid = 0

missing_images = []
missing_masks = []
bad_images = []
bad_masks = []
empty_masks = []
invalid_mask_values = []
dimension_mismatch = []

# ==========================================
# GET IMAGE FILES
# ==========================================

image_files = sorted(
    [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(".jpg")],
    key=lambda x: int(os.path.splitext(x)[0])
)

print(f"Found {len(image_files)} images.")

# ==========================================
# CHECK EACH IMAGE + MASK
# ==========================================

for image_file in image_files:

    total += 1

    image_id = os.path.splitext(image_file)[0]

    image_path = os.path.join(
        IMAGE_DIR,
        image_file
    )

    mask_path = os.path.join(
        MASK_DIR,
        f"{image_id}.mat"
    )

    # --------------------------------------
    # CHECK IMAGE
    # --------------------------------------

    if not os.path.exists(image_path):

        missing_images.append(image_id)
        continue

    image = cv2.imread(image_path)

    if image is None:

        bad_images.append(image_id)
        continue

    # --------------------------------------
    # CHECK MASK FILE
    # --------------------------------------

    if not os.path.exists(mask_path):

        missing_masks.append(image_id)
        continue

    try:

        mat_data = sio.loadmat(mask_path)

    except Exception:

        bad_masks.append(image_id)
        continue

    # --------------------------------------
    # CHECK "mask" VARIABLE
    # --------------------------------------

    if "mask" not in mat_data:

        bad_masks.append(image_id)
        continue

    mask = mat_data["mask"]

    # --------------------------------------
    # CHECK DIMENSIONS
    # --------------------------------------

    image_height, image_width = image.shape[:2]

    mask_height, mask_width = mask.shape[:2]

    if (
        image_height != mask_height
        or
        image_width != mask_width
    ):

        dimension_mismatch.append(
            (
                image_id,
                (image_height, image_width),
                (mask_height, mask_width)
            )
        )

    # --------------------------------------
    # CHECK MASK VALUES
    # --------------------------------------

    unique_values = np.unique(mask)

    if not np.all(
        np.isin(unique_values, [0, 1])
    ):

        invalid_mask_values.append(
            (image_id, unique_values.tolist())
        )

    # --------------------------------------
    # CHECK EMPTY MASK
    # --------------------------------------

    if np.sum(mask) == 0:

        empty_masks.append(image_id)

    else:

        valid += 1


# ==========================================
# FINAL REPORT
# ==========================================

print("\n==========================================")
print("DATASET VALIDATION REPORT")
print("==========================================")

print(f"Total images checked : {total}")
print(f"Valid non-empty masks: {valid}")

print("\nMissing images:")
print(len(missing_images))

print("\nMissing masks:")
print(len(missing_masks))

print("\nUnreadable images:")
print(len(bad_images))

print("\nUnreadable masks:")
print(len(bad_masks))

print("\nDimension mismatches:")
print(len(dimension_mismatch))

print("\nInvalid mask values:")
print(len(invalid_mask_values))

print("\nEmpty masks:")
print(len(empty_masks))

# ==========================================
# DETAILS IF PROBLEMS EXIST
# ==========================================

if missing_images:
    print("\nMissing image IDs:")
    print(missing_images[:20])

if missing_masks:
    print("\nMissing mask IDs:")
    print(missing_masks[:20])

if dimension_mismatch:
    print("\nFirst dimension mismatches:")
    for item in dimension_mismatch[:10]:
        print(item)

if invalid_mask_values:
    print("\nFirst invalid mask values:")
    for item in invalid_mask_values[:10]:
        print(item)

if empty_masks:
    print("\nFirst empty masks:")
    print(empty_masks[:20])

print("\n==========================================")
print("VALIDATION COMPLETE")
print("==========================================")