import os
import cv2
import numpy as np
from scipy.io import loadmat

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\HP\OneDrive\Desktop\NephroPredictorr"

IMAGE_SOURCE = os.path.join(
    BASE_DIR,
    "dataset",
    "kidney_images",
    "images"
)

MASK_SOURCE = os.path.join(
    BASE_DIR,
    "dataset",
    "kidney_images",
    "masks"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "segmentation"
)

OUTPUT_IMAGES = os.path.join(
    OUTPUT_DIR,
    "images"
)

OUTPUT_MASKS = os.path.join(
    OUTPUT_DIR,
    "masks"
)

# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (256, 256)

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_MASKS, exist_ok=True)

# ============================================================
# FIND ORIGINAL IMAGES
# ============================================================

image_files = [
    f for f in os.listdir(IMAGE_SOURCE)
    if f.lower().endswith(".jpg")
]

image_files.sort(
    key=lambda x: int(os.path.splitext(x)[0])
)

print("=" * 60)
print("U-NET SEGMENTATION DATA PREPARATION")
print("=" * 60)

print("\nImages found:", len(image_files))

processed = 0
failed = 0

# ============================================================
# PROCESS EACH IMAGE + MASK
# ============================================================

for image_file in image_files:

    image_id = os.path.splitext(image_file)[0]

    image_path = os.path.join(
        IMAGE_SOURCE,
        image_file
    )

    mask_path = os.path.join(
        MASK_SOURCE,
        image_id + ".mat"
    )

    # --------------------------------------------------------
    # CHECK MASK
    # --------------------------------------------------------

    if not os.path.exists(mask_path):

        print(f"Missing mask: {image_id}.mat")
        failed += 1
        continue

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image = cv2.imread(
        image_path,
        cv2.IMREAD_COLOR
    )

    if image is None:

        print(f"Unreadable image: {image_file}")
        failed += 1
        continue

    # --------------------------------------------------------
    # READ .MAT MASK
    # --------------------------------------------------------

    try:

        mat_data = loadmat(mask_path)

        mask = mat_data["mask"]

    except Exception as e:

        print(
            f"Error reading mask {image_id}: {e}"
        )

        failed += 1
        continue

    # --------------------------------------------------------
    # CONVERT MASK
    # --------------------------------------------------------

    mask = mask.astype(np.uint8)

    # Original:
    # 0 = background
    # 1 = kidney
    #
    # Convert to:
    # 0   = background
    # 255 = kidney

    mask = mask * 255

    # --------------------------------------------------------
    # RESIZE IMAGE
    # --------------------------------------------------------

    image_resized = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # RESIZE MASK
    # IMPORTANT:
    # Use NEAREST interpolation for masks.
    # --------------------------------------------------------

    mask_resized = cv2.resize(
        mask,
        IMAGE_SIZE,
        interpolation=cv2.INTER_NEAREST
    )

    # --------------------------------------------------------
    # SAVE IMAGE
    # --------------------------------------------------------

    output_image_path = os.path.join(
        OUTPUT_IMAGES,
        image_id + ".jpg"
    )

    cv2.imwrite(
        output_image_path,
        image_resized
    )

    # --------------------------------------------------------
    # SAVE MASK
    # --------------------------------------------------------

    output_mask_path = os.path.join(
        OUTPUT_MASKS,
        image_id + ".png"
    )

    cv2.imwrite(
        output_mask_path,
        mask_resized
    )

    processed += 1

    # --------------------------------------------------------
    # SHOW PROGRESS
    # --------------------------------------------------------

    if processed % 100 == 0:

        print(
            f"Processed: {processed}/{len(image_files)}"
        )

# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 60)
print("PREPARATION COMPLETE")
print("=" * 60)

print(
    "Images processed:",
    processed
)

print(
    "Failed:",
    failed
)

print(
    "\nOutput images:"
)

print(OUTPUT_IMAGES)

print(
    "\nOutput masks:"
)

print(OUTPUT_MASKS)

print("\nExpected images:", len(image_files))
print("Expected masks :", len(image_files))

# ============================================================
# SUCCESS CHECK
# ============================================================

if processed == len(image_files) and failed == 0:

    print("\n✅ ALL 1,985 IMAGE-MASK PAIRS PREPARED")

else:

    print("\n⚠️ SOME FILES FAILED — CHECK THE REPORT")