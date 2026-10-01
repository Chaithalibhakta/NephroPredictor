import os
import random
import cv2
import numpy as np
import pandas as pd
import scipy.io as sio

# ============================================================
# CONFIGURATION
# ============================================================

TOTAL_SYNTHETIC = 500_000

# Balanced dataset
HEALTHY_TARGET = 250_000
PATHOLOGICAL_TARGET = 250_000

IMAGE_SIZE = 224

# JPEG quality
JPEG_QUALITY = 90

SEED = 42

# ============================================================
# PATHS
# ============================================================

TRAIN_CSV = "kidney_images/train.csv"

IMAGE_DIR = "kidney_images/images"
MASK_DIR = "kidney_images/masks"

OUTPUT_DIR = r"D:\NephroPredictor_Synthetic\synthetic_500k"

OUTPUT_IMAGE_DIR = os.path.join(
    OUTPUT_DIR,
    "images"
)

OUTPUT_MASK_DIR = os.path.join(
    OUTPUT_DIR,
    "masks"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "labels.csv"
)

# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    OUTPUT_IMAGE_DIR,
    exist_ok=True
)

os.makedirs(
    OUTPUT_MASK_DIR,
    exist_ok=True
)

# ============================================================
# LOAD TRAINING CSV
# ============================================================

df = pd.read_csv(TRAIN_CSV)

print("==========================================")
print("500K SYNTHETIC DATASET GENERATOR")
print("==========================================")

print("\nTraining samples available:", len(df))

# ============================================================
# SEPARATE CLASSES
# ============================================================

healthy_df = df[
    df["diagnosis"] == 0
].reset_index(drop=True)

pathological_df = df[
    df["diagnosis"] == 1
].reset_index(drop=True)

print("\nTraining class counts:")
print("Healthy:", len(healthy_df))
print("Pathological:", len(pathological_df))

# ============================================================
# RANDOM GENERATOR
# ============================================================

rng = random.Random(SEED)

# ============================================================
# AUGMENTATION
# ============================================================

def augment_image_and_mask(image, mask, rng):

    # --------------------------------------------------------
    # Resize to 224 x 224
    # --------------------------------------------------------

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    mask = cv2.resize(
        mask,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_NEAREST
    )

    # --------------------------------------------------------
    # Rotation + scaling
    # Same transformation for image and mask
    # --------------------------------------------------------

    angle = rng.uniform(-15, 15)
    scale = rng.uniform(0.90, 1.10)

    center = (
        IMAGE_SIZE // 2,
        IMAGE_SIZE // 2
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        scale
    )

    image = cv2.warpAffine(
        image,
        matrix,
        (IMAGE_SIZE, IMAGE_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    mask = cv2.warpAffine(
        mask,
        matrix,
        (IMAGE_SIZE, IMAGE_SIZE),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # --------------------------------------------------------
    # Horizontal flip
    # --------------------------------------------------------

    if rng.random() < 0.5:

        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    # --------------------------------------------------------
    # Small translation
    # --------------------------------------------------------

    tx = rng.uniform(-0.05, 0.05) * IMAGE_SIZE
    ty = rng.uniform(-0.05, 0.05) * IMAGE_SIZE

    translation_matrix = np.float32([
        [1, 0, tx],
        [0, 1, ty]
    ])

    image = cv2.warpAffine(
        image,
        translation_matrix,
        (IMAGE_SIZE, IMAGE_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    mask = cv2.warpAffine(
        mask,
        translation_matrix,
        (IMAGE_SIZE, IMAGE_SIZE),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # --------------------------------------------------------
    # Brightness
    # --------------------------------------------------------

    brightness = rng.uniform(-15, 15)

    image = image.astype(np.float32) + brightness

    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    contrast = rng.uniform(0.90, 1.10)

    image = (
        image - 127.5
    ) * contrast + 127.5

    image = np.clip(
        image,
        0,
        255
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Gamma
    # --------------------------------------------------------

    gamma = rng.uniform(0.90, 1.10)

    lookup = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in range(256)
    ]).astype(np.uint8)

    image = cv2.LUT(
        image,
        lookup
    )

    # --------------------------------------------------------
    # Speckle-like multiplicative noise
    # --------------------------------------------------------

    if rng.random() < 0.70:

        noise_strength = rng.uniform(
            0.02,
            0.08
        )

        noise = np.random.default_rng(
            rng.randint(0, 2**32 - 1)
        ).normal(
            1.0,
            noise_strength,
            image.shape
        )

        image = (
            image.astype(np.float32) * noise
        )

        image = np.clip(
            image,
            0,
            255
        ).astype(np.uint8)

    # --------------------------------------------------------
    # Slight Gaussian blur
    # --------------------------------------------------------

    if rng.random() < 0.25:

        image = cv2.GaussianBlur(
            image,
            (3, 3),
            0
        )

    # --------------------------------------------------------
    # Final binary mask
    # --------------------------------------------------------

    mask = (
        mask > 0
    ).astype(np.uint8)

    return image, mask


# ============================================================
# LOAD EXISTING PROGRESS
# ============================================================

existing_records = []

if os.path.exists(OUTPUT_CSV):

    try:

        existing_df = pd.read_csv(
            OUTPUT_CSV
        )

        existing_records = existing_df.to_dict(
            "records"
        )

        print(
            "\nExisting records found:",
            len(existing_records)
        )

    except Exception:

        print(
            "\nCould not read existing CSV."
        )

# ============================================================
# CURRENT COUNTS
# ============================================================

healthy_done = sum(
    1
    for r in existing_records
    if int(r["diagnosis"]) == 0
)

pathological_done = sum(
    1
    for r in existing_records
    if int(r["diagnosis"]) == 1
)

print("\nAlready generated:")
print("Healthy:", healthy_done)
print("Pathological:", pathological_done)

# ============================================================
# GENERATION
# ============================================================

records = existing_records.copy()

next_id = len(records) + 1

print("\nTarget:")
print("Healthy:", HEALTHY_TARGET)
print("Pathological:", PATHOLOGICAL_TARGET)

print("\nStarting generation...")
print("=" * 60)

while (
    healthy_done < HEALTHY_TARGET
    or
    pathological_done < PATHOLOGICAL_TARGET
):

    # --------------------------------------------------------
    # Choose class that still needs samples
    # --------------------------------------------------------

    if healthy_done < HEALTHY_TARGET:

        if pathological_done < PATHOLOGICAL_TARGET:

            # Randomly alternate between classes
            if rng.random() < 0.5:
                target_class = 0
            else:
                target_class = 1

        else:
            target_class = 0

    else:
        target_class = 1

    # --------------------------------------------------------
    # Select source image
    # --------------------------------------------------------

    if target_class == 0:

        source_row = healthy_df.iloc[
            rng.randrange(len(healthy_df))
        ]

    else:

        source_row = pathological_df.iloc[
            rng.randrange(len(pathological_df))
        ]

    source_id = str(
        source_row["image_id"]
    )

    image_path = os.path.join(
        IMAGE_DIR,
        f"{source_id}.jpg"
    )

    mask_path = os.path.join(
        MASK_DIR,
        f"{source_id}.mat"
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            "\nWARNING: Could not read",
            image_path
        )

        continue

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Load mask
    # --------------------------------------------------------

    try:

        mat = sio.loadmat(
            mask_path
        )

        mask = mat["mask"]

    except Exception as e:

        print(
            "\nWARNING: Could not read mask",
            mask_path,
            e
        )

        continue

    # --------------------------------------------------------
    # Generate synthetic sample
    # --------------------------------------------------------

    synthetic_image, synthetic_mask = (
        augment_image_and_mask(
            image,
            mask,
            rng
        )
    )

    synthetic_id = (
        f"synthetic_{next_id:07d}"
    )

    # --------------------------------------------------------
    # Save image
    # --------------------------------------------------------

    output_image = os.path.join(
        OUTPUT_IMAGE_DIR,
        synthetic_id + ".jpg"
    )

    image_bgr = cv2.cvtColor(
        synthetic_image,
        cv2.COLOR_RGB2BGR
    )

    success = cv2.imwrite(
        output_image,
        image_bgr,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            JPEG_QUALITY
        ]
    )

    if not success:

        print(
            "\nWARNING: Could not save",
            output_image
        )

        continue

    # --------------------------------------------------------
    # Save mask
    # --------------------------------------------------------

    output_mask = os.path.join(
        OUTPUT_MASK_DIR,
        synthetic_id + ".png"
    )

    cv2.imwrite(
        output_mask,
        synthetic_mask * 255
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    records.append({
        "synthetic_id": synthetic_id,
        "source_id": source_id,
        "diagnosis": int(
            source_row["diagnosis"]
        ),
        "class_name": source_row["class_name"]
    })

    # --------------------------------------------------------
    # Update counter
    # --------------------------------------------------------

    if target_class == 0:
        healthy_done += 1
    else:
        pathological_done += 1

    next_id += 1

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    generated = (
        healthy_done +
        pathological_done
    )

    if generated % 1000 == 0:

        print(
            f"{generated:,}/{TOTAL_SYNTHETIC:,} "
            f"| Healthy: {healthy_done:,} "
            f"| Pathological: {pathological_done:,}"
        )

        # Save progress periodically
        pd.DataFrame(records).to_csv(
            OUTPUT_CSV,
            index=False
        )

# ============================================================
# FINAL CSV
# ============================================================

result_df = pd.DataFrame(
    records
)

result_df.to_csv(
    OUTPUT_CSV,
    index=False
)

# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 60)
print("500K SYNTHETIC DATASET COMPLETE")
print("=" * 60)

print(
    "Total generated:",
    len(result_df)
)

print(
    "\nHealthy:",
    (result_df["diagnosis"] == 0).sum()
)

print(
    "Pathological:",
    (result_df["diagnosis"] == 1).sum()
)

print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)

print("\nCSV:")
print(OUTPUT_CSV)

print("\nGeneration complete.")