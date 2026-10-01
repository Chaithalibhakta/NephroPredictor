import os
import random
import cv2
import numpy as np
import pandas as pd
import scipy.io as sio

# ============================================================
# SETTINGS
# ============================================================

NUM_SYNTHETIC = 20
IMAGE_SIZE = 224

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# PATHS
# ============================================================

TRAIN_CSV = "kidney_images/train.csv"

IMAGE_DIR = "kidney_images/images"
MASK_DIR = "kidney_images/masks"

OUTPUT_DIR = "kidney_images/synthetic_test"

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

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

# ============================================================
# LOAD TRAINING DATA
# ============================================================

df = pd.read_csv(TRAIN_CSV)

print("Training images available:", len(df))

# ============================================================
# AUGMENTATION FUNCTION
# ============================================================

def augment_image_and_mask(image, mask):

    # --------------------------------------------------------
    # Resize first
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
    # Random rotation
    # --------------------------------------------------------

    angle = random.uniform(-15, 15)

    center = (
        IMAGE_SIZE // 2,
        IMAGE_SIZE // 2
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    image = cv2.warpAffine(
        image,
        matrix,
        (IMAGE_SIZE, IMAGE_SIZE),
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
    # Random horizontal flip
    # --------------------------------------------------------

    if random.random() < 0.5:

        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    # --------------------------------------------------------
    # Random zoom
    # --------------------------------------------------------

    scale = random.uniform(0.90, 1.10)

    new_size = int(IMAGE_SIZE * scale)

    image_zoom = cv2.resize(
        image,
        (new_size, new_size),
        interpolation=cv2.INTER_AREA
    )

    mask_zoom = cv2.resize(
        mask,
        (new_size, new_size),
        interpolation=cv2.INTER_NEAREST
    )

    if scale >= 1:

        start = (new_size - IMAGE_SIZE) // 2

        image = image_zoom[
            start:start + IMAGE_SIZE,
            start:start + IMAGE_SIZE
        ]

        mask = mask_zoom[
            start:start + IMAGE_SIZE,
            start:start + IMAGE_SIZE
        ]

    else:

        image = cv2.copyMakeBorder(
            image_zoom,
            (IMAGE_SIZE - new_size) // 2,
            (IMAGE_SIZE - new_size + 1) // 2,
            (IMAGE_SIZE - new_size) // 2,
            (IMAGE_SIZE - new_size + 1) // 2,
            cv2.BORDER_REFLECT
        )

        mask = cv2.copyMakeBorder(
            mask_zoom,
            (IMAGE_SIZE - new_size) // 2,
            (IMAGE_SIZE - new_size + 1) // 2,
            (IMAGE_SIZE - new_size) // 2,
            (IMAGE_SIZE - new_size + 1) // 2,
            cv2.BORDER_CONSTANT,
            value=0
        )

    # --------------------------------------------------------
    # Brightness
    # --------------------------------------------------------

    brightness = random.uniform(-15, 15)

    image = image.astype(np.float32) + brightness

    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    contrast = random.uniform(0.90, 1.10)

    image = (image - 127.5) * contrast + 127.5

    # --------------------------------------------------------
    # Gamma
    # --------------------------------------------------------

    gamma = random.uniform(0.90, 1.10)

    image = np.clip(image, 0, 255).astype(np.uint8)

    lookup = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in range(256)
    ]).astype(np.uint8)

    image = cv2.LUT(image, lookup)

    # --------------------------------------------------------
    # Ultrasound-like noise
    # --------------------------------------------------------

    noise_strength = random.uniform(0.0, 8.0)

    noise = np.random.normal(
        0,
        noise_strength,
        image.shape
    )

    image = image.astype(np.float32) + noise

    image = np.clip(
        image,
        0,
        255
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Slight Gaussian blur
    # --------------------------------------------------------

    if random.random() < 0.25:

        image = cv2.GaussianBlur(
            image,
            (3, 3),
            0
        )

    # --------------------------------------------------------
    # Ensure mask is binary
    # --------------------------------------------------------

    mask = (mask > 0).astype(np.uint8)

    return image, mask


# ============================================================
# GENERATE TEST DATA
# ============================================================

records = []

print("\nGenerating", NUM_SYNTHETIC, "synthetic images...")
print("=" * 60)

for i in range(NUM_SYNTHETIC):

    # Select random training sample
    row = df.sample(
        n=1,
        random_state=SEED + i
    ).iloc[0]

    image_id = str(row["image_id"])

    image_path = os.path.join(
        IMAGE_DIR,
        f"{image_id}.jpg"
    )

    mask_path = os.path.join(
        MASK_DIR,
        f"{image_id}.mat"
    )

    # Load image
    image = cv2.imread(image_path)

    if image is None:
        print("Skipping unreadable image:", image_id)
        continue

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Load mask
    mat = sio.loadmat(mask_path)

    mask = mat["mask"]

    # Augment
    synthetic_image, synthetic_mask = augment_image_and_mask(
        image,
        mask
    )

    synthetic_id = f"synthetic_{i + 1:05d}"

    # Save image
    image_output = os.path.join(
        OUTPUT_IMAGE_DIR,
        synthetic_id + ".jpg"
    )

    # Convert RGB → BGR for OpenCV
    image_bgr = cv2.cvtColor(
        synthetic_image,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        image_output,
        image_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    # Save mask
    mask_output = os.path.join(
        OUTPUT_MASK_DIR,
        synthetic_id + ".png"
    )

    cv2.imwrite(
        mask_output,
        synthetic_mask * 255
    )

    # Save metadata
    records.append({
        "synthetic_id": synthetic_id,
        "source_id": image_id,
        "diagnosis": int(row["diagnosis"]),
        "class_name": row["class_name"]
    })

    print(
        f"{i + 1:02d}/{NUM_SYNTHETIC} "
        f"source={image_id} "
        f"class={row['class_name']}"
    )

# ============================================================
# SAVE CSV
# ============================================================

result_df = pd.DataFrame(records)

result_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\n" + "=" * 60)
print("SYNTHETIC TEST GENERATION COMPLETE")
print("=" * 60)

print("Generated:", len(result_df))

print("\nClass distribution:")
print(result_df["class_name"].value_counts())

print("\nOutput:")
print(OUTPUT_DIR)