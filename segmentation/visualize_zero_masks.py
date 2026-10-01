import os
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from unet_model import UNet


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\HP\OneDrive\Desktop\NephroPredictorr"

SEG_DIR = os.path.join(
    BASE_DIR,
    "segmentation"
)

MODEL_PATH = os.path.join(
    SEG_DIR,
    "models",
    "best_unet.pth"
)

ZERO_MASK_CSV = os.path.join(
    SEG_DIR,
    "features",
    "zero_kidney_area_images.csv"
)

OUTPUT_DIR = os.path.join(
    SEG_DIR,
    "features",
    "zero_mask_visualization"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 256
THRESHOLD = 0.5


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("ZERO-MASK VISUAL INSPECTION")
print("=" * 60)

print("\nDevice:", device)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"U-Net model not found:\n{MODEL_PATH}"
    )

if not os.path.exists(ZERO_MASK_CSV):
    raise FileNotFoundError(
        f"Zero-mask CSV not found:\n{ZERO_MASK_CSV}"
    )


# ============================================================
# LOAD U-NET
# ============================================================

print("\nLoading U-Net...")

model = UNet().to(device)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    state_dict
)

model.eval()

print("✓ U-Net loaded.")


# ============================================================
# LOAD ZERO-MASK CSV
# ============================================================

df = pd.read_csv(
    ZERO_MASK_CSV
)

print(
    "\nZero-mask images:",
    len(df)
)


# ============================================================
# PROCESS EACH IMAGE
# ============================================================

for index, row in df.iterrows():

    # --------------------------------------------------------
    # IMAGE PATH
    # --------------------------------------------------------

    relative_path = str(
        row["image_path"]
    )

    image_path = os.path.join(
        BASE_DIR,
        "dataset",
        relative_path
    )

    image_id = str(
        row["image_id"]
    )

    class_name = str(
        row["class_name"]
    )

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image = cv2.imread(
        image_path,
        cv2.IMREAD_COLOR
    )

    if image is None:

        print(
            f"[{index + 1}/{len(df)}] "
            f"Could not read: {image_path}"
        )

        continue

    # --------------------------------------------------------
    # ORIGINAL IMAGE FOR DISPLAY
    # --------------------------------------------------------

    original_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # RESIZE IMAGE TO 256 × 256
    #
    # This prevents U-Net skip-connection size errors
    # such as 560 vs 561.
    # --------------------------------------------------------

    resized_image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # BGR → RGB
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        resized_image,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    input_image = (
        rgb.astype(
            np.float32
        ) / 255.0
    )

    # --------------------------------------------------------
    # HWC → CHW
    # --------------------------------------------------------

    input_image = np.transpose(
        input_image,
        (2, 0, 1)
    )

    # --------------------------------------------------------
    # NUMPY → TENSOR
    # --------------------------------------------------------

    tensor = torch.tensor(
        input_image,
        dtype=torch.float32
    )

    # Add batch dimension
    tensor = tensor.unsqueeze(
        0
    ).to(device)

    # --------------------------------------------------------
    # U-NET PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            tensor
        )

        probability = torch.sigmoid(
            output
        )

        prediction = (
            probability > THRESHOLD
        ).float()

    # --------------------------------------------------------
    # CONVERT MASK
    # --------------------------------------------------------

    predicted_mask = (
        prediction
        .squeeze()
        .cpu()
        .numpy()
        * 255
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # KIDNEY PIXEL COUNT
    # --------------------------------------------------------

    mask_pixels = cv2.countNonZero(
        predicted_mask
    )

    # --------------------------------------------------------
    # CREATE MASK DISPLAY
    # --------------------------------------------------------

    mask_rgb = cv2.cvtColor(
        predicted_mask,
        cv2.COLOR_GRAY2RGB
    )

    # --------------------------------------------------------
    # CREATE OVERLAY
    # --------------------------------------------------------

    overlay = rgb.copy()

    # Red overlay on predicted kidney area
    overlay[predicted_mask > 127] = [
        255,
        0,
        0
    ]

    overlay = cv2.addWeighted(
        rgb,
        0.70,
        overlay,
        0.30,
        0
    )

    # --------------------------------------------------------
    # CREATE FIGURE
    # --------------------------------------------------------

    plt.figure(
        figsize=(15, 5)
    )

    # ========================================================
    # ORIGINAL
    # ========================================================

    plt.subplot(
        1,
        3,
        1
    )

    plt.imshow(
        original_rgb
    )

    plt.title(
        f"Original Image\n"
        f"ID: {image_id}"
    )

    plt.axis(
        "off"
    )

    # ========================================================
    # PREDICTED MASK
    # ========================================================

    plt.subplot(
        1,
        3,
        2
    )

    plt.imshow(
        predicted_mask,
        cmap="gray"
    )

    plt.title(
        f"Predicted Mask\n"
        f"Pixels: {mask_pixels}"
    )

    plt.axis(
        "off"
    )

    # ========================================================
    # OVERLAY
    # ========================================================

    plt.subplot(
        1,
        3,
        3
    )

    plt.imshow(
        overlay
    )

    plt.title(
        f"Overlay\n"
        f"{class_name}"
    )

    plt.axis(
        "off"
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_filename = (
        f"image_{image_id}_{class_name}.png"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_filename
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[{index + 1}/{len(df)}] "
        f"Saved: {output_filename}"
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print("VISUAL INSPECTION COMPLETE")
print("=" * 60)

print("\nOutput folder:")
print(OUTPUT_DIR)

print(
    "\nOpen the generated images and inspect "
    "the Original, Predicted Mask and Overlay."
)

print(
    "\nDo NOT delete or modify the original dataset."
)