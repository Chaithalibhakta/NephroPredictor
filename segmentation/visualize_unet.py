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

SEGMENTATION_DIR = os.path.join(
    BASE_DIR,
    "segmentation"
)

IMAGE_DIR = os.path.join(
    SEGMENTATION_DIR,
    "images"
)

MASK_DIR = os.path.join(
    SEGMENTATION_DIR,
    "masks"
)

TEST_CSV = os.path.join(
    BASE_DIR,
    "dataset",
    "kidney_images",
    "test.csv"
)

MODEL_PATH = os.path.join(
    SEGMENTATION_DIR,
    "models",
    "best_unet.pth"
)

RESULT_DIR = os.path.join(
    SEGMENTATION_DIR,
    "results",
    "visualizations"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

NUM_IMAGES = 10


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("U-NET SEGMENTATION VISUALIZATION")
print("=" * 60)

print("\nDevice:", device)


# ============================================================
# LOAD TEST CSV
# ============================================================

print("\nLoading test dataset...")

test_df = pd.read_csv(TEST_CSV)

print(
    "Test samples:",
    len(test_df)
)


# ============================================================
# CREATE MODEL
# ============================================================

print("\nCreating U-Net...")

model = UNet().to(device)

print("U-Net created successfully.")


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\nLoading best model...")

state_dict = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(state_dict)

model.eval()

print("✓ Best U-Net model loaded successfully.")


# ============================================================
# VISUALIZATION
# ============================================================

print("\nGenerating visualizations...")

num_images = min(
    NUM_IMAGES,
    len(test_df)
)

for i in range(num_images):

    row = test_df.iloc[i]

    # --------------------------------------------------------
    # IMAGE ID
    # --------------------------------------------------------

    image_name = os.path.basename(
        str(row["image_path"])
    )

    image_id = os.path.splitext(
        image_name
    )[0]

    image_path = os.path.join(
        IMAGE_DIR,
        image_id + ".jpg"
    )

    mask_path = os.path.join(
        MASK_DIR,
        image_id + ".png"
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
            "Skipping image:",
            image_path
        )
        continue

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # READ GROUND TRUTH MASK
    # --------------------------------------------------------

    ground_truth = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    if ground_truth is None:
        print(
            "Skipping mask:",
            mask_path
        )
        continue

    # --------------------------------------------------------
    # PREPARE IMAGE FOR U-NET
    # --------------------------------------------------------

    input_image = image_rgb.astype(
        np.float32
    ) / 255.0

    input_image = np.transpose(
        input_image,
        (2, 0, 1)
    )

    input_tensor = torch.tensor(
        input_image,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            input_tensor
        )

        prediction = torch.sigmoid(
            output
        )

        prediction = (
            prediction > 0.5
        ).float()

    prediction = prediction.squeeze().cpu().numpy()

    # --------------------------------------------------------
    # CONVERT MASKS TO DISPLAY FORMAT
    # --------------------------------------------------------

    ground_truth_display = (
        ground_truth > 127
    ).astype(np.uint8) * 255

    prediction_display = (
        prediction > 0.5
    ).astype(np.uint8) * 255

    # --------------------------------------------------------
    # CREATE FIGURE
    # --------------------------------------------------------

    plt.figure(
        figsize=(15, 5)
    )

    plt.subplot(
        1,
        3,
        1
    )

    plt.imshow(
        image_rgb
    )

    plt.title(
        "Original Image"
    )

    plt.axis("off")

    plt.subplot(
        1,
        3,
        2
    )

    plt.imshow(
        ground_truth_display,
        cmap="gray"
    )

    plt.title(
        "Ground Truth Mask"
    )

    plt.axis("off")

    plt.subplot(
        1,
        3,
        3
    )

    plt.imshow(
        prediction_display,
        cmap="gray"
    )

    plt.title(
        "U-Net Prediction"
    )

    plt.axis("off")

    plt.suptitle(
        f"Kidney Segmentation - Image {image_id}"
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_path = os.path.join(
        RESULT_DIR,
        f"{image_id}_segmentation.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved {i + 1}/{num_images}: "
        f"{output_path}"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("VISUALIZATION COMPLETE")
print("=" * 60)

print(
    "\nResults saved in:"
)

print(
    RESULT_DIR
)