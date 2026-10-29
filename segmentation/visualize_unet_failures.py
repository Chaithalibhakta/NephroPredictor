
import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from unet_model import UNet


# 1. Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_unet.pth")
REPORT_PATH = os.path.join(
    BASE_DIR, "results", "unet_failure_report.csv"
)

IMAGE_DIR = os.path.join(BASE_DIR, "images")
MASK_DIR = os.path.join(BASE_DIR, "masks")

OUTPUT_DIR = os.path.join(
    BASE_DIR, "results", "failure_visualizations"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

THRESHOLD = 0.5


# 2. Load the existing model
print("Loading U-Net model...")

model = UNet().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint)
model.eval()

print("Model loaded successfully.")


# 3. Load the failure report
report = pd.read_csv(REPORT_PATH)

# Select the 10 lowest-Dice images
worst_images = report.sort_values(
    "dice_score", ascending=True
).head(10)

print("\nImages selected for visualization:")
print(worst_images[["image_id", "dice_score", "iou_score"]].to_string(index=False))


# 4. Generate visualizations
for _, row in worst_images.iterrows():

    image_id = str(row["image_id"])

    if image_id.endswith(".0"):
        image_id = image_id[:-2]

    image_path = os.path.join(
        IMAGE_DIR, image_id + ".jpg"
    )

    mask_path = os.path.join(
        MASK_DIR, image_id + ".png"
    )

    if not os.path.exists(image_path):
        print("Missing image:", image_id)
        continue

    if not os.path.exists(mask_path):
        print("Missing ground-truth mask:", image_id)
        continue

    # Read original image
    original_bgr = cv2.imread(image_path)

    if original_bgr is None:
        print("Could not read image:", image_id)
        continue

    original_rgb = cv2.cvtColor(
        original_bgr, cv2.COLOR_BGR2RGB
    )

    # Prepare input using the same preprocessing
    image = original_rgb.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))

    image_tensor = torch.from_numpy(
        image
    ).unsqueeze(0).to(DEVICE)

    # Read ground truth
    ground_truth = cv2.imread(
        mask_path, cv2.IMREAD_GRAYSCALE
    )

    if ground_truth is None:
        print("Could not read mask:", image_id)
        continue

    ground_truth = (ground_truth > 127).astype(np.uint8)

    # Predict mask
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)

        predicted_mask = (
            probabilities > THRESHOLD
        ).float().squeeze().cpu().numpy().astype(np.uint8)

    # Calculate overlap
    intersection = int(
        np.sum(predicted_mask * ground_truth)
    )

    gt_pixels = int(np.sum(ground_truth))
    predicted_pixels = int(np.sum(predicted_mask))

    union = int(
        np.sum((predicted_mask + ground_truth) > 0)
    )

    dice = (
        1.0 if gt_pixels + predicted_pixels == 0
        else 2.0 * intersection / (gt_pixels + predicted_pixels)
    )

    iou = 1.0 if union == 0 else intersection / union

    # Display and save
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Ultrasound")

    axes[1].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground-Truth Mask")

    axes[2].imshow(predicted_mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Predicted Mask")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"Image {image_id} | Dice: {dice:.4f} | "
        f"IoU: {iou:.4f} | "
        f"GT pixels: {gt_pixels} | Predicted pixels: {predicted_pixels}"
    )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR, f"failure_{image_id}.png"
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(
        f"Saved image {image_id}: "
        f"Dice={dice:.4f}, IoU={iou:.4f}"
    )


print("\n========================================")
print("FAILURE VISUALIZATION COMPLETE")
print("========================================")
print("Output folder:")
print(OUTPUT_DIR)
print("\nNo model weights were modified.")

