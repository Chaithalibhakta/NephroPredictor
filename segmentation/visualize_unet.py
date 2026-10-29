
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

SEGMENTATION_DIR = os.path.join(BASE_DIR, "segmentation")
IMAGE_DIR = os.path.join(SEGMENTATION_DIR, "images")
MASK_DIR = os.path.join(SEGMENTATION_DIR, "masks")

TEST_CSV = os.path.join(
    BASE_DIR, "dataset", "kidney_images", "test.csv"
)

MODEL_PATH = os.path.join(
    SEGMENTATION_DIR, "models", "best_unet.pth"
)

RESULT_DIR = os.path.join(
    SEGMENTATION_DIR, "results", "visualizations"
)

os.makedirs(RESULT_DIR, exist_ok=True)

NUM_IMAGES = 10
EPSILON = 1e-7

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("U-NET SEGMENTATION VISUALIZATION WITH DICE AND IOU")
print("=" * 60)
print("\nDevice:", device)

# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_df = pd.read_csv(TEST_CSV)
print("Test samples:", len(test_df))

# ============================================================
# LOAD MODEL
# ============================================================

print("\nCreating U-Net...")

model = UNet().to(device)

print("\nLoading trained weights...")

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

print("Trained U-Net loaded successfully.")

# ============================================================
# GENERATE VISUALIZATIONS
# ============================================================

print("\nGenerating visualizations...")

dice_scores = []
iou_scores = []

num_images = min(NUM_IMAGES, len(test_df))

for i in range(num_images):

    row = test_df.iloc[i]

    image_name = os.path.basename(str(row["image_path"]))
    image_id = os.path.splitext(image_name)[0]

    image_path = os.path.join(IMAGE_DIR, image_id + ".jpg")
    mask_path = os.path.join(MASK_DIR, image_id + ".png")

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    ground_truth = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or ground_truth is None:
        print("Skipping image:", image_id)
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Prepare image
    input_image = image_rgb.astype(np.float32) / 255.0
    input_image = np.transpose(input_image, (2, 0, 1))

    input_tensor = torch.from_numpy(
        input_image
    ).float().unsqueeze(0).to(device)

    # Predict mask
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.sigmoid(output)
        prediction = (probabilities > 0.5).float()

    prediction = prediction.squeeze().cpu().numpy().astype(bool)

    # Convert ground truth to binary mask
    ground_truth_binary = ground_truth > 127

    # Ensure mask dimensions match prediction
    if ground_truth_binary.shape != prediction.shape:
        ground_truth_binary = cv2.resize(
            ground_truth_binary.astype(np.uint8),
            (prediction.shape[1], prediction.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    # Calculate per-image Dice and IoU
    intersection = np.logical_and(
        prediction, ground_truth_binary
    ).sum()

    predicted_pixels = prediction.sum()
    ground_truth_pixels = ground_truth_binary.sum()

    union = np.logical_or(
        prediction, ground_truth_binary
    ).sum()

    dice = (
        (2.0 * intersection + EPSILON)
        / (predicted_pixels + ground_truth_pixels + EPSILON)
    )

    iou = (
        (intersection + EPSILON)
        / (union + EPSILON)
    )

    dice_scores.append(dice)
    iou_scores.append(iou)

    # Display masks
    ground_truth_display = ground_truth_binary.astype(np.uint8) * 255
    prediction_display = prediction.astype(np.uint8) * 255

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title("Original Ultrasound")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth_display, cmap="gray", vmin=0, vmax=255)
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(prediction_display, cmap="gray", vmin=0, vmax=255)
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.suptitle(
        f"Image {image_id} | Dice: {dice:.4f} ({dice * 100:.2f}%)"
        f" | IoU: {iou:.4f} ({iou * 100:.2f}%)"
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULT_DIR, f"{image_id}_segmentation.png"
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(
        f"{i + 1}/{num_images} | Image {image_id}"
        f" | Dice: {dice:.4f}"
        f" | IoU: {iou:.4f}"
    )

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PER-IMAGE VISUALIZATION SUMMARY")
print("=" * 60)

if dice_scores:
    print("Images evaluated:", len(dice_scores))
    print(f"Mean Dice: {np.mean(dice_scores):.4f}")
    print(f"Mean Dice (%): {np.mean(dice_scores) * 100:.2f}%")
    print(f"Mean IoU: {np.mean(iou_scores):.4f}")
    print(f"Mean IoU (%): {np.mean(iou_scores) * 100:.2f}%")
else:
    print("No images were successfully evaluated.")

print("\nSaved visualizations to:")
print(RESULT_DIR)

