
import os
import cv2
import torch
import numpy as np

from unet_model import UNet


# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "best_unet.pth"
)

IMAGE_DIR = os.path.join(BASE_DIR, "images")
MASK_DIR = os.path.join(BASE_DIR, "masks")

OUTPUT_DIR = os.path.join(
    BASE_DIR, "results", "threshold_inspection"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_IDS = ["967", "1771"]

THRESHOLDS = [0.5, 0.3, 0.2, 0.1]


# --------------------------------------------------
# 2. Load the existing model
# --------------------------------------------------

print("Loading existing U-Net model...")

model = UNet().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint)
model.eval()

print("Model loaded successfully.")
print("Device:", DEVICE)


# --------------------------------------------------
# 3. Inspect each image
# --------------------------------------------------

for image_id in IMAGE_IDS:

    print("\n" + "=" * 60)
    print("IMAGE ID:", image_id)
    print("=" * 60)

    image_path = os.path.join(
        IMAGE_DIR, image_id + ".jpg"
    )

    mask_path = os.path.join(
        MASK_DIR, image_id + ".png"
    )

    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        print("ERROR: Could not read image:", image_path)
        continue

    image_rgb = cv2.cvtColor(
        image_bgr, cv2.COLOR_BGR2RGB
    )

    gt_raw = cv2.imread(
        mask_path, cv2.IMREAD_GRAYSCALE
    )

    if gt_raw is None:
        print("ERROR: Could not read mask:", mask_path)
        continue

    ground_truth = (
        gt_raw > 127
    ).astype(np.uint8)

    # Same preprocessing used in evaluation
    image = image_rgb.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))

    tensor = torch.from_numpy(
        image
    ).unsqueeze(0).to(DEVICE)

    # Obtain probabilities once
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.sigmoid(logits)

    prob = probabilities.squeeze().cpu().numpy()

    print("Ground-truth pixels:", int(ground_truth.sum()))
    print("Maximum probability:", float(prob.max()))
    print("Mean probability:", float(prob.mean()))

    print("\nResults at different thresholds:")
    print(
        f"{'Threshold':<12}"
        f"{'Predicted':<14}"
        f"{'Intersection':<16}"
        f"{'Dice':<12}"
        f"{'IoU':<12}"
    )

    for threshold in THRESHOLDS:

        prediction = (
            prob > threshold
        ).astype(np.uint8)

        predicted_pixels = int(prediction.sum())

        intersection = int(
            np.sum(prediction * ground_truth)
        )

        union = int(
            np.sum((prediction + ground_truth) > 0)
        )

        denominator = predicted_pixels + int(
            ground_truth.sum()
        )

        dice = (
            1.0 if denominator == 0
            else 2.0 * intersection / denominator
        )

        iou = (
            1.0 if union == 0
            else intersection / union
        )

        print(
            f"{threshold:<12.2f}"
            f"{predicted_pixels:<14}"
            f"{intersection:<16}"
            f"{dice:<12.4f}"
            f"{iou:<12.4f}"
        )

    # Save a probability-map visualization
    # Bright pixels have higher predicted probabilities.
    probability_image = np.clip(
        prob * 255, 0, 255
    ).astype(np.uint8)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"image_{image_id}_probability.png"
    )

    cv2.imwrite(
        output_path,
        probability_image
    )

    print("\nProbability map saved to:")
    print(output_path)


print("\n" + "=" * 60)
print("THRESHOLD INSPECTION COMPLETE")
print("No model weights were modified.")
print("=" * 60)

