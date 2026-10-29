
import os
import cv2
import torch
import numpy as np
import pandas as pd

from unet_model import UNet


# --------------------------------------------------
# 1. Set project paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_unet.pth")
TEST_CSV = os.path.join(PROJECT_DIR, "dataset", "kidney_images", "test.csv")

IMAGE_DIR = os.path.join(BASE_DIR, "images")
MASK_DIR = os.path.join(BASE_DIR, "masks")

RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(
    RESULTS_DIR,
    "unet_failure_report.csv"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5


# --------------------------------------------------
# 2. Load the existing model
# --------------------------------------------------

print("\nLoading U-Net model...")

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
# 3. Load test dataset
# --------------------------------------------------

print("\nLoading test dataset...")

df = pd.read_csv(TEST_CSV)

print("Test images listed:", len(df))


# --------------------------------------------------
# 4. Evaluate every test image
# --------------------------------------------------

results = []

with torch.no_grad():

    for index, row in df.iterrows():

        image_id = str(row["image_id"])

        # Handle IDs such as 967.0 if present in the CSV
        if image_id.endswith(".0"):
            image_id = image_id[:-2]

        image_path = os.path.join(
            IMAGE_DIR,
            image_id + ".jpg"
        )

        mask_path = os.path.join(
            MASK_DIR,
            image_id + ".png"
        )

        if not os.path.exists(image_path):
            print("Missing image:", image_id)
            continue

        if not os.path.exists(mask_path):
            print("Missing mask:", image_id)
            continue

        # Read ultrasound image and convert BGR to RGB
        image = cv2.imread(image_path)

        if image is None:
            print("Could not read image:", image_id)
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        # Convert HWC to CHW
        image = np.transpose(image, (2, 0, 1))

        image_tensor = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

        # Read ground-truth mask
        ground_truth = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if ground_truth is None:
            print("Could not read mask:", image_id)
            continue

        ground_truth = (ground_truth > 127).astype(np.uint8)

        # Predict segmentation mask
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)

        predicted = (
            probabilities > THRESHOLD
        ).float().squeeze().cpu().numpy().astype(np.uint8)

        # Calculate pixel counts
        gt_pixels = int(np.sum(ground_truth))
        predicted_pixels = int(np.sum(predicted))

        # Calculate Dice and IoU
        intersection = int(
            np.sum(predicted * ground_truth)
        )

        union = int(
            np.sum((predicted + ground_truth) > 0)
        )

        if predicted_pixels + gt_pixels == 0:
            dice = 1.0
        else:
            dice = (
                2.0 * intersection
                / (predicted_pixels + gt_pixels)
            )

        if union == 0:
            iou = 1.0
        else:
            iou = intersection / union

        # Probability information helps investigate failures
        probability_array = probabilities.squeeze().cpu().numpy()

        results.append({
            "image_id": image_id,
            "ground_truth_pixels": gt_pixels,
            "predicted_pixels": predicted_pixels,
            "dice_score": round(dice, 4),
            "iou_score": round(iou, 4),
            "empty_prediction": predicted_pixels == 0,
            "max_probability": round(
                float(np.max(probability_array)), 6
            ),
            "mean_probability": round(
                float(np.mean(probability_array)), 6
            )
        })

        if (index + 1) % 25 == 0:
            print(
                f"Processed {index + 1}/{len(df)} test entries..."
            )


# --------------------------------------------------
# 5. Save report, worst-performing images first
# --------------------------------------------------

if not results:
    print("\nNo images were evaluated.")
    print("Check the test CSV and image/mask paths.")
    raise SystemExit(1)

report = pd.DataFrame(results)

report = report.sort_values(
    by="dice_score",
    ascending=True
)

report.to_csv(OUTPUT_CSV, index=False)

print("\n========================================")
print("U-NET FAILURE ANALYSIS COMPLETE")
print("========================================")

print("Images evaluated:", len(report))
print("Mean Dice:", round(report["dice_score"].mean(), 4))
print("Mean IoU:", round(report["iou_score"].mean(), 4))

print(
    "Empty predictions:",
    int(report["empty_prediction"].sum())
)

print("\n10 worst-performing images:")
print(
    report[
        [
            "image_id",
            "ground_truth_pixels",
            "predicted_pixels",
            "dice_score",
            "iou_score",
            "empty_prediction"
        ]
    ].head(10).to_string(index=False)
)

print("\nFull report saved to:")
print(OUTPUT_CSV)

print("\nNo model weights were modified.")

