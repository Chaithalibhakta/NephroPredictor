
import os
import cv2
import torch
import numpy as np
import pandas as pd

from unet_model import UNet

# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    "best_unet.pth": os.path.join(
        BASE_DIR, "models", "best_unet.pth"
    ),
    "best_unet_test.pth": os.path.join(
        BASE_DIR, "models", "best_unet_test.pth"
    ),
}

IMAGE_DIR = os.path.join(BASE_DIR, "images")
MASK_DIR = os.path.join(BASE_DIR, "masks")

OUTPUT_DIR = os.path.join(
    BASE_DIR, "results", "checkpoint_failure_comparison"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_IDS = ["967", "1771", "1721", "1784", "1342"]
THRESHOLD = 0.5

# --------------------------------------------------
# 2. Load both checkpoints
# --------------------------------------------------

models = {}

for model_name, model_path in MODEL_PATHS.items():
    print(f"\nLoading {model_name}...")

    model = UNet().to(DEVICE)

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model.eval()

    models[model_name] = model
    print("Loaded successfully.")

print("\nDevice:", DEVICE)

# --------------------------------------------------
# 3. Evaluate the same failed images
# --------------------------------------------------

results = []

for image_id in IMAGE_IDS:
    print("\n" + "=" * 75)
    print("IMAGE ID:", image_id)
    print("=" * 75)

    image_path = os.path.join(
        IMAGE_DIR, image_id + ".jpg"
    )
    mask_path = os.path.join(
        MASK_DIR, image_id + ".png"
    )

    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        print("Missing or unreadable image:", image_path)
        continue

    mask_raw = cv2.imread(
        mask_path, cv2.IMREAD_GRAYSCALE
    )

    if mask_raw is None:
        print("Missing or unreadable mask:", mask_path)
        continue

    image_rgb = cv2.cvtColor(
        image_bgr, cv2.COLOR_BGR2RGB
    )

    ground_truth = (mask_raw > 127).astype(np.uint8)

    # Same preprocessing as the existing evaluation
    image = image_rgb.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))

    tensor = torch.from_numpy(
        image
    ).unsqueeze(0).to(DEVICE)

    gt_pixels = int(ground_truth.sum())

    print("Ground-truth pixels:", gt_pixels)

    for model_name, model in models.items():

        with torch.no_grad():
            probabilities = torch.sigmoid(model(tensor))

        prob = probabilities.squeeze().cpu().numpy()

        prediction = (
            prob > THRESHOLD
        ).astype(np.uint8)

        pred_pixels = int(prediction.sum())
        intersection = int(
            np.sum(prediction * ground_truth)
        )
        union = int(
            np.sum((prediction + ground_truth) > 0)
        )

        denominator = pred_pixels + gt_pixels

        dice = (
            1.0 if denominator == 0
            else 2.0 * intersection / denominator
        )

        iou = (
            1.0 if union == 0
            else intersection / union
        )

        # Bounding boxes for spatial comparison
        def get_bbox(mask):
            ys, xs = np.where(mask > 0)

            if len(xs) == 0:
                return None

            return (
                int(xs.min()),
                int(ys.min()),
                int(xs.max()),
                int(ys.max())
            )

        gt_bbox = get_bbox(ground_truth)
        pred_bbox = get_bbox(prediction)

        print(f"\nModel: {model_name}")
        print("Maximum probability:", round(float(prob.max()), 6))
        print("Predicted pixels:", pred_pixels)
        print("Intersection:", intersection)
        print("Dice:", round(dice, 4))
        print("IoU:", round(iou, 4))
        print("Ground-truth box:", gt_bbox)
        print("Predicted box:", pred_bbox)

        results.append({
            "image_id": image_id,
            "model": model_name,
            "ground_truth_pixels": gt_pixels,
            "predicted_pixels": pred_pixels,
            "intersection_pixels": intersection,
            "max_probability": float(prob.max()),
            "dice": dice,
            "iou": iou,
            "ground_truth_bbox": str(gt_bbox),
            "predicted_bbox": str(pred_bbox),
            "empty_prediction": pred_pixels == 0,
        })

        # Save prediction and probability map for inspection
        cv2.imwrite(
            os.path.join(
                OUTPUT_DIR,
                f"{image_id}_{model_name.replace('.pth', '')}_prediction.png"
            ),
            prediction * 255
        )

        cv2.imwrite(
            os.path.join(
                OUTPUT_DIR,
                f"{image_id}_{model_name.replace('.pth', '')}_probability.png"
            ),
            np.clip(prob * 255, 0, 255).astype(np.uint8)
        )

# --------------------------------------------------
# 4. Save comparison report
# --------------------------------------------------

if results:
    report = pd.DataFrame(results)

    report_path = os.path.join(
        OUTPUT_DIR, "comparison_report.csv"
    )

    report.to_csv(report_path, index=False)

    print("\n" + "=" * 75)
    print("COMPARISON COMPLETE")
    print("=" * 75)

    print(
        report[
            [
                "image_id",
                "model",
                "predicted_pixels",
                "intersection_pixels",
                "dice",
                "iou",
            ]
        ].to_string(index=False)
    )

    print("\nCSV report:", report_path)
    print("Prediction images:", OUTPUT_DIR)

print("\nNo model weights were modified.")

