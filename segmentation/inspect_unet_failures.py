
import os
import cv2
import torch
import numpy as np
import pandas as pd

from unet_model import UNet


# 1. Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_unet.pth")
IMAGE_DIR = os.path.join(BASE_DIR, "images")
MASK_DIR = os.path.join(BASE_DIR, "masks")

REPORT_PATH = os.path.join(
    BASE_DIR, "results", "unet_failure_report.csv"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

THRESHOLD = 0.5

# Inspect these five failed images
IMAGE_IDS = ["967", "1771", "1721", "1784", "1342"]


# 2. Load model without modifying its weights
print("Loading existing U-Net...")

model = UNet().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint)
model.eval()

print("Model loaded. Device:", DEVICE)


# 3. Load failure report
report = pd.read_csv(REPORT_PATH)
report["image_id"] = report["image_id"].astype(str).str.replace(
    r"\.0$", "", regex=True
)


# 4. Inspect each image
for image_id in IMAGE_IDS:

    print("\n" + "=" * 65)
    print("IMAGE ID:", image_id)
    print("=" * 65)

    image_path = os.path.join(
        IMAGE_DIR, image_id + ".jpg"
    )
    mask_path = os.path.join(
        MASK_DIR, image_id + ".png"
    )

    if not os.path.exists(image_path):
        print("ERROR: Ultrasound image missing:", image_path)
        continue

    if not os.path.exists(mask_path):
        print("ERROR: Ground-truth mask missing:", mask_path)
        continue

    # Read image and ground-truth mask
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    gt_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    gt = (gt_raw > 127).astype(np.uint8)

    print("Original prepared image shape:", image_rgb.shape)
    print("Ground-truth mask shape:", gt_raw.shape)
    print("Ground-truth raw pixel values:", np.unique(gt_raw))

    # Run model
    image = image_rgb.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    tensor = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probabilities = torch.sigmoid(model(tensor))

    prob = probabilities.squeeze().cpu().numpy()
    pred = (prob > THRESHOLD).astype(np.uint8)

    print("\nGround-truth kidney pixels:", int(gt.sum()))
    print("Predicted kidney pixels:", int(pred.sum()))
    print("Maximum prediction probability:", float(prob.max()))
    print("Mean prediction probability:", float(prob.mean()))

    # Calculate overlap
    intersection = int(np.sum(gt * pred))
    union = int(np.sum((gt + pred) > 0))

    denominator = int(gt.sum() + pred.sum())

    dice = 1.0 if denominator == 0 else (
        2.0 * intersection / denominator
    )
    iou = 1.0 if union == 0 else intersection / union

    print("\nIntersection pixels:", intersection)
    print("Union pixels:", union)
    print("Recalculated Dice:", round(dice, 4))
    print("Recalculated IoU:", round(iou, 4))

    # Calculate bounding boxes: x_min, y_min, x_max, y_max
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

    gt_bbox = get_bbox(gt)
    pred_bbox = get_bbox(pred)

    print("\nGround-truth bounding box:", gt_bbox)
    print("Predicted bounding box:", pred_bbox)

    if gt_bbox is not None and pred_bbox is not None:
        gx1, gy1, gx2, gy2 = gt_bbox
        px1, py1, px2, py2 = pred_bbox

        boxes_overlap = (
            max(gx1, px1) <= min(gx2, px2)
            and max(gy1, py1) <= min(gy2, py2)
        )

        print("Bounding boxes overlap:", boxes_overlap)

    # Compare with saved report
    matching_row = report[report["image_id"] == image_id]

    if not matching_row.empty:
        saved_dice = matching_row.iloc[0]["dice_score"]
        saved_iou = matching_row.iloc[0]["iou_score"]

        print("\nSaved report Dice:", saved_dice)
        print("Saved report IoU:", saved_iou)

print("\n" + "=" * 65)
print("INSPECTION COMPLETE")
print("No model weights were modified.")
print("=" * 65)

