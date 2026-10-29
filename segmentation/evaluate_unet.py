import os
import cv2
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from unet_model import UNet

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\HP\OneDrive\Desktop\NephroPredictorr"

IMAGE_DIR = os.path.join(BASE_DIR, "segmentation", "images")
MASK_DIR = os.path.join(BASE_DIR, "segmentation", "masks")
TEST_CSV = os.path.join(BASE_DIR, "dataset", "kidney_images", "test.csv")
MODEL_DIR = os.path.join(BASE_DIR, "segmentation", "models")

MODEL_PATHS = [
    os.path.join(MODEL_DIR, "best_unet.pth"),
    os.path.join(MODEL_DIR, "best_unet_test.pth"),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DATASET
# ============================================================

class KidneyTestDataset(Dataset):

    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_name = os.path.basename(str(row["image_path"]))
        image_id = os.path.splitext(image_name)[0]

        image_path = os.path.join(IMAGE_DIR, image_id + ".jpg")
        mask_path = os.path.join(MASK_DIR, image_id + ".png")

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise RuntimeError(f"Cannot read image: {image_path}")

        if mask is None:
            raise RuntimeError(f"Cannot read mask: {mask_path}")

        # Same preprocessing as the existing evaluation
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        mask = (mask > 127).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        return (
            torch.from_numpy(image).float(),
            torch.from_numpy(mask).float()
        )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model_path, loader):

    model = UNet().to(device)

    try:
        state_dict = torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
        model.load_state_dict(state_dict)
    except Exception:
        del model
        raise

    model.eval()

    dice_scores = []
    iou_scores = []
    empty_predictions = 0

    with torch.no_grad():

        for batch_index, (images, masks) in enumerate(loader):

            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            predictions = (torch.sigmoid(logits) > 0.5).float()

            # Calculate metrics separately for each image
            for i in range(images.shape[0]):

                pred = predictions[i]
                target = masks[i]

                intersection = (pred * target).sum().item()
                pred_area = pred.sum().item()
                target_area = target.sum().item()
                union = pred_area + target_area - intersection

                # If both masks are empty, Dice and IoU are 1.
                if pred_area + target_area == 0:
                    dice = 1.0
                else:
                    dice = (2.0 * intersection) / (
                        pred_area + target_area
                    )

                if union == 0:
                    iou = 1.0
                else:
                    iou = intersection / union

                dice_scores.append(dice)
                iou_scores.append(iou)

                if pred_area == 0:
                    empty_predictions += 1

            if (batch_index + 1) % 100 == 0:
                print(
                    f"  Evaluated {batch_index + 1}/{len(loader)} batches"
                )

    result = {
        "images": len(dice_scores),
        "mean_dice": float(np.mean(dice_scores)),
        "mean_iou": float(np.mean(iou_scores)),
        "empty_predictions": empty_predictions,
    }

    del model
    return result


# ============================================================
# MAIN
# ============================================================

print("=" * 65)
print("COMPARING BOTH U-NET MODELS ON THE SAME TEST SET")
print("Device:", device)
print("=" * 65)

test_df = pd.read_csv(TEST_CSV)
print("Test CSV rows:", len(test_df))

dataset = KidneyTestDataset(test_df)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

results = []

for model_path in MODEL_PATHS:

    print("\nModel:", os.path.basename(model_path))

    if not os.path.exists(model_path):
        print("ERROR: Model file not found.")
        continue

    result = evaluate_model(model_path, loader)
    result["model"] = os.path.basename(model_path)
    results.append(result)

    print("Images evaluated:", result["images"])
    print(f"Mean Dice: {result['mean_dice']:.4f}")
    print(f"Mean Dice (%): {result['mean_dice'] * 100:.2f}%")
    print(f"Mean IoU: {result['mean_iou']:.4f}")
    print(f"Mean IoU (%): {result['mean_iou'] * 100:.2f}%")
    print("Empty predictions:", result["empty_predictions"])

print("\n" + "=" * 65)
print("FINAL COMPARISON")
print("=" * 65)

for result in results:
    print(
        f"{result['model']}: "
        f"Dice={result['mean_dice']:.4f}, "
        f"IoU={result['mean_iou']:.4f}, "
        f"Empty predictions={result['empty_predictions']}"
    )

if len(results) == 2:
    print("\nComparison based on mean Dice:")
    better = max(results, key=lambda r: r["mean_dice"])
    print("Higher mean Dice:", better["model"])

print("\nNo model weights were modified.")
print("=" * 65)