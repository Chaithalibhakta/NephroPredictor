
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from unet_model import UNet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

IMAGE_DIR = os.path.join(BASE_DIR, "images")
MASK_DIR = os.path.join(BASE_DIR, "masks")
TEST_CSV = os.path.join(
    PROJECT_DIR, "dataset", "kidney_images", "test.csv"
)
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(
    BASE_DIR, "results", "hybrid_test"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5

# Diagnostic thresholds only; these are not validated clinical settings.
FALLBACK_THRESHOLDS = [0.01, 0.05, 0.10, 0.20, 0.30]

class KidneyDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = os.path.splitext(
            os.path.basename(str(row["image_path"]))
        )[0]

        image_path = os.path.join(IMAGE_DIR, image_id + ".jpg")
        mask_path = os.path.join(MASK_DIR, image_id + ".png")

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise RuntimeError(f"Missing image or mask for {image_id}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        mask = (mask > 127).astype(np.float32)[None, ...]

        return image_id, torch.from_numpy(image), torch.from_numpy(mask)


def load_model(filename):
    model = UNet().to(DEVICE)
    path = os.path.join(MODEL_DIR, filename)
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def dice_iou(pred, target):
    intersection = np.logical_and(pred, target).sum()
    pred_area = pred.sum()
    target_area = target.sum()
    union = pred_area + target_area - intersection

    dice = (
        1.0 if pred_area + target_area == 0
        else 2.0 * intersection / (pred_area + target_area)
    )
    iou = 1.0 if union == 0 else intersection / union

    return float(dice), float(iou)


print("Loading test set and checkpoints...")
df = pd.read_csv(TEST_CSV)
loader = DataLoader(
    KidneyDataset(df), batch_size=1, shuffle=False, num_workers=0
)

primary = load_model("best_unet.pth")
secondary = load_model("best_unet_test.pth")

rows = []

with torch.no_grad():
    for index, (image_ids, images, masks) in enumerate(loader):
        images = images.to(DEVICE)

        p1 = torch.sigmoid(primary(images))[0, 0].cpu().numpy()
        p2 = torch.sigmoid(secondary(images))[0, 0].cpu().numpy()

        pred1 = p1 > THRESHOLD
        pred2 = p2 > THRESHOLD
        target = masks[0, 0].numpy() > 0.5

        d1, i1 = dice_iou(pred1, target)
        d2, i2 = dice_iou(pred2, target)

        row = {
            "image_id": image_ids[0],
            "primary_dice": d1,
            "primary_iou": i1,
            "secondary_dice": d2,
            "secondary_iou": i2,
            "primary_max_probability": float(p1.max()),
        }

        for threshold in FALLBACK_THRESHOLDS:
            # Fallback rule: use secondary only if primary's maximum
            # probability is below the chosen diagnostic threshold.
            use_secondary = p1.max() < threshold
            hybrid_pred = pred2 if use_secondary else pred1

            dh, ih = dice_iou(hybrid_pred, target)

            label = str(threshold).replace(".", "_")
            row[f"hybrid_{label}_dice"] = dh
            row[f"hybrid_{label}_iou"] = ih
            row[f"hybrid_{label}_used_secondary"] = use_secondary

        rows.append(row)

        if (index + 1) % 50 == 0:
            print(f"Processed {index + 1}/{len(loader)} images")

results = pd.DataFrame(rows)
csv_path = os.path.join(OUTPUT_DIR, "hybrid_per_image_results.csv")
results.to_csv(csv_path, index=False)

print("\n========== SUMMARY: DIAGNOSTIC ONLY ==========")
print(f"Images evaluated: {len(results)}")
print(f"Primary mean Dice: {results['primary_dice'].mean():.4f}")
print(f"Secondary mean Dice: {results['secondary_dice'].mean():.4f}")

summary = []

for threshold in FALLBACK_THRESHOLDS:
    label = str(threshold).replace(".", "_")
    dice_col = f"hybrid_{label}_dice"
    iou_col = f"hybrid_{label}_iou"
    used_col = f"hybrid_{label}_used_secondary"

    summary.append({
        "fallback_threshold": threshold,
        "mean_dice": results[dice_col].mean(),
        "mean_iou": results[iou_col].mean(),
        "secondary_used_on_images": int(results[used_col].sum()),
        "images_better_than_primary": int(
            (results[dice_col] > results["primary_dice"] + 1e-12).sum()
        ),
        "images_worse_than_primary": int(
            (results[dice_col] < results["primary_dice"] - 1e-12).sum()
        ),
    })

summary_df = pd.DataFrame(summary)
summary_path = os.path.join(OUTPUT_DIR, "hybrid_summary.csv")
summary_df.to_csv(summary_path, index=False)

print("\n", summary_df.to_string(index=False))
print("\nSaved per-image results:", csv_path)
print("Saved summary:", summary_path)
print("Both original checkpoints remain unchanged.")

