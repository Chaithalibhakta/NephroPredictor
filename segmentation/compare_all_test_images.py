
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_DIR = os.path.dirname(BASE_DIR)

IMAGE_DIR = os.path.join(BASE_DIR, "images")
MASK_DIR = os.path.join(BASE_DIR, "masks")

TEST_CSV = os.path.join(
    PROJECT_DIR, "dataset", "kidney_images", "test.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

OUTPUT_DIR = os.path.join(
    BASE_DIR, "results", "all_test_comparison"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATHS = [
    os.path.join(MODEL_DIR, "best_unet.pth"),
    os.path.join(MODEL_DIR, "best_unet_test.pth"),
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

THRESHOLD = 0.5


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

        image_path = os.path.join(
            IMAGE_DIR, image_id + ".jpg"
        )

        mask_path = os.path.join(
            MASK_DIR, image_id + ".png"
        )

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise RuntimeError(f"Cannot read image: {image_path}")

        if mask is None:
            raise RuntimeError(f"Cannot read mask: {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        mask = (mask > 127).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        return (
            image_id,
            torch.from_numpy(image).float(),
            torch.from_numpy(mask).float()
        )


# ============================================================
# EVALUATE ONE CHECKPOINT
# ============================================================

def evaluate_model(model_path, loader):

    model = UNet().to(DEVICE)

    state_dict = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    rows = []

    with torch.no_grad():

        for batch_index, (
            image_ids, images, masks
        ) in enumerate(loader):

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            probabilities = torch.sigmoid(model(images))
            predictions = (probabilities > THRESHOLD).float()

            for i in range(images.shape[0]):

                pred = predictions[i]
                target = masks[i]

                intersection = (pred * target).sum().item()
                pred_area = pred.sum().item()
                target_area = target.sum().item()

                union = pred_area + target_area - intersection

                if pred_area + target_area == 0:
                    dice = 1.0
                else:
                    dice = (
                        2.0 * intersection
                        / (pred_area + target_area)
                    )

                if union == 0:
                    iou = 1.0
                else:
                    iou = intersection / union

                rows.append({
                    "image_id": image_ids[i],
                    "model": os.path.basename(model_path),
                    "ground_truth_pixels": int(target_area),
                    "predicted_pixels": int(pred_area),
                    "intersection_pixels": int(intersection),
                    "max_probability": float(
                        probabilities[i].max().item()
                    ),
                    "dice": float(dice),
                    "iou": float(iou),
                    "empty_prediction": pred_area == 0,
                })

            if (batch_index + 1) % 50 == 0:
                print(
                    f"  Processed {batch_index + 1}/"
                    f"{len(loader)} test images"
                )

    del model

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

print("=" * 65)
print("FULL TEST-SET CHECKPOINT COMPARISON")
print("Device:", DEVICE)
print("Threshold:", THRESHOLD)
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

all_results = []

for model_path in MODEL_PATHS:

    if not os.path.isfile(model_path):
        print("\nMissing checkpoint:", model_path)
        continue

    print("\nEvaluating:", os.path.basename(model_path))

    result_df = evaluate_model(model_path, loader)

    print("Images evaluated:", len(result_df))
    print(f"Mean Dice: {result_df['dice'].mean():.4f}")
    print(f"Mean IoU: {result_df['iou'].mean():.4f}")
    print(
        "Empty predictions:",
        int(result_df["empty_prediction"].sum())
    )

    all_results.append(result_df)


# ============================================================
# SAVE FULL REPORT
# ============================================================

if len(all_results) != 2:
    raise RuntimeError(
        "Both checkpoints must be evaluated successfully."
    )

combined = pd.concat(all_results, ignore_index=True)

combined_path = os.path.join(
    OUTPUT_DIR, "all_image_scores.csv"
)

combined.to_csv(combined_path, index=False)

dice_table = combined.pivot(
    index="image_id",
    columns="model",
    values="dice"
)

iou_table = combined.pivot(
    index="image_id",
    columns="model",
    values="iou"
)

model_a = os.path.basename(MODEL_PATHS[0])
model_b = os.path.basename(MODEL_PATHS[1])

comparison = pd.DataFrame(index=dice_table.index)

comparison["dice_best_unet"] = dice_table[model_a]
comparison["dice_best_unet_test"] = dice_table[model_b]

comparison["iou_best_unet"] = iou_table[model_a]
comparison["iou_best_unet_test"] = iou_table[model_b]

comparison["dice_difference_test_minus_primary"] = (
    comparison["dice_best_unet_test"]
    - comparison["dice_best_unet"]
)

comparison["better_dice_checkpoint"] = np.where(
    comparison["dice_difference_test_minus_primary"] > 1e-12,
    model_b,
    np.where(
        comparison["dice_difference_test_minus_primary"] < -1e-12,
        model_a,
        "Tie"
    )
)

comparison["both_dice_zero"] = (
    (comparison["dice_best_unet"] == 0)
    & (comparison["dice_best_unet_test"] == 0)
)

comparison_path = os.path.join(
    OUTPUT_DIR, "checkpoint_comparison_by_image.csv"
)

comparison.to_csv(comparison_path)

print("\n" + "=" * 65)
print("FINAL COMPARISON")
print("=" * 65)

print("Images compared:", len(comparison))

print(
    f"{model_a} mean Dice:",
    f"{comparison['dice_best_unet'].mean():.4f}"
)

print(
    f"{model_b} mean Dice:",
    f"{comparison['dice_best_unet_test'].mean():.4f}"
)

print(
    "Images where best_unet.pth wins:",
    int(
        (
            comparison["better_dice_checkpoint"] == model_a
        ).sum()
    )
)

print(
    "Images where best_unet_test.pth wins:",
    int(
        (
            comparison["better_dice_checkpoint"] == model_b
        ).sum()
    )
)

print(
    "Tied images:",
    int(
        (
            comparison["better_dice_checkpoint"] == "Tie"
        ).sum()
    )
)

print(
    "Images where both Dice scores are zero:",
    int(comparison["both_dice_zero"].sum())
)

print("\nLowest Dice scores for the primary checkpoint:")
print(
    comparison.sort_values("dice_best_unet")[
        [
            "dice_best_unet",
            "dice_best_unet_test",
            "dice_difference_test_minus_primary"
        ]
    ].head(15).to_string()
)

print("\nReports saved:")
print(combined_path)
print(comparison_path)

print("\nNo model weights were modified.")

