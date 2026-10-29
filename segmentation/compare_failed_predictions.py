
import os
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from unet_model import UNet

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

IMAGE_DIR = os.path.join(BASE_DIR, "images")
MASK_DIR = os.path.join(BASE_DIR, "masks")
MODEL_DIR = os.path.join(BASE_DIR, "models")

REPORT_PATH = os.path.join(
    BASE_DIR,
    "results",
    "all_test_comparison",
    "checkpoint_comparison_by_image.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "checkpoint_failure_comparison"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINTS = {
    "Primary": os.path.join(MODEL_DIR, "best_unet.pth"),
    "Second": os.path.join(MODEL_DIR, "best_unet_test.pth"),
}


def load_model(path):
    model = UNet().to(DEVICE)
    state = torch.load(
        path,
        map_location=DEVICE,
        weights_only=True
    )
    model.load_state_dict(state)
    model.eval()
    return model


def load_image_and_mask(image_id):
    image_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")
    mask_path = os.path.join(MASK_DIR, f"{image_id}.png")

    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image_bgr is None or mask is None:
        raise RuntimeError(f"Could not load image or mask for {image_id}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    tensor = image_rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = torch.from_numpy(tensor).float().unsqueeze(0).to(DEVICE)

    ground_truth = (mask > 127).astype(np.uint8)

    return image_rgb, tensor, ground_truth


def predict(model, tensor):
    with torch.no_grad():
        probabilities = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()

    return (probabilities > 0.5).astype(np.uint8)


print("Loading checkpoint comparison report...")

comparison = pd.read_csv(REPORT_PATH)

comparison = comparison.sort_values(
    "dice_best_unet",
    ascending=True
).head(10)

print("\nImages selected for visual inspection:")
print(
    comparison[
        [
            "image_id",
            "dice_best_unet",
            "dice_best_unet_test"
        ]
    ].to_string(index=False)
)

print("\nLoading models...")

primary_model = load_model(CHECKPOINTS["Primary"])
second_model = load_model(CHECKPOINTS["Second"])

for _, row in comparison.iterrows():

    image_id = str(row["image_id"])

    # Remove a trailing .0 if the CSV read numeric IDs as floats
    if image_id.endswith(".0"):
        image_id = image_id[:-2]

    image_rgb, tensor, ground_truth = load_image_and_mask(image_id)

    primary_prediction = predict(primary_model, tensor)
    second_prediction = predict(second_model, tensor)

    def calculate_dice(pred, target):
        intersection = np.logical_and(pred, target).sum()
        denominator = pred.sum() + target.sum()

        if denominator == 0:
            return 1.0

        return 2.0 * intersection / denominator

    primary_dice = calculate_dice(primary_prediction, ground_truth)
    second_dice = calculate_dice(second_prediction, ground_truth)

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    axes[0].imshow(image_rgb)
    axes[0].set_title(f"Ultrasound {image_id}")

    axes[1].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")

    axes[2].imshow(primary_prediction, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Primary Dice: {primary_dice:.3f}")

    axes[3].imshow(second_prediction, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title(f"Second Dice: {second_dice:.3f}")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"comparison_{image_id}.png"
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", output_path)

print("\nFinished.")
print("Visualizations saved in:")
print(OUTPUT_DIR)

print("\nNeither checkpoint was modified.")

