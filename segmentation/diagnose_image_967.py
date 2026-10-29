import os
import cv2
import numpy as np
import pandas as pd
import torch

from unet_model import UNet

BASE_DIR = r"C:\Users\HP\OneDrive\Desktop\NephroPredictorr"

IMAGE_DIR = os.path.join(BASE_DIR, "segmentation", "images")
MASK_DIR = os.path.join(BASE_DIR, "segmentation", "masks")
TEST_CSV = os.path.join(BASE_DIR, "dataset", "kidney_images", "test.csv")

MODEL_PATHS = [
    os.path.join(BASE_DIR, "segmentation", "models", "best_unet.pth"),
    os.path.join(BASE_DIR, "segmentation", "models", "best_unet_test.pth"),
]

IMAGE_ID = "967"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Locate Image 967 in the test CSV
df = pd.read_csv(TEST_CSV)

matches = df[
    df["image_path"].astype(str).apply(
        lambda p: os.path.splitext(os.path.basename(p))[0] == IMAGE_ID
    )
]

if matches.empty:
    raise RuntimeError("Image 967 was not found in the test CSV.")

image_path = os.path.join(IMAGE_DIR, IMAGE_ID + ".jpg")
mask_path = os.path.join(MASK_DIR, IMAGE_ID + ".png")

image = cv2.imread(image_path, cv2.IMREAD_COLOR)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if image is None or mask is None:
    raise RuntimeError("Could not read Image 967 or its mask.")

# Use the same preprocessing as the existing diagnostic
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.astype(np.float32) / 255.0
image = np.transpose(image, (2, 0, 1))

tensor = torch.from_numpy(image).float().unsqueeze(0).to(device)
ground_truth = mask > 127

print("=" * 65)
print("COMPARING BOTH U-NET MODELS ON IMAGE 967")
print("Device:", device)
print("Ground-truth kidney pixels:", int(ground_truth.sum()))
print("=" * 65)

for model_path in MODEL_PATHS:

    print("\nModel:", os.path.basename(model_path))

    if not os.path.exists(model_path):
        print("ERROR: Model file not found.")
        continue

    model = UNet().to(device)

    try:
        state_dict = torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
        model.load_state_dict(state_dict)
    except Exception as error:
        print("Could not load model:", error)
        continue

    model.eval()

    with torch.no_grad():
        logits = model(tensor)
        probabilities = (
            torch.sigmoid(logits)
            .squeeze()
            .cpu()
            .numpy()
        )

    prediction = probabilities > 0.5

    intersection = np.logical_and(
        prediction, ground_truth
    ).sum()

    union = np.logical_or(
        prediction, ground_truth
    ).sum()

    dice = (
        2.0 * intersection
        / (prediction.sum() + ground_truth.sum())
        if (prediction.sum() + ground_truth.sum()) > 0
        else 1.0
    )

    iou = intersection / union if union > 0 else 1.0

    print("Maximum probability:", float(probabilities.max()))
    print("Mean probability:", float(probabilities.mean()))
    print("Predicted kidney pixels:", int(prediction.sum()))
    print("Dice score:", round(float(dice), 4))
    print("IoU score:", round(float(iou), 4))

    del model

print("\n" + "=" * 65)
print("COMPARISON COMPLETE — NO MODEL FILES WERE CHANGED")
print("=" * 65)